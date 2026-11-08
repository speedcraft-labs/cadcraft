//! Initial display-list builds, edits and view changes run on workers so rebuilding never holds
//! up the UI frame.
//!
//! The canvas keeps drawing its current list while a worker builds a replacement. This covers
//! the first list, document edits and view changes, so a large drawing doesn't hold up the input
//! frame that opened or changed it. A new build for the same view cancels the one in flight
//! (dropping its [`Job`] sets the renderer's cancel flag) and starts one for the latest state.
//! The web build has no threads and rebuilds synchronously.

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

use cadcraft_color::Rgb;
use cadcraft_doc::{Drawing, Space};
use cadcraft_geom::{Bounds2, Vec2};
use cadcraft_render::{DisplayList, Kind, Options};
use craft_jobs::{JobEvent, JobHandle};
#[cfg(not(target_arch = "wasm32"))]
use craft_jobs::{JobSystem, Priority};

/// What a build needs: the drawing (shared, never copied), the space and the options.
pub(super) struct Request {
    pub doc: Arc<Drawing>,
    pub space: Space,
    pub opts: Options,
    /// Also convert the list to GPU vertex data (GPU canvas only).
    pub mesh: Option<MeshRequest>,
}

/// How to convert the list for the GPU: the view the vertex origin is chosen for and the
/// background the colours resolve against (paper when the list is a layout's sheet).
pub(super) struct MeshRequest {
    pub view_center: Vec2,
    pub view_height: f64,
    pub model_bg: Rgb,
    pub paper_bg: Rgb,
    pub lwdisplay: bool,
    /// Create the GPU buffers on the worker too, so the frame that swaps the list in only swaps
    /// buffers (`None`: the paint callback uploads them).
    pub device: Option<egui_wgpu::wgpu::Device>,
}

/// A finished build.
pub(super) struct Built {
    pub list: DisplayList,
    pub build_ms: f64,
    pub mesh: Option<BuiltMesh>,
}

pub(super) struct BuiltMesh {
    pub data: crate::gpu::MeshData,
    pub origin: Vec2,
    pub bg: Rgb,
    pub lwdisplay: bool,
    /// Indices of the infinite lines (drawn on the CPU in GPU mode).
    pub infinite: Vec<usize>,
    pub ms: f64,
}

/// A build in flight: the list key and area it is for, and where its result arrives. Dropping
/// it cancels the build.
pub(super) struct Job {
    pub key: super::ListKey,
    pub window: Option<Bounds2>,
    handle: JobHandle<Built>,
}

/// What [`Job::poll`] found.
pub(super) enum Poll {
    Pending,
    Done(Box<Built>),
    /// The worker died without a result (it panicked).
    Lost,
}

impl Job {
    pub fn poll(&self) -> Poll {
        match self.handle.try_recv() {
            Ok(JobEvent::Progress(_)) | Err(std::sync::mpsc::TryRecvError::Empty) => Poll::Pending,
            Ok(JobEvent::Completed(Ok(built))) => Poll::Done(Box::new(built)),
            Ok(JobEvent::Completed(Err(_))) | Err(std::sync::mpsc::TryRecvError::Disconnected) => Poll::Lost,
        }
    }

    /// Whether this build serves list `key` for the visible area `visible`.
    pub fn serves(&self, key: &super::ListKey, visible: &Bounds2) -> bool {
        self.key == *key && self.window.is_none_or(|w| visible.is_empty() || w.contains_box(visible))
    }
}

/// Build `req` on the calling thread; `None` once it has been cancelled (the partial list is
/// thrown away, and no vertex data is made for it).
fn run(req: Request) -> Option<Built> {
    let t0 = crate::now_ms();
    let list = cadcraft_render::build(&req.doc, &req.space, &req.opts);
    if req.opts.cancelled() {
        return None;
    }
    let build_ms = crate::now_ms() - t0;
    let mesh = match req.mesh {
        None => None,
        Some(m) => {
            let t1 = crate::now_ms();
            let origin = crate::gpu::choose_origin(&list.bounds, m.view_center, m.view_height);
            let bg = if list.sheet.is_some() { m.paper_bg } else { m.model_bg };
            let mut data = crate::gpu::build_mesh(&list, origin, bg, m.lwdisplay);
            if req.opts.cancelled() {
                return None;
            }
            if let Some(device) = &m.device {
                data.upload_now(device);
            }
            let infinite = list.prims.iter().enumerate().filter(|(_, p)| matches!(p.kind, Kind::Infinite { .. })).map(|(i, _)| i).collect();
            Some(BuiltMesh { data, origin, bg, lwdisplay: m.lwdisplay, infinite, ms: crate::now_ms() - t1 })
        }
    };
    Some(Built { list, build_ms, mesh })
}

/// Start building `req` on a worker thread for `key` / `window`; the worker asks `ctx` for a
/// frame when it is done. `None` when no native worker could be started; the caller keeps its
/// current list. The wasm path calls the synchronous renderer directly.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn spawn(mut req: Request, key: super::ListKey, window: Option<Bounds2>, ctx: &egui::Context) -> Option<Job> {
    static JOBS: OnceLock<JobSystem> = OnceLock::new();
    let jobs = JOBS.get_or_init(|| JobSystem::new(2));
    if !jobs.is_available() {
        log::warn!("canvas: no craft-jobs worker could be started; keeping the current display list");
        return None;
    }
    let ctx = ctx.clone();
    let key_name = format!("cadcraft-display-list-{}-{}", key.0, key.7);
    let handle = jobs.submit_latest(key_name, Priority::Interactive, move |job_ctx| {
        req.opts.cancel = Some(job_ctx.cancellation().atomic_flag());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(req)));
        match result {
            Ok(Some(built)) => {
                ctx.request_repaint();
                job_ctx.report_progress(100);
                Ok(built)
            }
            Ok(None) => Err(craft_jobs::JobError::Cancelled),
            Err(_) => {
                ctx.request_repaint();
                Err(craft_jobs::JobError::Failed("display-list build panicked".into()))
            }
        }
    });
    Some(Job { key, window, handle })
}

#[cfg(target_arch = "wasm32")]
pub(super) fn spawn(_req: Request, _key: super::ListKey, _window: Option<Bounds2>, _ctx: &egui::Context) -> Option<Job> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn worker_build_matches_a_synchronous_one() {
        use cadcraft_geom::Vec3;
        let mut d = Drawing::new_imperial();
        let line = cadcraft_doc::Line { a: Vec3::new(0.0, 0.0, 0.0), b: Vec3::new(10.0, 5.0, 0.0) };
        let Ok(_) = d.add(&Space::Model, cadcraft_doc::Common::default(), cadcraft_doc::EntityKind::Line(line)) else { panic!("add") };
        let doc = Arc::new(d);
        let opts = Options::default();
        let sync = cadcraft_render::build(&doc, &Space::Model, &opts);
        let req = Request {
            doc: doc.clone(),
            space: Space::Model,
            opts: opts.clone(),
            mesh: Some(MeshRequest {
                view_center: Vec2::ZERO,
                view_height: 20.0,
                model_bg: Rgb(0, 0, 0),
                paper_bg: Rgb(255, 255, 255),
                lwdisplay: false,
                device: None,
            }),
        };
        let ctx = egui::Context::default();
        let Some(job) = spawn(req, Default::default(), None, &ctx) else { panic!("no worker") };
        let built = loop {
            match job.poll() {
                Poll::Done(b) => break b,
                Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(2)),
                Poll::Lost => panic!("worker lost"),
            }
        };
        assert_eq!(built.list.prims.len(), sync.prims.len());
        assert_eq!(built.list.verts, sync.verts);
        let Some(mesh) = built.mesh else { panic!("no mesh") };
        assert_eq!(mesh.bg, Rgb(0, 0, 0));
        assert!(mesh.data.line_vertices() >= 2);
    }

    #[test]
    fn cancelled_run_makes_nothing() {
        let opts = Options { cancel: Some(Arc::new(AtomicBool::new(true))), ..Default::default() };
        let req = Request { doc: Arc::new(Drawing::new_imperial()), space: Space::Model, opts, mesh: None };
        assert!(run(req).is_none());
    }

    #[test]
    fn dropping_a_job_handle_requests_cancellation() {
        let jobs = JobSystem::new(1);
        let (token_tx, token_rx) = std::sync::mpsc::channel();
        let handle = jobs.submit(Priority::Interactive, move |ctx| {
            token_tx.send(ctx.cancellation()).ok();
            Ok(())
        });
        let token = token_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("worker should start");
        drop(handle);
        assert!(token.is_cancelled());
    }
}
