//! Input-to-frame latency for the first interaction while a real engineering drawing is loaded.

use cadcraft_engine::Session;
use cadcraft_ui_egui::{CadApp, Services};
use egui_kittest::Harness;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{Duration, Instant};

fn app(path: &Path) -> CadApp {
    let bytes = std::fs::read(path).expect("read pinned engineering drawing");
    let drawing = cadcraft_io::read(&bytes, path).expect("parse pinned engineering drawing");
    let mut session = Session::new();
    session.viewport_px = (1400.0, 900.0);
    session.open_drawing(drawing, "CADCraft Heavy Engineering sample", Some(path.to_string_lossy().into_owned()));
    CadApp::new(session, Services::default())
}

fn main() {
    let input = std::env::args().nth(1).expect("corpus drawing path argument");
    let path = Path::new(&input);
    let app = app(path);
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_ui_state(
        |ui, app: &mut CadApp| {
            app.logic(ui.ctx());
            app.ui(ui);
        },
        app,
    );
    let before = h.state().session.state().expect("drawing").view().height;
    // The drawing occupies the center of the harness window. Queue a real viewport input before
    // the first frame, when its initial display-list build must be admitted.
    h.event(egui::Event::PointerMoved(egui::pos2(700.0, 450.0)));
    h.event(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Line,
        delta: egui::vec2(0.0, -3.0),
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::NONE,
        source: egui::MouseWheelSource::Unknown,
    });
    let start = Instant::now();
    h.step();
    let latency = start.elapsed();
    let deadline = start + Duration::from_secs(120);
    while h.state().canvas.list.is_none() || h.state().canvas.building() {
        assert!(Instant::now() < deadline, "input-triggered display-list rebuild timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(1));
    }
    // The timed frame is the first response to input; wait for the final list only for the output
    // equality check below, outside the latency measurement.
    let after = h.state().session.state().expect("drawing").view().height;
    assert!((after - before).abs() > 1e-6, "wheel input did not zoom the viewport");
    let list = h.state().canvas.list.as_ref().expect("rebuilt display list");
    assert!(!list.prims.is_empty(), "loaded display list is empty");
    // Display-list primitive order and raster-image indices can vary across processes because
    // the drawing engine builds several parts from hash maps. Object handles can also be
    // reassigned across revisions, so compare rendered content independent of those IDs.
    let mut primitives = Vec::with_capacity(list.prims.len());
    for prim in &list.prims {
        let mut fingerprint = Sha256::new();
        fingerprint.update([prim.color.0, prim.color.1, prim.color.2, prim.transparency]);
        fingerprint.update([u8::from(prim.aci7)]);
        fingerprint.update(prim.lw.to_bits().to_le_bytes());
        fingerprint.update(prim.len.to_le_bytes());
        match prim.kind {
            cadcraft_render::Kind::Polyline => fingerprint.update([0]),
            cadcraft_render::Kind::Tris => fingerprint.update([1]),
            cadcraft_render::Kind::Mask => fingerprint.update([2]),
            cadcraft_render::Kind::Point => fingerprint.update([3]),
            cadcraft_render::Kind::Infinite { ray } => fingerprint.update([4, u8::from(ray)]),
            cadcraft_render::Kind::Image { index } => {
                fingerprint.update([5]);
                let image = &list.images[index as usize];
                fingerprint.update(image.raster.width().to_le_bytes());
                fingerprint.update(image.raster.height().to_le_bytes());
                fingerprint.update(image.raster.rgba());
                fingerprint.update([image.fade]);
                fingerprint.update((image.clips.len() as u64).to_le_bytes());
                for clip in &image.clips {
                    fingerprint.update((clip.len() as u64).to_le_bytes());
                    for point in clip {
                        fingerprint.update(point.x.to_bits().to_le_bytes());
                        fingerprint.update(point.y.to_bits().to_le_bytes());
                    }
                }
            }
        }
        let points = list.points(prim);
        if matches!(prim.kind, cadcraft_render::Kind::Tris | cadcraft_render::Kind::Mask) {
            // Triangle order and winding have no effect on the filled output, but hatch
            // tessellation may emit them in a different order across processes.
            let mut triangles = points
                .chunks_exact(3)
                .map(|triangle| {
                    let mut vertices: [(u64, u64); 3] = std::array::from_fn(|i| (triangle[i].x.to_bits(), triangle[i].y.to_bits()));
                    vertices.sort_unstable();
                    vertices
                })
                .collect::<Vec<_>>();
            triangles.sort_unstable();
            for triangle in triangles {
                for (x, y) in triangle {
                    fingerprint.update(x.to_le_bytes());
                    fingerprint.update(y.to_le_bytes());
                }
            }
        } else {
            for point in points {
                fingerprint.update(point.x.to_bits().to_le_bytes());
                fingerprint.update(point.y.to_bits().to_le_bytes());
            }
        }
        let mut primitive = [0; 32];
        primitive.copy_from_slice(&fingerprint.finalize());
        primitives.push(primitive);
    }
    primitives.sort_unstable();
    let mut digest = Sha256::new();
    for primitive in primitives {
        digest.update(primitive);
    }

    println!("CRAFT_JOURNEY_NS={}", latency.as_nanos());
    println!("CRAFT_OUTPUT_SHA256={:x}", digest.finalize());
}
