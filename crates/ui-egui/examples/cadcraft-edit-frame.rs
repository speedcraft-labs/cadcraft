use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cadcraft_doc::{EntityKind, Space};
use cadcraft_geom::{Bounds2, Vec2};
use cadcraft_render::{DisplayList, Options, build, build_cached};
use cadcraft_ui_egui::gpu::{self, CanvasCallback, GpuFaults, LAYER_DRAWING, MeshSlot, Stencil};
use egui_wgpu::wgpu;
use egui_wgpu::{CallbackTrait, ScreenDescriptor};

const WIDTH: u32 = 1600;
const HEIGHT: u32 = 1000;
const BACKGROUND: cadcraft_color::Rgb = cadcraft_color::Rgb(24, 24, 24);

fn stable_hash(debug: &impl std::fmt::Debug) -> u64 {
    struct Fnv(u64);
    impl std::fmt::Write for Fnv {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            for byte in value.bytes() {
                self.0 ^= u64::from(byte);
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
            }
            Ok(())
        }
    }
    let mut hash = Fnv(0xcbf2_9ce4_8422_2325);
    let _ = std::fmt::write(&mut hash, format_args!("{debug:?}"));
    hash.0
}

fn same_list(a: &DisplayList, b: &DisplayList) -> bool {
    a.prims == b.prims
        && a.verts == b.verts
        && a.tris == b.tris
        && a.images == b.images
        && a.prim_layers == b.prim_layers
        && a.texts == b.texts
        && a.pens == b.pens
        && a.bounds == b.bounds
        && a.expanded == b.expanded
        && a.truncated == b.truncated
}

fn list_summary(list: &DisplayList) -> String {
    format!(
        "prims={}/{} verts={}/{} tris={}/{} images={}/{} layers={}/{} texts={}/{} pens={}/{} bounds={:?} expanded={} truncated={}",
        list.prims.len(),
        stable_hash(&list.prims),
        list.verts.len(),
        stable_hash(&list.verts),
        list.tris.len(),
        stable_hash(&list.tris),
        list.images.len(),
        stable_hash(&list.images),
        list.prim_layers.len(),
        stable_hash(&list.prim_layers),
        list.texts.len(),
        stable_hash(&list.texts),
        list.pens.len(),
        stable_hash(&list.pens),
        list.bounds,
        list.expanded,
        list.truncated
    )
}

fn edit_options(drawing: &cadcraft_doc::Drawing) -> Result<Options, Box<dyn Error>> {
    let extents = drawing.extents(&Space::Model);
    let height = extents.width().max(extents.height());
    if extents.is_empty() || !height.is_finite() || height <= 0.0 {
        return Err("drawing has empty extents".into());
    }
    let pixel = 2f64.powi((height / 1000.0).max(1e-300).log2().floor() as i32);
    let half = Vec2::new(height * 0.8, height * 0.5);
    let visible = Bounds2::new(extents.center() - half, extents.center() + half);
    Ok(Options { tolerance: pixel * 0.5, min_dash: pixel * 2.0, view: Some(visible.expand(height * 0.8)), ..Default::default() })
}

fn nvidia_state() -> egui_wgpu::RenderState {
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = wgpu::Backends::VULKAN;
    setup.native_adapter_selector = Some(Arc::new(|adapters, _surface| {
        adapters
            .iter()
            .find(|adapter| adapter.get_info().vendor == 0x10de && adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .cloned()
            .ok_or_else(|| "NVIDIA discrete Vulkan adapter is unavailable".to_owned())
    }));
    egui_kittest::wgpu::create_render_state(egui_wgpu::WgpuSetup::CreateNew(setup), egui_wgpu::RendererOptions::PREDICTABLE)
}

fn readback_hash(rs: &egui_wgpu::RenderState, texture: &wgpu::Texture) -> Result<u64, Box<dyn Error>> {
    let bytes_per_row = WIDTH * 4;
    let padded_row = bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = rs.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cadcraft edit frame readback"),
        size: u64::from(padded_row) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = rs.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cadcraft frame readback") });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row), rows_per_image: Some(HEIGHT) },
        },
        wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
    );
    rs.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    rs.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(20)) })?;
    rx.recv()?.map_err(|e| std::io::Error::other(format!("readback map failed: {e}")))?;
    let mapped = buffer.slice(..).get_mapped_range()?;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for row in mapped.chunks_exact(padded_row as usize).take(HEIGHT as usize) {
        for byte in &row[..bytes_per_row as usize] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    drop(mapped);
    buffer.unmap();
    Ok(hash)
}

fn render_callback(
    rs: &egui_wgpu::RenderState,
    texture: &wgpu::TextureView,
    screen: &ScreenDescriptor,
    callback: &CanvasCallback,
    faults: &GpuFaults,
) -> Result<(Duration, Duration), Box<dyn Error>> {
    let mut encoder = rs.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cadcraft edit frame") });
    let mut renderer = rs.renderer.write();
    let upload_started = Instant::now();
    let _ = callback.prepare(&rs.device, &rs.queue, screen, &mut encoder, &mut renderer.callback_resources);
    let upload_time = upload_started.elapsed();
    let draw_started = Instant::now();
    {
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cadcraft edit frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: texture,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                ..Default::default()
            })
            .forget_lifetime();
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(WIDTH as f32, HEIGHT as f32));
        let info = egui::PaintCallbackInfo { viewport: rect, clip_rect: rect, pixels_per_point: 1.0, screen_size_px: [WIDTH, HEIGHT] };
        callback.paint(info, &mut pass, &renderer.callback_resources);
    }
    rs.queue.submit([encoder.finish()]);
    rs.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(60)) })?;
    let draw_time = draw_started.elapsed();
    drop(renderer);
    if let Some(fault) = faults.take() {
        return Err(format!("GPU callback fault: {}", fault.message).into());
    }
    Ok((upload_time, draw_time))
}

fn run(path: &Path) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let drawing = cadcraft_io::read(&bytes, path)?;
    let options = edit_options(&drawing)?;
    let store = drawing.space(&Space::Model).ok_or("drawing has no model space")?;
    let handle = store
        .iter()
        .find(|entity| matches!(entity.kind, EntityKind::Hatch(_)))
        .map(|entity| entity.handle)
        .or_else(|| store.iter().last().map(|entity| entity.handle))
        .ok_or("drawing has no editable entity")?;
    let mode = std::env::var("CADCRAFT_PERF_EDIT_MODE").unwrap_or_else(|_| "incremental".into());
    if mode != "full" && mode != "incremental" {
        return Err(format!("unknown CADCRAFT_PERF_EDIT_MODE={mode:?}").into());
    }
    let rs = nvidia_state();
    let adapter = rs.adapter.get_info();
    let faults = Arc::new(GpuFaults::default());
    gpu::install(&rs, &faults, None);
    let extent = wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 };
    let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cadcraft entity edit frame"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: rs.target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let screen = ScreenDescriptor { size_in_pixels: [WIDTH, HEIGHT], pixels_per_point: 1.0 };
    let extents = drawing.extents(&Space::Model);
    let scale = (WIDTH as f64 / extents.width()).min(HEIGHT as f64 / extents.height());
    let center = extents.center();

    // Load and draw the original document once, outside the measured edit frame.
    let base_list = Arc::new(build(&drawing, &Space::Model, &options));
    let mut cache = cadcraft_render::EntityDisplayCache::default();
    let (initial_mesh, ranges) = if mode == "incremental" {
        let cached = build_cached(&drawing, &Space::Model, &options, &mut cache);
        if !same_list(&base_list, &cached) {
            return Err(format!("full and cached base lists differ: full [{}], cached [{}]", list_summary(&base_list), list_summary(&cached)).into());
        }
        gpu::build_mesh_entities(&cached, cache.fragments(), gpu::choose_origin(&cached.bounds, center, HEIGHT as f64 / scale), BACKGROUND, false)
    } else {
        (gpu::build_mesh(&base_list, gpu::choose_origin(&base_list.bounds, center, HEIGHT as f64 / scale), BACKGROUND, false), Default::default())
    };
    let base_origin = gpu::choose_origin(&base_list.bounds, center, HEIGHT as f64 / scale);
    let slot: MeshSlot = Arc::new(Mutex::new(Some(initial_mesh)));
    let base_callback = CanvasCallback {
        key: 1,
        slot: slot.clone(),
        offset: [(base_origin.x - center.x) as f32, (base_origin.y - center.y) as f32],
        scale: scale as f32,
        center: [WIDTH as f32 * 0.5, HEIGHT as f32 * 0.5],
        layer: LAYER_DRAWING,
        xform: gpu::IDENTITY_XFORM,
        dash: 0.0,
        stencil: Stencil::Off,
    };
    let _ = render_callback(&rs, &view, &screen, &base_callback, &faults)?;

    let mut edited = drawing.clone();
    let store = edited.space_mut(&Space::Model).ok_or("edited drawing has no model space")?;
    if !store.modify(handle, |entity| {
        if let EntityKind::Hatch(hatch) = &mut entity.kind {
            hatch.scale *= 1.1;
        } else {
            entity.common.color = match entity.common.color {
                cadcraft_doc::color::Color::Index(1) => cadcraft_doc::color::Color::Index(2),
                _ => cadcraft_doc::color::Color::Index(1),
            };
        }
    }) {
        return Err("could not edit selected entity".into());
    }

    let started = Instant::now();
    let list_started = Instant::now();
    let list =
        if mode == "full" { Arc::new(build(&edited, &Space::Model, &options)) } else { build_cached(&edited, &Space::Model, &options, &mut cache) };
    let list_time = list_started.elapsed();
    let mesh_started = Instant::now();
    let origin = gpu::choose_origin(&list.bounds, center, HEIGHT as f64 / scale);
    let (mesh, update_path) = if mode == "full" {
        (gpu::build_mesh(&list, origin, BACKGROUND, false), "full mesh rebuild")
    } else {
        let patch = cache.take_gpu_changes().and_then(|changed| {
            let [changed] = changed.as_slice() else { return None };
            let range = *ranges.get(changed)?;
            let fragment = cache.fragment(*changed)?;
            gpu::build_mesh_patch(fragment, origin, BACKGROUND, false, 1, range, list.clone())
        });
        if let Some(patch) = patch {
            (patch, "single-entity GPU patch")
        } else if cache.has_cached_fragments() {
            let (full, _) = gpu::build_mesh_entities(&list, cache.fragments(), origin, BACKGROUND, false);
            (full, "full mesh fallback")
        } else {
            (gpu::build_mesh(&list, origin, BACKGROUND, false), "full mesh fallback")
        }
    };
    let mesh_time = mesh_started.elapsed();
    *slot.lock().map_err(|_| std::io::Error::other("GPU mesh slot poisoned"))? = Some(mesh);
    let callback = CanvasCallback {
        key: 2,
        slot: slot.clone(),
        offset: [(origin.x - center.x) as f32, (origin.y - center.y) as f32],
        scale: scale as f32,
        center: [WIDTH as f32 * 0.5, HEIGHT as f32 * 0.5],
        layer: LAYER_DRAWING,
        xform: gpu::IDENTITY_XFORM,
        dash: 0.0,
        stencil: Stencil::Off,
    };
    let (upload_time, draw_time) = render_callback(&rs, &view, &screen, &callback, &faults)?;
    let elapsed = started.elapsed();

    let full = build(&edited, &Space::Model, &options);
    if !same_list(&full, &list) {
        return Err(format!("full and incremental display lists differ: full [{}], cached [{}]", list_summary(&full), list_summary(&list)).into());
    }
    let pixels = readback_hash(&rs, &texture)?;
    let output = stable_hash(&full);
    println!("GPU adapter: {} ({:?}, vendor {:04x})", adapter.name, adapter.device_type, adapter.vendor);
    println!("drawing: {} entities; edited handle {:?}; list primitives {}; update {update_path}", drawing.entity_count(), handle, list.prims.len());
    println!(
        "frame stages: list {:.2} ms, mesh {:.2} ms, upload {:.2} ms, draw/wait {:.2} ms",
        list_time.as_secs_f64() * 1000.0,
        mesh_time.as_secs_f64() * 1000.0,
        upload_time.as_secs_f64() * 1000.0,
        draw_time.as_secs_f64() * 1000.0
    );
    println!(
        "entity edit GPU frame: {mode} {:.2} ms (list + mesh + upload + completed draw); outputs identical=true",
        elapsed.as_secs_f64() * 1000.0
    );
    println!("CRAFT_JOURNEY_NS={}", elapsed.as_nanos());
    println!("CRAFT_OUTPUT_HASH={output:016x}{pixels:016x}");
    Ok(())
}

fn main() {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: cadcraft-edit-frame <drawing>");
        std::process::exit(2);
    };
    if let Err(error) = run(Path::new(&path)) {
        eprintln!("cadcraft edit frame: {error}");
        std::process::exit(1);
    }
}
