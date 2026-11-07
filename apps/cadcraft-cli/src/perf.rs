use super::args::PerfTarget;
use super::error::Result;
use super::open;
use super::output::Streams;
use cadcraft_engine::Session;

fn output_hash(value: &impl std::fmt::Debug) -> u64 {
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
    let _ = std::fmt::write(&mut hash, format_args!("{value:?}"));
    hash.0
}

/// Why `perf` cannot run.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PerfError {
    #[error("perf: the drawing is empty")]
    EmptyDrawing,
    #[error("perf: full and incremental display lists differ")]
    OutputMismatch,
}

/// Small deterministic generator (xorshift64*) so perf runs are repeatable without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// A synthetic drawing with `n` mixed entities (lines, circles, arcs, polylines, text) spread
/// over a square whose side grows with sqrt(n), so density stays constant.
fn synthetic(n: usize, side: f64) -> Result<cadcraft_engine::doc::Drawing, cadcraft_engine::doc::DocError> {
    use cadcraft_engine::doc::{self as d, Common, EntityKind, Space};
    use cadcraft_engine::geom::{PolyVertex, Vec2, Vec3};
    let mut dr = d::Drawing::new_imperial();
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let v3 = |x: f64, y: f64| Vec3::new(x, y, 0.0);
    for i in 0..n {
        let (x, y) = (r.range(0.0, side), r.range(0.0, side));
        let s = r.range(0.5, 8.0);
        let kind = match i % 5 {
            0 => EntityKind::Line(d::Line { a: v3(x, y), b: v3(x + r.range(-s, s), y + r.range(-s, s)) }),
            1 => EntityKind::Circle(d::Circle { center: v3(x, y), radius: s * 0.5 }),
            2 => EntityKind::Arc(d::Arc { center: v3(x, y), radius: s * 0.5, start: r.range(0.0, 3.0), end: r.range(3.2, 6.2) }),
            3 => EntityKind::LwPolyline(d::LwPolyline {
                vertices: (0..6)
                    .map(|k| PolyVertex {
                        p: Vec2::new(x + f64::from(k) * s * 0.3, y + r.range(-s, s) * 0.3),
                        bulge: if k == 2 { 0.4 } else { 0.0 },
                        start_width: 0.0,
                        end_width: 0.0,
                    })
                    .collect(),
                closed: i % 2 == 0,
                const_width: 0.0,
                elevation: 0.0,
                plinegen: false,
            }),
            _ => EntityKind::Text(d::Text {
                insert: v3(x, y),
                align_pt: None,
                height: s * 0.2,
                value: format!("T{i}"),
                rotation: 0.0,
                width_factor: 1.0,
                oblique: 0.0,
                style: "Standard".into(),
                halign: d::HAlign::Left,
                valign: d::VAlign::Baseline,
            }),
        };
        dr.add(&Space::Model, Common::default(), kind)?;
    }
    Ok(dr)
}

/// Deterministic hatch heavy drawing for the one-entity edit journey when no suitable DWG is
/// present in the pinned corpus. The entity count and regular pattern make its cost reproducible.
fn synthetic_hatched(n: usize) -> Result<cadcraft_engine::doc::Drawing, cadcraft_engine::doc::DocError> {
    use cadcraft_engine::doc::{self as d, Common, EntityKind, Hatch, HatchLoop, Space};
    use cadcraft_engine::geom::{PolyVertex, Vec2};
    let mut drawing = d::Drawing::new_imperial();
    let side = (n as f64).sqrt().ceil();
    for i in 0..n {
        let (x, y) = ((i as f64 % side) * 12.0, (i as f64 / side).floor() * 12.0);
        let loop_vertices = |radius: f64, count: usize| {
            (0..count)
                .map(|point| {
                    let angle = std::f64::consts::TAU * point as f64 / count as f64;
                    PolyVertex { p: Vec2::new(x + radius * angle.cos(), y + radius * angle.sin()), bulge: 0.0, start_width: 0.0, end_width: 0.0 }
                })
                .collect()
        };
        let hatch = Hatch {
            pattern: "ANSI31".into(),
            solid: false,
            loops: vec![
                HatchLoop { vertices: loop_vertices(5.0, 256), outer: true, sources: Vec::new() },
                HatchLoop { vertices: loop_vertices(1.4, 128), outer: false, sources: Vec::new() },
            ],
            scale: 8.0,
            angle: 0.0,
            associative: false,
            style: 0,
            elevation: 0.0,
            gradient: None,
            origin: Vec2::ZERO,
            background: None,
            pattern_lines: Vec::new(),
        };
        drawing.add(&Space::Model, Common::default(), EntityKind::Hatch(hatch))?;
    }
    Ok(drawing)
}

/// `perf FILE`: the display list built as the canvas builds it (a 1600 x 1000 px view at the
/// drawing's centre: tolerance, dash level of detail and view window from the zoom band), at
/// zoom extents and zoomed in 4x at a time, next to the full-detail build plots use.
fn perf_file(path: &std::path::Path, io: &mut Streams) -> Result {
    use cadcraft_engine::geom::{Bounds2, Vec2};
    use cadcraft_engine::render::{Options, build};
    use std::time::Instant;
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    let mut s = Session::empty();
    let t = Instant::now();
    open(&mut s, path)?;
    let open_ms = ms(t);
    let st = s.state()?;
    let (d, space) = (&st.doc, &st.space);
    let ext = d.extents(space);
    let size = ext.width().max(ext.height());
    if ext.is_empty() || !(size.is_finite() && size > 0.0) {
        return Err(PerfError::EmptyDrawing.into());
    }
    io.outln(format_args!("open {open_ms:.0} ms, {} entities, extents {:.1} x {:.1}", d.entity_count(), ext.width(), ext.height()));
    let edit_mode = std::env::var("CADCRAFT_PERF_EDIT_MODE").ok();
    io.outln(format_args!("{:>8} {:>10} {:>10}", "zoom", "build ms", "prims"));
    let mut finest = Options::default();
    let mut zoom_extents = Options::default();
    if edit_mode.is_some() {
        let h = size;
        let pixel = 2f64.powi((h / 1000.0).max(1e-300).log2().floor() as i32);
        let half = Vec2::new(h * 0.8, h * 0.5);
        let visible = Bounds2::new(ext.center() - half, ext.center() + half);
        zoom_extents = Options { tolerance: pixel * 0.5, min_dash: pixel * 2.0, view: Some(visible.expand(h * 0.8)), ..Default::default() };
        finest = zoom_extents.clone();
    } else {
        for k in 0..5 {
            let zoom = 4f64.powi(k);
            let h = size / zoom;
            let pixel = 2f64.powi((h / 1000.0).max(1e-300).log2().floor() as i32);
            let half = Vec2::new(h * 0.8, h * 0.5);
            let visible = Bounds2::new(ext.center() - half, ext.center() + half);
            let screen = Options { tolerance: pixel * 0.5, min_dash: pixel * 2.0, view: Some(visible.expand(h * 0.8)), ..Default::default() };
            if k == 0 {
                zoom_extents = screen.clone();
            }
            let t = Instant::now();
            let n = build(d, space, &screen).prims.len();
            io.outln(format_args!("{:>7}x {:>10.1} {n:>10}", zoom, ms(t)));
            finest = screen;
        }
    }
    // A representative single-entity property edit in a fit-to-drawing view. Journey mode skips
    // the unrelated zoom sweep and plot build so only the requested edit contributes work.
    // The timing excludes file open and initial cache population.
    // The real-file journey matches a fit-to-drawing view; the open-DXF probe keeps its zoomed-in
    // diagnostic measurement.
    let edit_opts = if edit_mode.is_some() { &zoom_extents } else { &finest };
    let edited_handle = d.space(space).and_then(|store| {
        store
            .iter()
            .find(|entity| matches!(entity.kind, cadcraft_engine::doc::EntityKind::Line(_)) && entity.common.visible)
            .or_else(|| store.iter().find(|entity| matches!(entity.kind, cadcraft_engine::doc::EntityKind::Hatch(_))))
            .or_else(|| store.iter().last())
            .map(|entity| entity.handle)
    });
    if let Some(handle) = edited_handle {
        let incremental_mode = edit_mode.as_deref() == Some("incremental");
        let full_mode = edit_mode.as_deref() == Some("full");
        let mut cache = cadcraft_engine::render::EntityDisplayCache::default();
        if incremental_mode || !full_mode {
            let _ = cadcraft_engine::render::build_cached(d, space, edit_opts, &mut cache);
        }
        let mut edited = d.as_ref().clone();
        if let Some(store) = edited.space_mut(space) {
            store.modify(handle, |entity| {
                if let cadcraft_engine::doc::EntityKind::Hatch(hatch) = &mut entity.kind {
                    hatch.scale *= 1.1;
                } else {
                    let color = entity.common.color;
                    entity.common.color = match color {
                        cadcraft_engine::doc::color::Color::Index(1) => cadcraft_engine::doc::color::Color::Index(2),
                        _ => cadcraft_engine::doc::color::Color::Index(1),
                    };
                }
            });
        }
        if incremental_mode {
            let t = Instant::now();
            let incremental = cadcraft_engine::render::build_cached(&edited, space, edit_opts, &mut cache);
            let incremental_duration = t.elapsed();
            let t = Instant::now();
            let full = build(&edited, space, edit_opts);
            let full_duration = t.elapsed();
            let full_ms = full_duration.as_secs_f64() * 1000.0;
            let incremental_ms = incremental_duration.as_secs_f64() * 1000.0;
            let identical = incremental.prims == full.prims
                && incremental.verts == full.verts
                && incremental.tris == full.tris
                && incremental.images == full.images
                && incremental.prim_layers == full.prim_layers
                && incremental.texts == full.texts
                && incremental.pens == full.pens
                && incremental.bounds == full.bounds
                && incremental.expanded == full.expanded
                && incremental.truncated == full.truncated;
            io.outln(format_args!(
                "entity edit display list: full {full_ms:.2} ms, incremental {incremental_ms:.2} ms, outputs identical={identical}"
            ));
            let hash = output_hash(&incremental);
            io.outln(format_args!("CRAFT_JOURNEY_NS={}", incremental_duration.as_nanos()));
            io.outln(format_args!("CRAFT_OUTPUT_HASH={hash:016x}"));
            if !identical {
                return Err(PerfError::OutputMismatch.into());
            }
        } else if full_mode {
            let t = Instant::now();
            let full = build(&edited, space, edit_opts);
            let full_duration = t.elapsed();
            let full_ms = full_duration.as_secs_f64() * 1000.0;
            io.outln(format_args!("entity edit display list: full {full_ms:.2} ms, incremental not run, outputs identical=canonical"));
            let hash = output_hash(&full);
            io.outln(format_args!("CRAFT_JOURNEY_NS={}", full_duration.as_nanos()));
            io.outln(format_args!("CRAFT_OUTPUT_HASH={hash:016x}"));
        } else {
            let t = Instant::now();
            let _incremental = cadcraft_engine::render::build_cached(&edited, space, edit_opts, &mut cache);
            let incremental_ms = ms(t);
            io.outln(format_args!("entity edit display list: incremental {incremental_ms:.2} ms"));
        }
    }
    if edit_mode.is_none() {
        // Plots: every pattern line, everywhere (last, so its large list doesn't weigh on above).
        let t = Instant::now();
        let n = build(d, space, &Options { min_dash: 0.0, view: None, ..finest }).prims.len();
        io.outln(format_args!("{:>8} {:>10.1} {n:>10}", "full", ms(t)));
    }
    Ok(())
}

pub(crate) fn perf(target: Option<&PerfTarget>, io: &mut Streams) -> Result {
    use cadcraft_engine::doc::Space;
    use cadcraft_engine::geom::{Bounds2, Vec2};
    use cadcraft_engine::{select, snap};
    use std::time::Instant;
    let n = match target {
        Some(PerfTarget::File(path)) => return perf_file(path, io),
        Some(PerfTarget::Count(n)) => *n,
        None => 200_000,
    };
    let side = (n as f64).sqrt() * 10.0;
    let mut rows: Vec<(String, f64, String)> = Vec::new();
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let hatched = std::env::var_os("CADCRAFT_PERF_HATCHED").is_some();
    let d = if hatched { synthetic_hatched(n)? } else { synthetic(n, side)? };
    rows.push((if hatched { "build synthetic hatched drawing" } else { "build synthetic drawing" }.into(), ms(t), format!("{n} entities")));

    // Display list at a 1600 px wide "zoom extents" view (tolerance = half a pixel).
    let px = side / 1600.0;
    let opts = cadcraft_engine::render::Options { tolerance: px * 0.5, min_dash: px * 2.0, ..Default::default() };
    if hatched {
        use std::hash::{Hash, Hasher};
        let mut cache = cadcraft_engine::render::EntityDisplayCache::default();
        let _ = cadcraft_engine::render::build_cached(&d, &Space::Model, &opts, &mut cache);
        let mut edited = d.clone();
        if let Some(handle) = edited.model.iter().last().map(|e| e.handle) {
            let _ = edited.model.modify(handle, |e| {
                if let cadcraft_engine::doc::EntityKind::Hatch(hatch) = &mut e.kind {
                    hatch.scale *= 1.1;
                }
            });
            let incremental_mode = std::env::var("CADCRAFT_PERF_HATCHED_MODE").as_deref() == Ok("incremental");
            let t = Instant::now();
            let measured = if incremental_mode {
                cadcraft_engine::render::build_cached(&edited, &Space::Model, &opts, &mut cache)
            } else {
                std::sync::Arc::new(cadcraft_engine::render::build(&edited, &Space::Model, &opts))
            };
            let measured_ms = ms(t);
            let reference = if incremental_mode {
                std::sync::Arc::new(cadcraft_engine::render::build(&edited, &Space::Model, &opts))
            } else {
                cadcraft_engine::render::build_cached(&edited, &Space::Model, &opts, &mut cache)
            };
            let identical = measured.prims == reference.prims
                && measured.verts == reference.verts
                && measured.tris == reference.tris
                && measured.images == reference.images
                && measured.prim_layers == reference.prim_layers
                && measured.texts == reference.texts
                && measured.pens == reference.pens
                && measured.bounds == reference.bounds
                && measured.expanded == reference.expanded
                && measured.truncated == reference.truncated;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            format!("{:?}", measured.prims).hash(&mut hash);
            for p in measured.verts.iter().chain(&measured.tris) {
                p.x.to_bits().hash(&mut hash);
                p.y.to_bits().hash(&mut hash);
            }
            io.outln(format_args!(
                "entity edit {} {:.3} ms; identical={identical}",
                if incremental_mode { "incremental" } else { "full" },
                measured_ms
            ));
            io.outln(format_args!("CRAFT_JOURNEY_NS={}", (measured_ms * 1_000_000.0) as u64));
            io.outln(format_args!("CRAFT_OUTPUT_HASH={:016x}", hash.finish()));
        }
        return Ok(());
    }
    let t = Instant::now();
    let list = cadcraft_engine::render::build(&d, &Space::Model, &opts);
    rows.push(("display list build".into(), ms(t), format!("{} prims, {} segments", list.prims.len(), list.segment_count())));

    // Hit testing on a zoomed-in view (aperture = 5 px at 0.02 units/px).
    let ap = 0.1;
    let mut r = Rng(42);
    let pts: Vec<Vec2> = (0..1000).map(|_| Vec2::new(r.range(0.0, side), r.range(0.0, side))).collect();
    let t = Instant::now();
    let first = select::pick(&d, &Space::Model, pts.first().copied().unwrap_or(Vec2::ZERO), ap);
    rows.push(("first pick (incl. index build)".into(), ms(t), format!("{first:?}")));
    let t = Instant::now();
    let hits = pts.iter().filter(|p| select::pick(&d, &Space::Model, **p, ap).is_some()).count();
    let total = ms(t);
    rows.push(("pick x1000".into(), total, format!("{hits} hits, {:.3} ms/pick", total / 1000.0)));

    // One edit, then a pick: only the touched chunk is re-indexed.
    let mut edited = d.clone();
    if let Some(h) = edited.model.last().map(|e| e.handle) {
        edited.model.modify(h, |e| e.common.layer = "0".into());
    }
    let t = Instant::now();
    let _ = select::pick(&edited, &Space::Model, pts.first().copied().unwrap_or(Vec2::ZERO), ap);
    rows.push(("pick after one edit (incremental re-index)".into(), ms(t), String::new()));

    let mut sel = 0usize;
    let t = Instant::now();
    for k in 0..20 {
        let c = Vec2::new(r.range(0.0, side), r.range(0.0, side));
        let half = side * 0.05;
        sel += select::select_window(&d, &Space::Model, Bounds2::new(c - Vec2::new(half, half), c + Vec2::new(half, half)), k % 2 == 1).len();
    }
    rows.push(("window/crossing select x20 (10% side)".into(), ms(t), format!("{sel} selected")));

    let t = Instant::now();
    let fence: Vec<Vec2> = (0..4).map(|_| Vec2::new(r.range(0.0, side), r.range(0.0, side))).collect();
    let fsel = select::select_fence(&d, &Space::Model, &fence).len();
    rows.push(("fence select (3 legs)".into(), ms(t), format!("{fsel} selected")));

    let osmode = snap::mode::END | snap::mode::MID | snap::mode::CEN | snap::mode::QUA | snap::mode::INT;
    let t = Instant::now();
    let snaps = pts.iter().filter(|p| snap::osnap(&d, &Space::Model, **p, ap * 2.0, osmode, None, false).is_some()).count();
    let total = ms(t);
    rows.push(("osnap x1000".into(), total, format!("{snaps} hits, {:.3} ms/query", total / 1000.0)));

    let t = Instant::now();
    let bytes = cadcraft_io::write(&d, "perf.dxf")?;
    rows.push(("DXF write".into(), ms(t), format!("{:.1} MB", bytes.len() as f64 / 1e6)));
    let t = Instant::now();
    let back = cadcraft_io::read(&bytes, "perf.dxf")?;
    rows.push(("DXF read".into(), ms(t), format!("{} entities", back.entity_count())));

    io.outln(format_args!("{:<40} {:>12}  notes", "operation", "ms"));
    io.outln(format_args!("{}", "-".repeat(80)));
    for (name, t, note) in rows {
        io.outln(format_args!("{name:<40} {t:>12.2}  {note}"));
    }
    Ok(())
}
