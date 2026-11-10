//! Measure a full model-space display-list build on a pinned real drawing.
//!
//! Usage: `hatch_render_bench DRAWING`.

use std::error::Error;
use std::time::Instant;

use cadcraft_engine::doc::Space;
use cadcraft_engine::geom::{Bounds2, Vec2};
use cadcraft_engine::render::{self, Options};
use sha2::{Digest, Sha256};

fn run() -> Result<(), Box<dyn Error>> {
    let input = std::env::args().nth(1).ok_or("missing drawing path")?;
    let bytes = std::fs::read(&input)?;
    let drawing = cadcraft_io::read(&bytes, &input)?;
    let space = Space::Model;
    let ext = drawing.extents(&space);
    let size = ext.width().max(ext.height());
    if ext.is_empty() || !(size.is_finite() && size > 0.0) {
        return Err("drawing has no finite model-space extents".into());
    }

    // Match CADCraft's 1,600 × 1,000-pixel zoom-extents display-list profile.
    let pixel = 2f64.powi((size / 1000.0).max(1e-300).log2().floor() as i32);
    let half = Vec2::new(size * 0.8, size * 0.5);
    let visible = Bounds2::new(ext.center() - half, ext.center() + half);
    let options = Options { tolerance: pixel * 0.5, min_dash: pixel * 2.0, view: Some(visible.expand(size * 0.8)), ..Default::default() };

    let repeats = std::env::var("CRAFT_PROFILE_RENDER_REPEATS").ok().and_then(|s| s.parse().ok()).filter(|n| *n > 0).unwrap_or(1).min(100);
    let started = Instant::now();
    let mut list = render::build(&drawing, &space, &options);
    for _ in 1..repeats {
        list = render::build(&drawing, &space, &options);
    }
    let elapsed = started.elapsed();
    println!("CRAFT_JOURNEY_NS={}", elapsed.as_nanos());
    // Profiling isolates render::build from this intentionally expensive identity check. Normal
    // journey runs still hash the complete display list so upstream/fork output stays comparable.
    if std::env::var_os("CRAFT_SKIP_OUTPUT_HASH").is_none() {
        let identity = format!("{list:?}");
        let digest = Sha256::digest(identity.as_bytes());
        println!("CRAFT_OUTPUT_SHA256={digest:x}");
    }
    println!(
        "hatch-render: {} entities, {} primitives, {} vertices, {} triangles, truncated={}",
        drawing.entity_count(),
        list.prims.len(),
        list.verts.len(),
        list.tris.len(),
        list.truncated
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
