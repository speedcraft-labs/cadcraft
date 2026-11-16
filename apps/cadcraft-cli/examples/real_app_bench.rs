//! Single-operation adapter for corpus-backed CADCraft journeys.
//!
//! Usage: `real_app_bench open|edit|save|verify INPUT [OUTPUT]`.

use std::error::Error;
use std::fs;
use std::time::Instant;

use cadcraft_engine::{
    Session,
    doc::{Drawing, Space, color::Color},
};
use serde_json::json;
use sha2::{Digest, Sha256};

fn identity(drawing: &Drawing) -> String {
    let ext = drawing.extents(&Space::Model);
    let sample = match drawing.model.iter().next() {
        Some(entity) => format!("|sample={}:{}", entity.handle.hex(), entity.common.color.name()),
        None => String::new(),
    };
    format!("entities={}|extents={:.6},{:.6},{:.6},{:.6}{sample}", drawing.entity_count(), ext.min.x, ext.min.y, ext.max.x, ext.max.y)
}

fn digest(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn print_markers(duration: std::time::Duration, stable_identity: &str) {
    println!("CRAFT_JOURNEY_NS={}", duration.as_nanos());
    println!("CRAFT_OUTPUT_SHA256={}", digest(stable_identity));
}

fn install_io() {
    cadcraft_engine::cmd::file::set_io(cadcraft_engine::cmd::file::IoHooks {
        read: |bytes, name| cadcraft_io::read(bytes, name).map_err(|error| error.to_string()),
        write: |drawing, name| cadcraft_io::write(drawing, name).map_err(|error| error.to_string()),
        plot: Some(|drawing, space, options| cadcraft_io::plotter::plot(drawing, space, options).map_err(|error| error.to_string())),
    });
    cadcraft_engine::cmd::file::set_image_decoder(|bytes| cadcraft_io::raster::decode(bytes).map_err(|error| error.to_string()));
    cadcraft_engine::cmd::file::set_write_as(|drawing, name, options| {
        cadcraft_io::write_with(drawing, name, &cadcraft_io::SaveOptions { version: options.version, binary: options.binary })
            .map_err(|error| error.to_string())
    });
    cadcraft_engine::cmd::file::set_entity_codec(cadcraft_engine::cmd::file::EntityCodec {
        groups: cadcraft_io::entity_groups,
        entity: cadcraft_io::entity_from_groups,
    });
}

fn run() -> Result<(), Box<dyn Error>> {
    install_io();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let operation = args.first().map(String::as_str).ok_or("missing operation")?;
    let input = args.get(1).map(String::as_str).ok_or("missing input path")?;

    if operation == "verify" {
        let bytes = fs::read(input)?;
        let drawing = cadcraft_io::read(&bytes, input)?;
        if drawing.model.is_empty() {
            return Err("saved drawing contains no model-space entities".into());
        }
        println!("CRAFT_OUTPUT_SHA256={}", digest(&identity(&drawing)));
        return Ok(());
    }

    let mut session = Session::empty();
    let elapsed;
    match operation {
        "open" => {
            let started = Instant::now();
            session.execute("open", &json!({"path": input}))?;
            elapsed = started.elapsed();
        }
        "edit" => {
            session.execute("open", &json!({"path": input}))?;
            let state = session.state()?;
            let entity = state.doc.model.iter().next().ok_or("input has no model-space entities")?;
            let entity_handle = entity.handle;
            let handle = entity_handle.hex();
            let target = if entity.common.color == Color::Index(42) { 43 } else { 42 };
            let started = Instant::now();
            session.execute("properties.set", &json!({"handles": [handle], "color": target}))?;
            elapsed = started.elapsed();
            let state = session.state()?;
            let edited = state.doc.model.get(entity_handle).ok_or("edited entity disappeared")?;
            if edited.common.color != Color::Index(target) {
                return Err("selected entity did not receive the requested color edit".into());
            }
        }
        "save" => {
            let output = args.get(2).map(String::as_str).ok_or("save needs an output path")?;
            session.execute("open", &json!({"path": input}))?;
            let state = session.state()?;
            let entity = state.doc.model.iter().next().ok_or("input has no model-space entities")?;
            let handle = entity.handle.hex();
            let target = if entity.common.color == Color::Index(42) { 43 } else { 42 };
            session.execute("properties.set", &json!({"handles": [handle], "color": target}))?;
            let stable_identity = identity(&session.state()?.doc);
            let started = Instant::now();
            let bytes = cadcraft_io::write(&session.state()?.doc, output)?;
            fs::write(output, bytes)?;
            elapsed = started.elapsed();
            print_markers(elapsed, &stable_identity);
            return Ok(());
        }
        _ => return Err(format!("unknown operation {operation:?}").into()),
    }

    let state = session.state()?;
    let stable_identity = if operation == "edit" {
        let entity = state.doc.model.iter().next().ok_or("input has no model-space entities")?;
        format!("{}|handle={}|color={}", identity(&state.doc), entity.handle.hex(), entity.common.color.name())
    } else {
        identity(&state.doc)
    };
    print_markers(elapsed, &stable_identity);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
