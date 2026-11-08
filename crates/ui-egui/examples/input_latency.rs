//! Input-to-frame latency for the first interaction while a real engineering drawing is loaded.

use cadcraft_engine::Session;
use cadcraft_ui_egui::{CadApp, Services};
use egui_kittest::Harness;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
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
    let deadline = Instant::now() + Duration::from_secs(120);
    while h.state().canvas.list.is_none() || h.state().canvas.building() {
        assert!(Instant::now() < deadline, "initial display-list build timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(1));
    }
    let lines = {
        let state = h.state().session.state().expect("drawing");
        let mut lines = BTreeMap::<([[u64; 3]; 2], String), Vec<_>>::new();
        for entity in state.doc.space(&state.space).expect("drawing space").iter() {
            let cadcraft_doc::EntityKind::Line(line) = &entity.kind else { continue };
            let mut endpoints =
                [[line.a.x.to_bits(), line.a.y.to_bits(), line.a.z.to_bits()], [line.b.x.to_bits(), line.b.y.to_bits(), line.b.z.to_bits()]];
            endpoints.sort_unstable();
            // Skip endpoint/style duplicates so the moved entity is stable across unordered
            // imports and app revisions. Handles are not part of the cross-revision key.
            let style = serde_json::to_string(&entity.common).expect("serialize line style");
            lines.entry((endpoints, style)).or_default().push(entity.handle);
        }
        lines
    };
    let ((line_key, _style), handles) = lines.into_iter().find(|(_, handles)| handles.len() == 1).expect("loaded drawing has a unique line to edit");
    let handle = handles[0];
    println!("CRAFT_INPUT_LINE_KEY={line_key:?}");
    let revision = h.state().session.state().expect("drawing").revision;
    // Move one entity after the heavy drawing has settled. The resulting display-list rebuild
    // is the UI-thread stall this journey targets; the first paint is timed, while completion is
    // awaited below solely for the output equality check.
    let start = Instant::now();
    h.state_mut().session.execute("move", &serde_json::json!({ "handles": [handle.hex()], "delta": [1.0, 0.0] })).expect("move one entity");
    h.step();
    let latency = start.elapsed();
    let deadline = Instant::now() + Duration::from_secs(120);
    while h.state().canvas.list.is_none() || h.state().canvas.building() {
        assert!(Instant::now() < deadline, "edit-triggered display-list rebuild timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(h.state().session.state().expect("drawing").revision > revision, "entity edit did not update the drawing");
    let list = h.state().canvas.list.as_ref().expect("rebuilt display list");
    assert!(!list.prims.is_empty(), "loaded display list is empty");
    // Renderer tessellation fragments can split in different places across worker runs. Check
    // the edited document's canonical entity content instead; ignore handles and iteration order.
    let state = h.state().session.state().expect("drawing");
    let mut entities = state
        .doc
        .space(&state.space)
        .expect("drawing space")
        .iter()
        .map(|entity| {
            let common = serde_json::to_vec(&entity.common).expect("serialize entity properties");
            let kind = serde_json::to_vec(&entity.kind).expect("serialize entity geometry");
            let mut fingerprint = Sha256::new();
            fingerprint.update((common.len() as u64).to_le_bytes());
            fingerprint.update(common);
            fingerprint.update((kind.len() as u64).to_le_bytes());
            fingerprint.update(kind);
            let mut digest = [0; 32];
            digest.copy_from_slice(&fingerprint.finalize());
            digest
        })
        .collect::<Vec<_>>();
    entities.sort_unstable();
    let mut digest = Sha256::new();
    digest.update((entities.len() as u64).to_le_bytes());
    for entity in entities {
        digest.update(entity);
    }

    println!("CRAFT_JOURNEY_NS={}", latency.as_nanos());
    println!("CRAFT_OUTPUT_SHA256={:x}", digest.finalize());
}
