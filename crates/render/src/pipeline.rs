#[cfg(not(target_arch = "wasm32"))]
use super::display::append;
use super::display::{DisplayList, Kind, NO_LAYER, PlotText, VIEWPORT_CONTENT};
use super::draw::entity;
use super::viewports::{MAX_VIEWPORTS, viewport};
use super::{Bounds2, Builder, Color, Ctx, Drawing, Entity, EntityKind, Handle, Lineweight, Mat3, Space, Vec2, paper, pen, plotstyle, point};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
struct FragmentRange {
    prims: (usize, usize),
    verts: (usize, usize),
    tris: (usize, usize),
    images: (usize, usize),
    prim_layers: (usize, usize),
    texts: (usize, usize),
    pens: (usize, usize),
}

impl FragmentRange {
    fn at(list: &DisplayList) -> Self {
        Self {
            prims: (list.prims.len(), 0),
            verts: (list.verts.len(), 0),
            tris: (list.tris.len(), 0),
            images: (list.images.len(), 0),
            prim_layers: (list.prim_layers.len(), 0),
            texts: (list.texts.len(), 0),
            pens: (list.pens.len(), 0),
        }
    }

    fn finish(mut self, end: &DisplayList) -> Self {
        self.prims.1 = end.prims.len() - self.prims.0;
        self.verts.1 = end.verts.len() - self.verts.0;
        self.tris.1 = end.tris.len() - self.tris.0;
        self.images.1 = end.images.len() - self.images.0;
        self.prim_layers.1 = end.prim_layers.len() - self.prim_layers.0;
        self.texts.1 = end.texts.len() - self.texts.0;
        self.pens.1 = end.pens.len() - self.pens.0;
        self
    }

    fn for_fragment(start: Self, fragment: &DisplayList) -> Self {
        Self {
            prims: (start.prims.0, fragment.prims.len()),
            verts: (start.verts.0, fragment.verts.len()),
            tris: (start.tris.0, fragment.tris.len()),
            images: (start.images.0, fragment.images.len()),
            prim_layers: (start.prim_layers.0, fragment.prim_layers.len()),
            texts: (start.texts.0, fragment.texts.len()),
            pens: (start.pens.0, fragment.pens.len()),
        }
    }

    fn same_shape(self, other: Self) -> bool {
        self.prims.1 == other.prims.1
            && self.verts.1 == other.verts.1
            && self.tris.1 == other.tris.1
            && self.images.1 == other.images.1
            && self.prim_layers.1 == other.prim_layers.1
            && self.texts.1 == other.texts.1
            && self.pens.1 == other.pens.1
    }
}

/// Reusable per-entity tessellations for interactive model-space edits.
///
/// Entries are keyed by handle and the address of the immutable entity `Arc`, so a copy-on-write
/// edit invalidates only the touched entity. The flattened list is still assembled in draw order;
/// geometry generation (especially hatch tessellation) is skipped for every unchanged entity.
#[derive(Default)]
pub struct EntityDisplayCache {
    options: Option<Vec<u64>>,
    fragments: HashMap<Handle, (Arc<Entity>, DisplayList)>,
    ranges: HashMap<Handle, FragmentRange>,
    order: Vec<Handle>,
    positions: HashMap<Handle, usize>,
    prefix_expansions: Vec<usize>,
    store_snapshot: Option<cadcraft_doc::EntityStore>,
    flat: Option<Arc<DisplayList>>,
    expansion: usize,
    truncated_fragments: usize,
    budgeted: bool,
    // Hold the definitions used to calculate the option key. Besides keeping pointer identities
    // stable, this makes any copy-on-write block edit invalidate fragments that expand a block.
    block_versions: Vec<std::sync::Arc<cadcraft_doc::Block>>,
    // Raster images can load after the first list build without changing an entity. Retain the
    // immutable map so a COW image load invalidates cached IMAGE fragments too.
    rasters: Option<cadcraft_doc::Rasters>,
}

impl EntityDisplayCache {
    pub fn clear(&mut self) {
        self.options = None;
        self.fragments.clear();
        self.ranges.clear();
        self.order.clear();
        self.positions.clear();
        self.prefix_expansions.clear();
        self.store_snapshot = None;
        self.flat = None;
        self.expansion = 0;
        self.truncated_fragments = 0;
        self.budgeted = false;
        self.block_versions.clear();
        self.rasters = None;
    }
}

/// Build a model-space display list while reusing tessellation for unchanged entities.
/// A changed block definition invalidates the cache. Edits patch the cached flat list in place
/// when fragment sizes stay stable, and ordered block-budget allocation keeps over-budget files
/// deterministic.
pub fn build_cached(d: &Drawing, space: &Space, opts: &Options, cache: &mut EntityDisplayCache) -> Arc<DisplayList> {
    use std::hash::{Hash, Hasher};

    let normalized_opts = with_point_view(d, space, opts);
    let opts = &normalized_opts;
    let mut style_hash = std::collections::hash_map::DefaultHasher::new();
    format!(
        "{:?}",
        (&d.header, &d.layers, &d.linetypes, &d.text_styles, &d.dim_styles, &d.mleader_styles, &d.table_styles, &d.mline_styles, &d.hidden_objects,)
    )
    .hash(&mut style_hash);
    for (name, block) in &d.blocks {
        name.hash(&mut style_hash);
        (Arc::as_ptr(block) as usize).hash(&mut style_hash);
    }
    let options = [
        opts.tolerance.to_bits(),
        opts.min_dash.to_bits(),
        u64::from(opts.text),
        u64::from(opts.linetypes),
        u64::from(opts.fill),
        u64::from(opts.lineweights),
        opts.view_height.to_bits(),
        opts.view.map_or(0, |v| v.min.x.to_bits()),
        opts.view.map_or(0, |v| v.min.y.to_bits()),
        opts.view.map_or(0, |v| v.max.x.to_bits()),
        opts.view.map_or(0, |v| v.max.y.to_bits()),
        u64::from(opts.transparency),
        opts.skip_viewport.map_or(0, |h| h.0),
        opts.plot_style_table.as_ref().map_or(0, |t| Arc::as_ptr(t) as usize as u64),
        style_hash.finish(),
    ];
    let Some(store) = d.space(space) else {
        cache.clear();
        return Arc::new(DisplayList::default());
    };
    let cacheable = matches!(space, Space::Model) && opts.plot_style_table.is_none() && opts.cancel.is_none();
    if !cacheable {
        cache.clear();
        return Arc::new(build(d, space, opts));
    }
    let rasters_changed = cache.rasters.as_ref().is_some_and(|old| !old.shares_storage_with(&d.rasters));
    let options_changed = cache.options.as_deref() != Some(options.as_slice()) || rasters_changed;
    if options_changed {
        cache.clear();
        cache.options = Some(options.to_vec());
        cache.block_versions = d.blocks.values().cloned().collect();
        cache.rasters = Some(d.rasters.clone());
    }

    let changes = if options_changed { None } else { cache.store_snapshot.as_ref().and_then(|old| store.changed_entities(old)) };
    if let Some(changed) = changes.as_ref() {
        if changed.iter().any(|entity| matches!(entity.kind, EntityKind::Viewport(_))) {
            cache.clear();
            return Arc::new(build(d, space, opts));
        }
        if !changed.is_empty() {
            if cache.flat.is_some() && apply_cached_changes(d, opts, cache, changed) {
                cache.store_snapshot = Some(store.clone());
                return cache.flat.as_ref().map(Arc::clone).unwrap_or_else(|| Arc::new(build(d, space, opts)));
            }
        } else if let Some(flat) = cache.flat.as_ref() {
            cache.store_snapshot = Some(store.clone());
            return Arc::clone(flat);
        }
    }

    let entities: Vec<_> = store.iter().cloned().collect();
    if entities.iter().any(|entity| matches!(entity.kind, EntityKind::Viewport(_))) {
        cache.clear();
        return Arc::new(build(d, space, opts));
    }
    let budgeted = block_expansion_upper_bound(d, store) > super::MAX_BLOCK_EXPANSION;
    populate_cache(d, store, opts, cache, options, &entities, budgeted)
}

fn populate_cache(
    d: &Drawing,
    store: &cadcraft_doc::EntityStore,
    opts: &Options,
    cache: &mut EntityDisplayCache,
    options: [u64; 15],
    entities: &[Arc<Entity>],
    budgeted: bool,
) -> Arc<DisplayList> {
    cache.fragments.clear();
    cache.ranges.clear();
    cache.order.clear();
    cache.positions.clear();
    cache.prefix_expansions.clear();
    cache.expansion = 0;
    cache.truncated_fragments = 0;
    cache.budgeted = budgeted;
    cache.options = Some(options.to_vec());
    cache.block_versions = d.blocks.values().cloned().collect();
    cache.rasters = Some(d.rasters.clone());

    use rayon::prelude::*;
    let fragments: Vec<_> =
        entities.par_iter().map(|entity| (entity.handle, entity.clone(), build_entities(d, std::iter::once(entity.as_ref()), opts))).collect();
    cache.fragments.extend(fragments.into_iter().map(|(handle, entity, fragment)| (handle, (entity, fragment))));
    if budgeted {
        let spent = std::sync::atomic::AtomicUsize::new(0);
        for entity in entities {
            if !matches!(entity.kind, EntityKind::Insert(_)) {
                continue;
            }
            let Some((_, fragment)) = cache.fragments.get(&entity.handle) else { continue };
            let expansions = fragment.expanded;
            let needs_ordered_budget =
                fragment.truncated || spent.load(std::sync::atomic::Ordering::Relaxed).saturating_add(expansions) > super::MAX_BLOCK_EXPANSION;
            if needs_ordered_budget {
                let fragment = build_entity_with_budget(d, entity, opts, &spent);
                cache.fragments.insert(entity.handle, (entity.clone(), fragment));
            } else {
                spent.fetch_add(expansions, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    let mut flat = DisplayList::default();
    for (index, entity) in entities.iter().enumerate() {
        let Some((_, fragment)) = cache.fragments.get(&entity.handle) else { continue };
        cache.positions.insert(entity.handle, index);
        cache.order.push(entity.handle);
        cache.prefix_expansions.push(cache.expansion);
        let before = FragmentRange::at(&flat);
        super::display::append_ref(&mut flat, fragment);
        cache.ranges.insert(entity.handle, before.finish(&flat));
        cache.expansion = cache.expansion.saturating_add(fragment.expanded);
        if fragment.truncated {
            cache.truncated_fragments = cache.truncated_fragments.saturating_add(1);
        }
    }
    cache.store_snapshot = Some(store.clone());
    cache.flat = Some(Arc::new(flat));
    cache.flat.as_ref().map(Arc::clone).unwrap_or_else(|| Arc::new(DisplayList::default()))
}

fn apply_cached_changes(d: &Drawing, opts: &Options, cache: &mut EntityDisplayCache, changed: &[Arc<Entity>]) -> bool {
    let mut updates = Vec::with_capacity(changed.len());
    for entity in changed {
        let Some(&position) = cache.positions.get(&entity.handle) else { return false };
        let Some((_, previous)) = cache.fragments.get(&entity.handle) else { return false };
        let spent_before = cache.prefix_expansions.get(position).copied().unwrap_or(0);
        let fragment = if cache.budgeted {
            let spent = std::sync::atomic::AtomicUsize::new(spent_before);
            build_entity_with_budget(d, entity, opts, &spent)
        } else {
            build_entities(d, std::iter::once(entity.as_ref()), opts)
        };
        let Some(&range) = cache.ranges.get(&entity.handle) else { return false };
        let next_range = FragmentRange::for_fragment(range, &fragment);
        if !range.same_shape(next_range) && !(changed.len() == 1 && position + 1 == cache.order.len()) {
            return false;
        }
        let next_total = cache.expansion.saturating_sub(previous.expanded).saturating_add(fragment.expanded);
        if cache.budgeted && previous.expanded != fragment.expanded && position + 1 < cache.order.len() && next_total > super::MAX_BLOCK_EXPANSION {
            return false;
        }
        updates.push((entity.clone(), fragment, range, position));
    }

    let Some(flat_arc) = cache.flat.as_mut() else { return false };
    let flat = Arc::make_mut(flat_arc);
    for (entity, fragment, old_range, position) in updates {
        let Some((_, previous)) = cache.fragments.get(&entity.handle) else { return false };
        let old_expanded = previous.expanded;
        let old_bounds = previous.bounds;
        let old_truncated = previous.truncated;
        let new_expanded = fragment.expanded;
        let new_truncated = fragment.truncated;
        let new_range = if old_range.same_shape(FragmentRange::for_fragment(old_range, &fragment)) {
            patch_fragment(flat, old_range, &fragment);
            old_range
        } else {
            truncate_fragment_tail(flat, old_range);
            let before = FragmentRange::at(flat);
            super::display::append_ref(flat, &fragment);
            before.finish(flat)
        };
        update_bounds(flat, old_bounds, &fragment, &cache.fragments, entity.handle);
        flat.expanded = cache.expansion.saturating_sub(old_expanded).saturating_add(new_expanded);
        cache.expansion = flat.expanded;
        if old_truncated {
            cache.truncated_fragments = cache.truncated_fragments.saturating_sub(1);
        }
        if new_truncated {
            cache.truncated_fragments = cache.truncated_fragments.saturating_add(1);
        }
        flat.truncated = cache.truncated_fragments != 0;
        if new_expanded != old_expanded {
            for prefix in cache.prefix_expansions.iter_mut().skip(position.saturating_add(1)) {
                *prefix = prefix.saturating_sub(old_expanded).saturating_add(new_expanded);
            }
        }
        cache.ranges.insert(entity.handle, new_range);
        cache.fragments.insert(entity.handle, (entity, fragment));
    }
    true
}

fn patch_fragment(flat: &mut DisplayList, range: FragmentRange, fragment: &DisplayList) {
    for (dst, source) in flat.prims[range.prims.0..range.prims.0 + range.prims.1].iter_mut().zip(&fragment.prims) {
        let mut prim = *source;
        prim.start += if matches!(prim.kind, super::Kind::Tris | super::Kind::Mask) { range.tris.0 as u32 } else { range.verts.0 as u32 };
        if let super::Kind::Image { index } = &mut prim.kind {
            *index = index.saturating_add(range.images.0 as u32);
        }
        *dst = prim;
    }
    flat.verts[range.verts.0..range.verts.0 + range.verts.1].copy_from_slice(&fragment.verts);
    flat.tris[range.tris.0..range.tris.0 + range.tris.1].copy_from_slice(&fragment.tris);
    flat.images[range.images.0..range.images.0 + range.images.1].clone_from_slice(&fragment.images);
    flat.prim_layers[range.prim_layers.0..range.prim_layers.0 + range.prim_layers.1].copy_from_slice(&fragment.prim_layers);
    for (dst, source) in flat.texts[range.texts.0..range.texts.0 + range.texts.1].iter_mut().zip(&fragment.texts) {
        let mut text = source.clone();
        if text.prim < super::PlotText::CLIPPED {
            text.prim += range.prims.0 as u32;
        }
        *dst = text;
    }
    flat.pens[range.pens.0..range.pens.0 + range.pens.1].copy_from_slice(&fragment.pens);
}

fn truncate_fragment_tail(flat: &mut DisplayList, range: FragmentRange) {
    flat.prims.truncate(range.prims.0);
    flat.verts.truncate(range.verts.0);
    flat.tris.truncate(range.tris.0);
    flat.images.truncate(range.images.0);
    flat.prim_layers.truncate(range.prim_layers.0);
    flat.texts.truncate(range.texts.0);
    flat.pens.truncate(range.pens.0);
}

fn update_bounds(flat: &mut DisplayList, old: Bounds2, new: &DisplayList, fragments: &HashMap<Handle, (Arc<Entity>, DisplayList)>, changed: Handle) {
    let b = flat.bounds;
    let lost_extreme = old.min.x == b.min.x && new.bounds.min.x > b.min.x
        || old.min.y == b.min.y && new.bounds.min.y > b.min.y
        || old.max.x == b.max.x && new.bounds.max.x < b.max.x
        || old.max.y == b.max.y && new.bounds.max.y < b.max.y;
    if lost_extreme {
        flat.bounds =
            fragments.iter().filter(|(handle, _)| **handle != changed).fold(new.bounds, |bounds, (_, (_, fragment))| bounds.union(&fragment.bounds));
    } else {
        flat.bounds = flat.bounds.union(&new.bounds);
    }
}

fn block_expansion_upper_bound(d: &Drawing, store: &cadcraft_doc::EntityStore) -> usize {
    let cap = super::MAX_BLOCK_EXPANSION.saturating_add(1);
    let mut total = 0usize;
    let mut memo: HashMap<(String, usize), usize> = HashMap::new();
    let mut active = std::collections::HashSet::new();
    for entity in store.iter() {
        if let EntityKind::Insert(insert) = &entity.kind {
            total = total.saturating_add(insert_expansion_upper_bound(d, insert, 0, cap, &mut memo, &mut active)).min(cap);
            if total == cap {
                return cap;
            }
        }
    }
    total
}

fn insert_expansion_upper_bound(
    d: &Drawing,
    insert: &cadcraft_doc::Insert,
    depth: usize,
    cap: usize,
    memo: &mut HashMap<(String, usize), usize>,
    active: &mut std::collections::HashSet<(String, usize)>,
) -> usize {
    if depth >= cadcraft_doc::MAX_BLOCK_DEPTH {
        return 0;
    }
    if d.block(&insert.block).is_none() {
        return 0;
    }
    let instances = (insert.cols.clamp(1, 10_000) as usize).saturating_mul(insert.rows.clamp(1, 10_000) as usize).min(cap);
    let body = block_body_expansion_upper_bound(d, &insert.block, depth + 1, cap, memo, active);
    instances.saturating_mul(body.saturating_add(1).min(cap)).min(cap)
}

fn block_body_expansion_upper_bound(
    d: &Drawing,
    name: &str,
    entity_depth: usize,
    cap: usize,
    memo: &mut HashMap<(String, usize), usize>,
    active: &mut std::collections::HashSet<(String, usize)>,
) -> usize {
    if let Some(value) = memo.get(&(name.to_owned(), entity_depth)) {
        return *value;
    }
    let key = (name.to_owned(), entity_depth);
    if !active.insert(key.clone()) {
        return cap;
    }
    let Some(block) = d.block(name) else {
        active.remove(&key);
        return 0;
    };
    let mut total = 0usize;
    for entity in block.entities.iter() {
        let nested = match &entity.kind {
            EntityKind::Insert(insert) => insert_expansion_upper_bound(d, insert, entity_depth, cap, memo, active),
            _ => 0,
        };
        total = total.saturating_add(1).saturating_add(nested).min(cap);
        if total == cap {
            break;
        }
    }
    active.remove(&key);
    memo.insert(key, total);
    total
}

/// Build options.
#[derive(Clone, Debug)]
pub struct Options {
    /// Chord tolerance in world units (about half a pixel at the current zoom).
    pub tolerance: f64,
    /// Two pixels in world units: shorter dashes are continuous and filled font text uses
    /// screen-size detail. Zero keeps full detail (plots and image/vector exports).
    pub min_dash: f64,
    /// Draw text (off for very coarse previews).
    pub text: bool,
    /// Draw lines with their linetypes. Off, every line is drawn whole and continuous, as picking
    /// sees it: a dashed line is hit across its gaps, and a small pattern on a long line makes
    /// millions of dashes.
    pub linetypes: bool,
    /// Highlighted (selected) handles are not special here; the canvas overlays them.
    pub fill: bool,
    pub lineweights: bool,
    /// Height of the visible area in world units of the space being built. Relative point
    /// sizes (`PDSIZE` <= 0) are a percentage of it; 0 = the height of the space's extents.
    pub view_height: f64,
    /// Plot style table to draw with. When `None`, [`build`] of a layout whose page setup has
    /// "Display plot styles" on uses the layout's table; [`build_plot`] uses none.
    pub plot_style_table: Option<std::sync::Arc<cadcraft_doc::PlotStyleTable>>,
    /// The area being looked at (world units of the space being built), for on-screen builds:
    /// hatch patterns are only generated inside it. `None` = everywhere (plots, exports, picks).
    pub view: Option<Bounds2>,
    /// Draw objects with their transparency ([`DPrim::transparency`]). Off by default: the screen
    /// follows TRANSPARENCYDISPLAY, plots the page setup's "Plot transparency".
    pub transparency: bool,
    /// A layout viewport whose model-space contents are left out (its border is still drawn):
    /// the canvas draws the active viewport's contents itself, see [`viewport_content`].
    pub skip_viewport: Option<Handle>,
    /// Set from another thread to stop the build early (the canvas cancels a background
    /// rebuild the view no longer needs). A cancelled build returns a partial list, which the
    /// caller must discard; see [`Options::cancelled`].
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Options {
    /// Whether [`Options::cancel`] has been set.
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }
}

impl Default for Options {
    fn default() -> Self {
        Options {
            tolerance: 0.001,
            min_dash: 0.0,
            text: true,
            linetypes: true,
            fill: true,
            lineweights: false,
            view_height: 0.0,
            plot_style_table: None,
            view: None,
            skip_viewport: None,
            transparency: false,
            cancel: None,
        }
    }
}

/// Build the display list for a space. For a layout (paper space) this includes the model-space
/// geometry seen through each viewport, clipped to the viewport, and [`DisplayList::sheet`].
pub fn build(d: &Drawing, space: &Space, opts: &Options) -> DisplayList {
    build_space(d, space, opts, false)
}

/// Like [`build`] but for plotting: layers marked "do not plot" (such as `Defpoints`) are left out.
pub fn build_plot(d: &Drawing, space: &Space, opts: &Options) -> DisplayList {
    build_space(d, space, opts, true)
}

pub(crate) fn top_ctx<'a>(d: &'a Drawing, xf: Mat3, top: Handle, frozen: &'a [String], vp_colors: &'a [(String, Color)]) -> Ctx<'a> {
    Ctx {
        d,
        xf,
        block_color: Color::Index(7),
        block_layer: None,
        block_lw: Lineweight::Default,
        block_ltype: "Continuous".into(),
        depth: 0,
        top,
        frozen,
        vp_colors,
        lt_factor: 1.0,
        block_pstyle: None,
        block_transparency: 0,
    }
}

fn build_space(d: &Drawing, space: &Space, opts: &Options, plotting: bool) -> DisplayList {
    let opts = &with_point_view(d, space, opts);
    let styler = style_table(d, space, opts, plotting).map(|t| plotstyle::Styler::new(d, t));
    let mut b = Builder { list: DisplayList::default(), opts, plotting, styles: styler.as_ref(), named: None, shared_budget: None, transparency: 0 };
    if let Space::Paper(name) = space {
        b.list.sheet = paper::sheet(d, name);
    }
    if let Some(store) = d.space(space) {
        let paper = matches!(space, Space::Paper(_));
        #[cfg(not(target_arch = "wasm32"))]
        if !paper && !plotting && b.styles.is_none() && store.len() >= 2 * PARALLEL_CHUNK {
            build_parallel(&mut b, d, store);
            return b.list;
        }
        let mut viewports = 0usize;
        for e in store.iter() {
            if paper && let EntityKind::Viewport(vp) = &e.kind {
                if viewports < MAX_VIEWPORTS {
                    viewports += 1;
                    viewport(&mut b, d, e, vp);
                    if plotting {
                        take_viewport_layers(&mut b.list);
                    }
                }
                continue;
            }
            entity(&mut b, &top_ctx(d, Mat3::IDENTITY, e.handle, &[], &[]), e);
        }
    }
    // A layout is drawn in paper units, so its pens become geometry now.
    if let Some(sheet) = b.list.sheet {
        pen::apply(&mut b.list, sheet.unit_mm);
    }
    b.list
}

/// Top-level entities per task when a display list is built on all cores.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const PARALLEL_CHUNK: usize = 256;

/// Build model space for the screen on all cores: the entities are split into chunks built
/// side by side, then joined in drawing order, so the list is the same as a one-thread build.
/// The canvas rebuilds the whole list at every 2x zoom step (in the background since zooming
/// stopped blocking the UI) and after every edit; one core made both slow on real drawings. Plots, layouts and plot styles keep the one-thread
/// path (they run once, and their pens and per-primitive layers are built along the way).
#[cfg(not(target_arch = "wasm32"))]
fn build_parallel(b: &mut Builder, d: &Drawing, store: &cadcraft_doc::EntityStore) {
    use rayon::prelude::*;
    let ents: Vec<&std::sync::Arc<Entity>> = store.iter().collect();
    // A raced shared cap lets whichever Rayon worker arrives first claim the last block slots.
    // If the conservative count can cross the cap, build ordered chunks on this thread so the
    // same prefix is always retained. Drawings below the cap remain fully parallel.
    if block_expansion_upper_bound(d, store) > super::MAX_BLOCK_EXPANSION {
        let budget = std::sync::atomic::AtomicUsize::new(0);
        let (opts, plotting) = (b.opts, b.plotting);
        for chunk in ents.chunks(PARALLEL_CHUNK) {
            let mut part =
                Builder { list: DisplayList::default(), opts, plotting, styles: None, named: None, shared_budget: Some(&budget), transparency: 0 };
            for e in chunk {
                entity(&mut part, &top_ctx(d, Mat3::IDENTITY, e.handle, &[], &[]), e);
            }
            append(&mut b.list, part.list);
        }
        return;
    }
    let budget = std::sync::atomic::AtomicUsize::new(0);
    let (opts, plotting) = (b.opts, b.plotting);
    let parts: Vec<DisplayList> = ents
        .par_chunks(PARALLEL_CHUNK)
        .map(|chunk| {
            let mut part =
                Builder { list: DisplayList::default(), opts, plotting, styles: None, named: None, shared_budget: Some(&budget), transparency: 0 };
            for e in chunk {
                entity(&mut part, &top_ctx(d, Mat3::IDENTITY, e.handle, &[], &[]), e);
            }
            part.list
        })
        .collect();
    // One allocation per array for the joined list.
    b.list.prims.reserve(parts.iter().map(|p| p.prims.len()).sum());
    b.list.verts.reserve(parts.iter().map(|p| p.verts.len()).sum());
    b.list.tris.reserve(parts.iter().map(|p| p.tris.len()).sum());
    for part in parts {
        append(&mut b.list, part);
    }
}

/// The plot style table a build draws with: the one in `opts`, else (on screen) the layout's
/// when its page setup has "Display plot styles" on.
pub(crate) fn style_table(d: &Drawing, space: &Space, opts: &Options, plotting: bool) -> Option<std::sync::Arc<cadcraft_doc::PlotStyleTable>> {
    opts.plot_style_table.clone().or_else(|| match space {
        Space::Paper(n) if !plotting => d
            .layout(n)
            .filter(|l| l.page.show_plot_styles)
            .and_then(|l| cadcraft_doc::plot_style_table(d, &l.page.plot_style_table))
            .map(std::sync::Arc::new),
        _ => None,
    })
}

/// `opts` with a view height for relative point sizes: the extents height when none is given
/// and points draw as figures sized relative to the view.
pub(crate) fn with_point_view(d: &Drawing, space: &Space, opts: &Options) -> Options {
    let mut o = opts.clone();
    let relative = d.header.f64("PDSIZE", 0.0) <= 0.0 && point::sized(d.header.i64("PDMODE", 0));
    if !(o.view_height.is_finite() && o.view_height > 0.0) && relative {
        let ext = d.extents(space);
        o.view_height = if ext.is_empty() { 0.0 } else { ext.height().max(ext.width() * 1e-3) };
    }
    o
}

/// Build the display list for a set of loose entities (previews, rubber bands).
pub fn build_entities<'a, I: IntoIterator<Item = &'a Entity>>(d: &Drawing, ents: I, opts: &Options) -> DisplayList {
    let ents: Vec<&Entity> = ents.into_iter().collect();
    // Only previews holding points need the model extents for relative point sizes.
    let with_view;
    let opts = if ents.iter().any(|e| matches!(e.kind, EntityKind::Point(_))) {
        with_view = with_point_view(d, &Space::Model, opts);
        &with_view
    } else {
        opts
    };
    let mut b = Builder { list: DisplayList::default(), opts, plotting: false, styles: None, named: None, shared_budget: None, transparency: 0 };
    for e in ents {
        entity(&mut b, &top_ctx(d, Mat3::IDENTITY, e.handle, &[], &[]), e);
    }
    b.list
}

fn build_entity_with_budget(d: &Drawing, e: &Entity, opts: &Options, budget: &std::sync::atomic::AtomicUsize) -> DisplayList {
    let with_view;
    let opts = if matches!(e.kind, EntityKind::Point(_)) {
        with_view = with_point_view(d, &Space::Model, opts);
        &with_view
    } else {
        opts
    };
    let mut b =
        Builder { list: DisplayList::default(), opts, plotting: false, styles: None, named: None, shared_budget: Some(budget), transparency: 0 };
    entity(&mut b, &top_ctx(d, Mat3::IDENTITY, e.handle, &[], &[]), e);
    b.list
}

/// After a viewport is appended to a plot: the layers its contents carry in their handles
/// (see [`entity`]) go to [`DisplayList::prim_layers`], and the handles back to
/// [`VIEWPORT_CONTENT`]. The viewport's border was tagged when it was drawn.
///
/// Text glyphs seen through the viewport (marked in the handles' high half) keep their outline
/// primitive when it came through the clip whole; a glyph cut by the viewport edge stays
/// geometry.
fn take_viewport_layers(list: &mut DisplayList) {
    let start = list.prim_layers.len();
    for i in start..list.prims.len() {
        let Some(p) = list.prims.get(i).copied() else { break };
        list.prim_layers.push(u32::try_from(p.handle.0 & 0xFFFF_FFFF).unwrap_or(NO_LAYER));
        let bounds = Bounds2::from_points(list.points(&p).iter().copied());
        if let Some(t) = (p.handle.0 >> 32).checked_sub(1).and_then(|k| list.texts.get_mut(k as usize)) {
            let same = |a: Vec2, b: Vec2| a.dist(b) <= 1e-9 * (1.0 + a.len());
            let whole = t.prim == PlotText::NONE
                && p.kind == Kind::Tris
                && p.len == t.verts
                && same(bounds.min, t.bounds.min)
                && same(bounds.max, t.bounds.max);
            // A second piece (or a cut one) means the clip changed the glyph.
            t.prim = if whole { u32::try_from(i).unwrap_or(PlotText::NONE) } else { PlotText::CLIPPED };
        }
        if let Some(p) = list.prims.get_mut(i) {
            p.handle = VIEWPORT_CONTENT;
        }
    }
}
