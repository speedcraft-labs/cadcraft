# Issue 95 hatch fill experiment: no measurable gain

This branch preserves the active-edge-table experiment and its measurements. The app source commit is `5875e8323ce68033ce3694e5d4da8fb93d8e4a06`; the fork baseline is `3ddb1820d9a35d09991a2d22202326ad395ed1e5`, and the upstream baseline is `a3d9348f63c2ad35391cc93980dc7ee7366f2ad6`.

## Payload and method

The journey renders `cadcraft-real-drawings/wall-brick.dxf`, a 10.5 MB real DXF with 17,237 entities and 380 patterned hatches. It builds the model-space display list with the same 1,600 x 1,000 zoom-extents profile and hashes the full display list. Each comparison was publishable, which means the output SHA-256 identities matched. Each side used 10 paired runs and 2,000 bootstrap resamples (seed 12648430). The primary measurement is the adapter's `CRAFT_JOURNEY_NS` render-build duration.

| Machine | Comparison | Old median | New median | New / old (95% CI) |
| --- | --- | ---: | ---: | ---: |
| Agentbox, Ryzen 9 5900XT | Upstream `a3d…` → candidate | 17.171 ms | 16.682 ms | 0.972 [0.901, 1.040] |
| Agentbox, Ryzen 9 5900XT | Fork `3ddb…` → candidate | 19.312 ms | 19.002 ms | 0.984 [0.909, 1.084] |
| Mac mini, Apple M6 | Upstream `a3d…` → candidate | 5.227 ms | 5.372 ms | 1.028 [0.982, 1.056] |
| Mac mini, Apple M6 | Fork `3ddb…` → candidate | 5.499 ms | 5.421 ms | 0.986 [0.959, 1.047] |

All four intervals include 1.0, and every point estimate is within 2.8% of parity. The change does not meet the 5% real-world gain threshold, so it is not being landed.

## Profile and validation

A profile of 20 render builds on the same payload measured 224.519 ms total (11.226 ms per build). `hatch::draw` accounted for 14.44% of child cycles in the renderer call path; even-odd triangulation accounted for 7.80%, and `Builder::tris_at` for 6.63%. This confirms hatch rendering is active in the typical payload, while the paired comparisons show no measurable gain from the edge-table change.

`cargo xtask ci` passed all seven app checks. The GPU stencil test passed when run under the shared GPU lock. The craft-libs `cargo test --workspace` run had one unrelated H.264 `rapid_resets_do_not_hang` timeout while multiple decoder robustness tests ran concurrently; that same test passed alone in 227.01 seconds. The full-workspace failure log and isolated rerun evidence are in the ticket NAS backup.
