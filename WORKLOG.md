# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-23 — Day 1
- Chose a Rust→WASM predictive-coding melody learner from the Cognition & AI backlog (new language and domain versus recent TypeScript projects).
- Built `foretone-core`: recurrent latent state, inference by error relaxation, local W/A updates, softmax predictions, surprise. Dependency-free with its own RNG.
- Tests: 3-note and 8-note cycles exceed 90% next-note accuracy online, surprise falls, a violated note spikes surprise >5x, determinism. First attempt passed without tuning.
- Added CI, a Pages workflow with a placeholder landing page, and a CLI demo example.
- Next: Day 2, two-layer hierarchy, precision weighting, finite-difference gradient test, corpus benchmark.

## 2026-11-23 — Day 2 (second entry)
- Added `HierModel`: fast layer `r1` predicts the note, slow layer `r2` (smaller inference step, own dynamics) feeds `r1`'s prior top-down; explicit precisions pi1/pi2. All updates stay local.
- Gradient correctness: central finite-difference test of the analytic energy gradient for both layers passes.
- `examples/bench.rs`: 5 songs, mean surprise on repeat 2. First run with default rates showed the hierarchy *worse* (2.03 vs 2.00, both near ln 8, i.e. nothing learned in one pass). A small sweep showed fast learning is needed; at best-tuned rates for each model: Day-1 1.702 vs hier 1.634. Small win, loses on 2 of 5 songs; the main benefit is stability at high learning rates.
- Next: Day 3 — note-name text and MIDI parsing, text mode, binary (de)serialization with bit-exact round trip.

## 2026-11-24 — Day 3
- Added `encode` (note-name parsing, `note_name`, generic sorted `Vocab`, char vocab for text mode), `midi` (in-crate SMF parser: running status, meta/sysex, multi-track time merge, bounds-checked) and `codec` + `to_bytes/from_bytes` for `Model` and `HierModel`.
- Tests (`tests/data.rs`, 12): note parsing edge cases, name round trip for all 128 pitches, fixture-byte MIDI files built in-test, truncation at every byte offset, bit-exact round trip including a NaN payload and identical continuation after restore, hostile size headers.
- Learned: my first MIDI->model test was too optimistic (12 repeats isn't enough for 0.3 nats); the single-layer model needs ~50 cycles. Also guarded length*size overflow on decode since WASM is 32-bit.
- Next: Day 4 — `foretone-wasm` crate with wasm-bindgen (observe/predict/reset/export/import), pages build, Node smoke test matching native output.

## 2026-11-25 — Day 4
- Added `foretone-wasm`: a dependency-free C-ABI cdylib (`ft_new/observe/predict/reset/export/import`, `ft_alloc/dealloc`), a safe `Session` core, and `web/foretone.mjs` wrapper. Chose C ABI over wasm-bindgen deliberately: tiny surface, no CLI to install in CI, plain `WebAssembly.instantiate` in browser and Node. Documented in PLAN/README.
- Tests: 6 native tests drive the exported functions (bit-exact vs `Model`, seed halves, null/invalid inputs return errors, export/import continues identically, failed import leaves model intact, reset keeps weights). `web/test/run-smoke.sh` builds the .wasm and replays 160 notes in Node against a native reference, plus export/import and error paths. check.sh runs it when the wasm target and Node exist (mandatory in CI); CI and pages workflows now build the wasm.
- Learned: wasm output is not bit-identical to native (libm differences), though printed values agree to 1e-6; so the criterion is "same within tolerance". Also fixed my own flawed reset test (C is ambiguous in the tune; use C D E -> G).
- Next: Day 5 — playground UI using web/foretone.mjs (piano roll, probability bars, surprise sparkline, Web Audio, light/dark).

## 2026-11-26 — Day 5
- Built the playground: piano keys (tap, pointer, keyboard A–K, accessible buttons), live next-note probability bars (the model's top guess highlighted), surprise sparkline, sequence input, Web Audio triangle-wave playback with a sound toggle, empty state, load-error state, light/dark via `prefers-color-scheme`.
- Split pure logic (`playground.mjs`) from DOM/audio (`app.mjs`) so it is testable; `web/test/playground.mjs` covers key mapping, note parsing, sparkline math, history cap, violation spike, bad-input errors. check.sh now runs it whenever Node exists.
- Learned: the dev environment had CI=true but no wasm32 target, so check.sh failed until I installed the target; the full wasm smoke test now passes here too. I could not drive a real browser, so the DOM layer is untested beyond code review; Day 6 Playwright e2e should cover it.
- Next: Day 6 — latent heatmap (needs a wasm export for r), error trace, dream mode, shortcuts, screenshots, Playwright e2e.

## 2026-11-27 — Day 6 (part 1)
- Fixed the environment first: `check.sh` failed under CI=true because the wasm32 target was not installed; installed it, full gate passes.
- Core: `Model::latent()` and `Model::dream(n)` (free-running argmax generation on a clone, so weights/state stay untouched). WASM: `ft_latent`, `ft_hidden`, `ft_dream` with argument validation; wrapper gets `latent()` and `dream(n)`.
- Playground: latent heatmap (last 24 notes), prediction-error trace, Dream button with audio, keyboard shortcuts 1/2/3, guard against dreaming before anything is learned. Pure helpers (`shortcut`, `heatColor`, `predictionError`, dream/latent history) are unit-tested in Node.
- Tests: native dream test (replays the 8-note tune twice, model bytes unchanged), ABI test (null/oversize args, read-only dream), Node smoke test of latent+dream through the real .wasm.
- Learned: 160 notes was too little training for the dream to reproduce the tune (it collapsed to C C C…); 800 notes works, consistent with Day 3's finding that the single layer needs many cycles.
- Not done: README screenshots and Playwright e2e. I have no browser here, and the new DOM code is still only reviewed, not driven. Day 6 stays open for that.
- Next: finish Day 6 (e2e + screenshots) if a browser is available, otherwise move to Day 7 (50k observe/s benchmark, v1).
