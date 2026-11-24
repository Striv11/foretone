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
