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
