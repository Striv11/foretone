# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-23 — Day 1
- Chose a Rust→WASM predictive-coding melody learner from the Cognition & AI backlog (new language and domain versus recent TypeScript projects).
- Built `foretone-core`: recurrent latent state, inference by error relaxation, local W/A updates, softmax predictions, surprise. Dependency-free with its own RNG.
- Tests: 3-note and 8-note cycles exceed 90% next-note accuracy online, surprise falls, a violated note spikes surprise >5x, determinism. First attempt passed without tuning.
- Added CI, a Pages workflow with a placeholder landing page, and a CLI demo example.
- Next: Day 2, two-layer hierarchy, precision weighting, finite-difference gradient test, corpus benchmark.
