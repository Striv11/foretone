# Plan

Seven-day arc. Every milestone has acceptance criteria that can be checked by a test or a command.

| Day | Milestone | Acceptance criteria | Status |
|-----|-----------|---------------------|--------|
| 1 | Core model + scaffold | `foretone-core` (no deps): recurrent predictive-coding learner with local update rules; tests show a 3-note and an 8-note cycle reach >90% next-note accuracy, surprise drops, violations spike surprise, seeded runs are deterministic; `cargo run --example melody` prints falling surprise; `bash check.sh` and CI pass; landing page deployed | [x] |
| 2 | Better model | Layered hierarchy (2 latent layers, slower top layer); precision weighting; benchmark on a 5-song corpus: mean surprise on repeat 2 is lower than the Day-1 model (`cargo run --example bench` prints a table); unit tests for gradient correctness via finite-difference check of the energy | [x] |
| 3 | Data & encoders | Melody input as note-name text and MIDI (`.mid` parser written in-crate, tests on fixture bytes); text/character mode; model state (de)serialization to a compact binary and round-trips bit-exactly | [x] |
| 4 | WASM bindings | `foretone-wasm` crate with wasm-bindgen: `observe`, `predict`, `reset`, `export/import`; pages.yml builds it; a Node smoke test loads the .wasm and replays a melody with the same output as native | [ ] |
| 5 | Playground UI | Responsive piano-roll page: play notes by tap/keyboard, live next-note probability bars, surprise sparkline, Web Audio playback, empty state, light/dark | [ ] |
| 6 | Visualization & polish | Latent-state heatmap, prediction-error trace, "dream" mode (free-running generation), keyboard shortcuts, loading/error states, screenshots in README; Playwright e2e for the core flows | [ ] |
| 7 | Hardening | Performance budget (>= 50k observe/s at 32 hidden units, benchmark in CI), README demo section final, docs for design decisions, v1 tag | [ ] |

## Stretch
- Compare against an n-gram and a small LSTM baseline on the same corpus.
- Share a trained model via URL fragment.
