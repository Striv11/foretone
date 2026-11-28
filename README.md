# foretone

> A brain-inspired model that learns melodies online and shows what it expects to hear next, and how surprised it is when you break the pattern. Rust, compiled to WebAssembly.

![CI](https://github.com/Striv11/foretone/actions/workflows/ci.yml/badge.svg)

**Live demo:** https://striv11.github.io/foretone/ (interactive playground; needs the wasm build, see below)

## Problem
Most sequence models are trained offline with backpropagation through time on big datasets. The brain appears to learn continuously, from a single stream, using only local signals: each region predicts its input and learns from the *prediction error*. It is hard to build intuition for that. foretone makes it tangible: play a few notes and watch the model learn the pattern in real time, with its expectations and surprise visualized.

## Motivation
Predictive coding is a compact, biologically motivated learning theory, and it is easy to get subtly wrong. Implementing it from scratch in Rust (no ML crates) explores: inference by energy minimisation, local Hebbian-style weight updates, online learning with no replay buffer, and running the same core natively and in the browser via WASM, with no data leaving the device.

## Architecture
```
 note ──► [ observe ]
            │  1. prior   r̂ = A·f(r_prev)
            │  2. infer   relax r:  Δr ∝ f'(r)⊙Wᵀe_x − π·(r − r̂)
            │  3. learn   W += η e_x f(r)ᵀ    A += η (r − r̂) f(r_prev)ᵀ
            ▼
        next-note distribution = softmax(W·f(A·f(r)))     surprise = −ln p(actual)

 crates/foretone-core   pure Rust model (no deps)           ← Day 1
 crates/foretone-wasm   C-ABI wasm module + web/foretone.mjs   ← Day 4
 web/                   static playground, GitHub Pages     ← Days 5-6
```
Invariants: deterministic for a given seed; all learning rules are local (error × presynaptic activity); no BPTT.

## Setup
Prerequisites: a stable Rust toolchain.
```
bash check.sh        # fmt, clippy, tests
```

## Demo
Status: **CLI demo and browser playground available.** Browser: build the wasm (`cargo build --release -p foretone-wasm --target wasm32-unknown-unknown`), copy `foretone_wasm.wasm` to `web/pkg/`, then serve `web/` over http (e.g. `python3 -m http.server -d web`). Tap the keys or use A–K; type a sequence and press Play. Shortcuts: `1` dream (the model free-runs 16 notes, without learning from them), `2` new phrase, `3` reset. Below the bars you get a prediction-error trace and a heatmap of the latent units over the last 24 notes.
```
cargo run --release -p foretone-core --example melody
```
Expected output (abridged): mean surprise falls from ~1.3 to ~0.03 nats over 60 epochs of the tune C D E G E D C C, next-note hits reach 31/31, and the free-running "dream" replays `C D E G E D C C`.

## Design decisions & trade-offs
- Softmax sensory prediction with cross-entropy error (`x − p`) instead of squared error: still a local error signal, but gives proper probabilities and stable learning.
- `Model` is the single-layer Day-1 baseline; `HierModel` adds a slow top layer `r2` that supplies top-down context to the fast layer, with explicit precisions (`pi1`, `pi2`). Its energy gradient is verified against finite differences in `tests/hier.rs`.
- Performance budget: `cargo run --release -p foretone-core --example throughput` measures `observe` calls/s at 32 hidden units (vocab 12). On the dev machine: single layer ~105k/s, hierarchical (32+16) ~32k/s. `check.sh` (and so CI) fails if the single layer drops below 50k/s; the hierarchical model is reported but not gated.
- Honest benchmark (`cargo run --release -p foretone-core --example bench`, mean surprise on the 2nd playthrough of 5 songs, both models at their best-tuned learning rates): Day-1 1.702 nats, hierarchy 1.634. The gain is modest and not uniform (the hierarchy loses on 2 of 5 songs); its clearer advantage is stability at high learning rates, where the single layer diverges (3.9 nats at lr_a=0.5).
- Inputs: `parse_melody("C4 D4 E4")` for note names, `midi_melody(bytes)` for `.mid` files (in-crate SMF parser: running status, meta/sysex, multi-track merge), `char_vocab` for text mode; `Vocab` maps tokens to dense indices.
- `Model::to_bytes/from_bytes` (and `HierModel`) use a tiny versioned binary format (`FTN1` magic) storing raw float bits, so a restored model continues bit-exactly; corrupt, truncated or oversized input returns an error rather than panicking or over-allocating.
- WASM uses a plain C ABI (`ft_new/ft_observe/ft_predict/ft_reset/ft_export/ft_import`, host-managed buffers) rather than wasm-bindgen: the surface is only numbers and byte buffers, so the module has zero dependencies and no extra CLI, and `web/foretone.mjs` loads it with standard `WebAssembly.instantiate`. Errors return sentinel values (never panic across the boundary). The wasm build is not bit-identical to native (Rust's libm vs the system libm for `tanh`/`exp`); the smoke test (`bash web/test/run-smoke.sh`) replays 160 notes and checks agreement within 1e-3, and `ft_latent/ft_hidden/ft_dream` expose the latent state and a read-only free-running generator (`Model::dream` works on a clone, so the model is provably untouched). `ft_*` is checked bit-for-bit against the native model in `crates/foretone-wasm/tests/api.rs`.
- The playground keeps DOM/audio code (`web/app.mjs`) separate from pure logic (`web/playground.mjs`: key map, note parsing, history, sparkline), which `web/test/playground.mjs` tests in Node with a fake model; check.sh runs it.
- Own xorshift RNG keeps the core dependency-free and WASM-friendly.

## Roadmap
- [x] Core recurrent predictive-coding learner with tests (Day 1)
- [x] Hierarchy and precision weighting (Day 2)
- [x] Note-name / MIDI / text encoders, binary serialization (Day 3)
- [x] WASM bindings (Day 4)
- [x] Playground UI (Day 5)
- [~] Visualizations (latent heatmap, error trace, dream, shortcuts) done; e2e tests and screenshots pending (Day 6)
- [ ] Performance budget and v1 (Day 7)

## License
MIT. See [LICENSE](LICENSE).
