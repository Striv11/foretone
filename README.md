# foretone

> A brain-inspired model that learns melodies online and shows what it expects to hear next, and how surprised it is when you break the pattern. Rust, compiled to WebAssembly.

![CI](https://github.com/Striv11/foretone/actions/workflows/ci.yml/badge.svg)

**Live demo:** https://striv11.github.io/foretone/ (landing page now; interactive playground planned, Day 5)

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
 crates/foretone-wasm   wasm-bindgen bindings               ← Day 4
 web/                   static playground, GitHub Pages     ← Days 5-6
```
Invariants: deterministic for a given seed; all learning rules are local (error × presynaptic activity); no BPTT.

## Setup
Prerequisites: a stable Rust toolchain.
```
bash check.sh        # fmt, clippy, tests
```

## Demo
Status: **CLI demo available; browser demo planned.**
```
cargo run --release -p foretone-core --example melody
```
Expected output (abridged): mean surprise falls from ~1.3 to ~0.03 nats over 60 epochs of the tune C D E G E D C C, next-note hits reach 31/31, and the free-running "dream" replays `C D E G E D C C`.

## Design decisions & trade-offs
- Softmax sensory prediction with cross-entropy error (`x − p`) instead of squared error: still a local error signal, but gives proper probabilities and stable learning.
- Single latent layer in v0; a two-layer hierarchy arrives on Day 2.
- Own xorshift RNG keeps the core dependency-free and WASM-friendly.

## Roadmap
- [x] Core recurrent predictive-coding learner with tests (Day 1)
- [ ] Hierarchy and precision weighting (Day 2)
- [ ] MIDI / text encoders, serialization (Day 3)
- [ ] WASM bindings (Day 4)
- [ ] Playground UI (Day 5)
- [ ] Visualizations, e2e tests, screenshots (Day 6)
- [ ] Performance budget and v1 (Day 7)

## License
MIT. See [LICENSE](LICENSE).
