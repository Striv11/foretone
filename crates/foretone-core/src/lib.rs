//! Online predictive-coding sequence learner.
//!
//! A single recurrent latent layer `r` is inferred at every step by minimising
//! two prediction errors (the "energy"):
//!
//! * sensory error `e_x = x_t - softmax(W f(r_t))` (top-down prediction of the input)
//! * temporal error `e_r = r_t - A f(r_{t-1})` (the dynamics prior)
//!
//! Inference relaxes `r` for a few gradient steps; learning then updates `W`
//! and `A` with *local* rules (error times presynaptic activity). There is no
//! backpropagation through time: this is the brain-inspired part.

mod codec;
mod encode;
mod hier;
mod midi;
mod rng;

pub use codec::DecodeError;
pub use encode::{char_vocab, note_name, parse_melody, parse_note, ParseError, Vocab};
pub use hier::{HierConfig, HierModel};
pub use midi::{midi_melody, parse_midi, MidiError, NoteOn};
pub use rng::Rng;

/// Hyper-parameters of the model.
#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub hidden: usize,
    /// Inference iterations per observed symbol.
    pub infer_steps: usize,
    /// Inference step size.
    pub lr_infer: f32,
    /// Learning rate for the generative weights `W`.
    pub lr_w: f32,
    /// Learning rate for the transition weights `A`.
    pub lr_a: f32,
    /// Weight of the temporal prior relative to the sensory error.
    pub prior_precision: f32,
    pub seed: u64,
}

impl Config {
    pub fn new(vocab: usize, hidden: usize) -> Self {
        Config {
            vocab,
            hidden,
            infer_steps: 12,
            lr_infer: 0.2,
            lr_w: 0.1,
            lr_a: 0.05,
            prior_precision: 0.5,
            seed: 42,
        }
    }
}

/// Result of observing one symbol.
#[derive(Clone, Debug)]
pub struct Step {
    /// Probability the model had assigned to the symbol that actually arrived.
    pub p_actual: f32,
    /// Surprise in nats: `-ln p_actual`.
    pub surprise: f32,
    /// Distribution over the *next* symbol, after observing this one.
    pub next: Vec<f32>,
}

#[derive(Clone)]
pub struct Model {
    cfg: Config,
    /// Generative weights, `vocab x hidden`, row-major.
    w: Vec<f32>,
    /// Transition weights, `hidden x hidden`, row-major.
    a: Vec<f32>,
    r: Vec<f32>,
    /// Current prediction of the next symbol.
    pred: Vec<f32>,
}

impl Model {
    /// Serialize config, weights and latent state to a compact binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        use codec::*;
        let c = &self.cfg;
        let mut w = Writer::new(KIND_MODEL);
        w.u32(c.vocab);
        w.u32(c.hidden);
        w.u32(c.infer_steps);
        for v in [c.lr_infer, c.lr_w, c.lr_a, c.prior_precision] {
            w.f32(v);
        }
        w.u64(c.seed);
        for v in [&self.w, &self.a, &self.r, &self.pred] {
            w.f32s(v);
        }
        w.0
    }

    /// Inverse of [`Model::to_bytes`]; the restored model continues bit-exactly.
    pub fn from_bytes(b: &[u8]) -> Result<Self, DecodeError> {
        use codec::*;
        let mut r = Reader::new(b, KIND_MODEL)?;
        let (vocab, hidden, infer_steps) = (r.u32()?, r.u32()?, r.u32()?);
        if vocab < 2 || hidden < 1 {
            return Err(DecodeError::Invalid("sizes"));
        }
        let cfg = Config {
            vocab,
            hidden,
            infer_steps,
            lr_infer: r.f32()?,
            lr_w: r.f32()?,
            lr_a: r.f32()?,
            prior_precision: r.f32()?,
            seed: r.u64()?,
        };
        let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(DecodeError::Invalid("sizes"));
        let w = r.f32s(mul(vocab, hidden)?)?;
        let a = r.f32s(mul(hidden, hidden)?)?;
        let rr = r.f32s(hidden)?;
        let pred = r.f32s(vocab)?;
        r.finish()?;
        Ok(Model {
            cfg,
            w,
            a,
            r: rr,
            pred,
        })
    }
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter_mut().for_each(|v| *v /= s);
    e
}

pub(crate) fn matvec(m: &[f32], rows: usize, cols: usize, v: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|i| {
            m[i * cols..(i + 1) * cols]
                .iter()
                .zip(v)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

impl Model {
    pub fn new(cfg: Config) -> Self {
        assert!(cfg.vocab >= 2 && cfg.hidden >= 1, "invalid sizes");
        let mut rng = Rng::new(cfg.seed);
        let mut init = |n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|_| (rng.next_f32() - 0.5) * scale).collect()
        };
        let w = init(cfg.vocab * cfg.hidden, 0.5);
        let a = init(cfg.hidden * cfg.hidden, 0.5);
        let uniform = vec![1.0 / cfg.vocab as f32; cfg.vocab];
        Model {
            r: vec![0.0; cfg.hidden],
            pred: uniform,
            w,
            a,
            cfg,
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Current distribution over the next symbol.
    pub fn predict(&self) -> &[f32] {
        &self.pred
    }

    /// Current latent activity `r` (length `hidden`), for visualization.
    pub fn latent(&self) -> &[f32] {
        &self.r
    }

    /// Free-run `n` symbols by repeatedly following the most likely next symbol.
    /// Works on a copy, so the model's state and weights are left untouched.
    pub fn dream(&self, n: usize) -> Vec<usize> {
        let mut m = self.clone();
        (0..n)
            .map(|_| {
                let s = argmax(m.predict());
                m.observe(s);
                s
            })
            .collect()
    }

    /// Forget the latent state (not the learned weights), e.g. at a song boundary.
    pub fn reset_state(&mut self) {
        self.r.iter_mut().for_each(|v| *v = 0.0);
        self.pred = vec![1.0 / self.cfg.vocab as f32; self.cfg.vocab];
    }

    /// Observe `symbol`, infer the latent state, learn, and predict the next one.
    pub fn observe(&mut self, symbol: usize) -> Step {
        let (v, h) = (self.cfg.vocab, self.cfg.hidden);
        assert!(symbol < v, "symbol out of range");
        let p_actual = self.pred[symbol].max(1e-9);

        let f_prev: Vec<f32> = self.r.iter().map(|x| x.tanh()).collect();
        let prior = matvec(&self.a, h, h, &f_prev);
        let mut r = prior.clone();

        // Inference: relax r to reduce sensory + temporal prediction error.
        let mut e_x = vec![0.0; v];
        let mut e_r = vec![0.0; h];
        for _ in 0..self.cfg.infer_steps {
            let f: Vec<f32> = r.iter().map(|x| x.tanh()).collect();
            let p = softmax(&matvec(&self.w, v, h, &f));
            for i in 0..v {
                e_x[i] = (i == symbol) as u8 as f32 - p[i];
            }
            for j in 0..h {
                e_r[j] = r[j] - prior[j];
            }
            for j in 0..h {
                let back: f32 = (0..v).map(|i| self.w[i * h + j] * e_x[i]).sum();
                let d = 1.0 - f[j] * f[j];
                r[j] += self.cfg.lr_infer * (d * back - self.cfg.prior_precision * e_r[j]);
            }
        }

        // Learning: local updates using the converged errors.
        let f: Vec<f32> = r.iter().map(|x| x.tanh()).collect();
        let p = softmax(&matvec(&self.w, v, h, &f));
        for i in 0..v {
            e_x[i] = (i == symbol) as u8 as f32 - p[i];
        }
        for j in 0..h {
            e_r[j] = r[j] - prior[j];
        }
        for i in 0..v {
            for j in 0..h {
                self.w[i * h + j] += self.cfg.lr_w * e_x[i] * f[j];
            }
        }
        for j in 0..h {
            for k in 0..h {
                self.a[j * h + k] += self.cfg.lr_a * e_r[j] * f_prev[k];
            }
        }

        // Predict the next symbol from the dynamics prior on the new state.
        let next_prior = matvec(&self.a, h, h, &f);
        let fn_: Vec<f32> = next_prior.iter().map(|x| x.tanh()).collect();
        self.pred = softmax(&matvec(&self.w, v, h, &fn_));
        self.r = r;

        Step {
            p_actual,
            surprise: -p_actual.ln(),
            next: self.pred.clone(),
        }
    }
}

/// Index of the largest value.
pub fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &x)| if x > b.1 { (i, x) } else { b },
        )
        .0
}
