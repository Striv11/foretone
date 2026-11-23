//! Two-layer predictive-coding hierarchy with precision weighting.
//!
//! * `r1` (fast) predicts the symbol: `p = softmax(W f(r1))`.
//! * `r2` (slow) predicts `r1` as extra context: `prior1 = A1 f(r1_prev) + B f(r2)`.
//! * `r2` has its own dynamics prior `prior2 = A2 f(r2_prev)` and a smaller
//!   inference step, so it drifts slowly and captures phrase-level context.
//!
//! Energy minimised per symbol (π are precisions, i.e. inverse variances):
//!
//! `E = CE(x, p) + π1/2 |r1 - prior1|² + π2/2 |r2 - prior2|²`
//!
//! Every update below is local: an error times the activity of the layer it
//! projects from. No backpropagation through time.

use crate::{matvec, softmax, Rng};

#[derive(Clone, Debug)]
pub struct HierConfig {
    pub vocab: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub infer_steps: usize,
    pub lr_infer: f32,
    /// Multiplier on the inference step of the top layer (< 1 makes it slow).
    pub top_slowness: f32,
    pub lr_w: f32,
    pub lr_a: f32,
    pub lr_top: f32,
    /// Precision of the layer-1 temporal/top-down prior.
    pub pi1: f32,
    /// Precision of the layer-2 dynamics prior.
    pub pi2: f32,
    pub seed: u64,
}

impl HierConfig {
    pub fn new(vocab: usize, hidden1: usize, hidden2: usize) -> Self {
        HierConfig {
            vocab,
            hidden1,
            hidden2,
            infer_steps: 12,
            lr_infer: 0.2,
            top_slowness: 0.3,
            lr_w: 1.0,
            lr_a: 0.2,
            lr_top: 0.2,
            pi1: 0.5,
            pi2: 1.0,
            seed: 42,
        }
    }
}

/// Result of observing one symbol (same shape as the single-layer model).
pub use crate::Step;

pub struct HierModel {
    cfg: HierConfig,
    w: Vec<f32>,  // vocab x h1
    a1: Vec<f32>, // h1 x h1
    b: Vec<f32>,  // h1 x h2 (top-down context)
    a2: Vec<f32>, // h2 x h2
    r1: Vec<f32>,
    r2: Vec<f32>,
    pred: Vec<f32>,
}

fn tanh_v(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x.tanh()).collect()
}

impl HierModel {
    pub fn new(cfg: HierConfig) -> Self {
        assert!(cfg.vocab >= 2 && cfg.hidden1 >= 1 && cfg.hidden2 >= 1);
        let (v, h1, h2) = (cfg.vocab, cfg.hidden1, cfg.hidden2);
        let mut rng = Rng::new(cfg.seed);
        let mut init =
            |n: usize, s: f32| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() - 0.5) * s).collect() };
        let w = init(v * h1, 0.5);
        let a1 = init(h1 * h1, 0.5);
        let b = init(h1 * h2, 0.5);
        let a2 = init(h2 * h2, 0.5);
        HierModel {
            r1: vec![0.0; h1],
            r2: vec![0.0; h2],
            pred: vec![1.0 / v as f32; v],
            w,
            a1,
            b,
            a2,
            cfg,
        }
    }

    pub fn config(&self) -> &HierConfig {
        &self.cfg
    }

    pub fn predict(&self) -> &[f32] {
        &self.pred
    }

    /// Current (fast, slow) latent states, for visualisation.
    pub fn latents(&self) -> (&[f32], &[f32]) {
        (&self.r1, &self.r2)
    }

    pub fn reset_state(&mut self) {
        self.r1.iter_mut().for_each(|v| *v = 0.0);
        self.r2.iter_mut().for_each(|v| *v = 0.0);
        self.pred = vec![1.0 / self.cfg.vocab as f32; self.cfg.vocab];
    }

    /// Priors for layers 1 and 2 given the previous states.
    fn priors(&self, r1_prev: &[f32], r2_prev: &[f32], r2: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (h1, h2) = (self.cfg.hidden1, self.cfg.hidden2);
        let mut p1 = matvec(&self.a1, h1, h1, &tanh_v(r1_prev));
        let top = matvec(&self.b, h1, h2, &tanh_v(r2));
        p1.iter_mut().zip(&top).for_each(|(a, b)| *a += b);
        let p2 = matvec(&self.a2, h2, h2, &tanh_v(r2_prev));
        (p1, p2)
    }

    /// Energy and its gradients w.r.t. `(r1, r2)` for fixed priors' inputs.
    /// Exposed (hidden) so tests can check the analytic gradient numerically.
    #[doc(hidden)]
    pub fn energy_and_grad(
        &self,
        symbol: usize,
        r1_prev: &[f32],
        r2_prev: &[f32],
        r1: &[f32],
        r2: &[f32],
    ) -> (f32, Vec<f32>, Vec<f32>) {
        let (v, h1, h2) = (self.cfg.vocab, self.cfg.hidden1, self.cfg.hidden2);
        let (prior1, prior2) = self.priors(r1_prev, r2_prev, r2);
        let f1 = tanh_v(r1);
        let f2 = tanh_v(r2);
        let p = softmax(&matvec(&self.w, v, h1, &f1));
        let e1: Vec<f32> = (0..h1).map(|j| r1[j] - prior1[j]).collect();
        let e2: Vec<f32> = (0..h2).map(|j| r2[j] - prior2[j]).collect();
        let sq = |e: &[f32]| e.iter().map(|x| x * x).sum::<f32>();
        let energy = -p[symbol].max(1e-12).ln()
            + 0.5 * self.cfg.pi1 * sq(&e1)
            + 0.5 * self.cfg.pi2 * sq(&e2);

        let mut g1 = vec![0.0; h1];
        for j in 0..h1 {
            let back: f32 = (0..v)
                .map(|i| self.w[i * h1 + j] * ((i == symbol) as u8 as f32 - p[i]))
                .sum();
            g1[j] = -(1.0 - f1[j] * f1[j]) * back + self.cfg.pi1 * e1[j];
        }
        let mut g2 = vec![0.0; h2];
        for k in 0..h2 {
            let down: f32 = (0..h1).map(|j| self.b[j * h2 + k] * e1[j]).sum();
            g2[k] = self.cfg.pi2 * e2[k] - self.cfg.pi1 * (1.0 - f2[k] * f2[k]) * down;
        }
        (energy, g1, g2)
    }

    pub fn observe(&mut self, symbol: usize) -> Step {
        let (v, h1, h2) = (self.cfg.vocab, self.cfg.hidden1, self.cfg.hidden2);
        assert!(symbol < v, "symbol out of range");
        let p_actual = self.pred[symbol].max(1e-9);

        let r1_prev = self.r1.clone();
        let r2_prev = self.r2.clone();
        let (prior1, _) = self.priors(&r1_prev, &r2_prev, &r2_prev);
        let mut r1 = prior1;
        let mut r2 = self.priors(&r1_prev, &r2_prev, &r2_prev).1;

        for _ in 0..self.cfg.infer_steps {
            let (_, g1, g2) = self.energy_and_grad(symbol, &r1_prev, &r2_prev, &r1, &r2);
            for j in 0..h1 {
                r1[j] -= self.cfg.lr_infer * g1[j];
            }
            for k in 0..h2 {
                r2[k] -= self.cfg.lr_infer * self.cfg.top_slowness * g2[k];
            }
        }

        // Local learning at the converged state.
        let (prior1, prior2) = self.priors(&r1_prev, &r2_prev, &r2);
        let f1 = tanh_v(&r1);
        let f2 = tanh_v(&r2);
        let p = softmax(&matvec(&self.w, v, h1, &f1));
        let f1p = tanh_v(&r1_prev);
        let f2p = tanh_v(&r2_prev);
        for i in 0..v {
            let ex = (i == symbol) as u8 as f32 - p[i];
            for j in 0..h1 {
                self.w[i * h1 + j] += self.cfg.lr_w * ex * f1[j];
            }
        }
        for j in 0..h1 {
            let e1 = self.cfg.pi1 * (r1[j] - prior1[j]);
            for k in 0..h1 {
                self.a1[j * h1 + k] += self.cfg.lr_a * e1 * f1p[k];
            }
            for k in 0..h2 {
                self.b[j * h2 + k] += self.cfg.lr_top * e1 * f2[k];
            }
        }
        for j in 0..h2 {
            let e2 = self.cfg.pi2 * (r2[j] - prior2[j]);
            for k in 0..h2 {
                self.a2[j * h2 + k] += self.cfg.lr_top * e2 * f2p[k];
            }
        }

        // Predict the next symbol.
        let (next_prior, _) = self.priors(&r1, &r2, &r2);
        self.pred = softmax(&matvec(&self.w, v, h1, &tanh_v(&next_prior)));
        self.r1 = r1;
        self.r2 = r2;
        Step {
            p_actual,
            surprise: -p_actual.ln(),
            next: self.pred.clone(),
        }
    }
}
