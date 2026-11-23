use foretone_core::{argmax, HierConfig, HierModel};

#[test]
fn analytic_gradient_matches_finite_differences() {
    let mut cfg = HierConfig::new(4, 6, 3);
    cfg.seed = 7;
    let m = HierModel::new(cfg);
    let r1p = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2];
    let r2p = [0.1, -0.3, 0.2];
    let r1 = [0.2, 0.4, -0.1, 0.3, 0.0, -0.5];
    let r2 = [-0.2, 0.3, 0.4];
    let (_, g1, g2) = m.energy_and_grad(2, &r1p, &r2p, &r1, &r2);
    let eps = 1e-2;
    let energy = |a: &[f32], b: &[f32]| m.energy_and_grad(2, &r1p, &r2p, a, b).0;
    for j in 0..r1.len() {
        let (mut hi, mut lo) = (r1, r1);
        hi[j] += eps;
        lo[j] -= eps;
        let num = (energy(&hi, &r2) - energy(&lo, &r2)) / (2.0 * eps);
        assert!(
            (num - g1[j]).abs() < 2e-3,
            "r1[{j}]: num {num} vs {}",
            g1[j]
        );
    }
    for k in 0..r2.len() {
        let (mut hi, mut lo) = (r2, r2);
        hi[k] += eps;
        lo[k] -= eps;
        let num = (energy(&r1, &hi) - energy(&r1, &lo)) / (2.0 * eps);
        assert!(
            (num - g2[k]).abs() < 2e-3,
            "r2[{k}]: num {num} vs {}",
            g2[k]
        );
    }
}

#[test]
fn hierarchy_learns_a_cycle_and_is_deterministic() {
    let seq: Vec<usize> = [0, 1, 2, 3, 2, 1, 0, 0]
        .iter()
        .cycle()
        .take(32)
        .cloned()
        .collect();
    let run = || {
        let mut m = HierModel::new(HierConfig::new(4, 16, 6));
        let mut hits = 0;
        let (mut first, mut last) = (0.0, 0.0);
        for e in 0..80 {
            let mut s = 0.0;
            for (i, &x) in seq.iter().enumerate() {
                let g = argmax(m.predict());
                s += m.observe(x).surprise;
                hits = if i == 0 {
                    0
                } else {
                    hits + (e == 79 && g == x) as usize
                };
            }
            if e == 0 {
                first = s;
            }
            last = s;
        }
        (hits, first, last)
    };
    let (hits, first, last) = run();
    assert!(hits >= 29, "hits {hits}/31");
    assert!(last < first * 0.3, "{first} -> {last}");
    let again = run();
    assert_eq!((hits, first, last), again);
}
