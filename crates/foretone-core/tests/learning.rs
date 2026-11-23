use foretone_core::{argmax, softmax, Config, Model};

/// Feed `seq` repeatedly; return next-symbol accuracy over the final `eval` steps
/// and the mean surprise of the first vs last epoch.
fn run(model: &mut Model, seq: &[usize], epochs: usize) -> (f32, f32, f32) {
    let mut first = 0.0;
    let mut last = 0.0;
    let mut hits = 0;
    for e in 0..epochs {
        let mut s = 0.0;
        for (i, &sym) in seq.iter().enumerate() {
            let guess = argmax(model.predict());
            let st = model.observe(sym);
            s += st.surprise;
            if e == epochs - 1 && i > 0 && guess == sym {
                hits += 1;
            }
        }
        if e == 0 {
            first = s / seq.len() as f32;
        }
        last = s / seq.len() as f32;
    }
    (hits as f32 / (seq.len() - 1) as f32, first, last)
}

#[test]
fn softmax_sums_to_one() {
    let p = softmax(&[1.0, 2.0, 3.0, -50.0]);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn learns_a_repeating_melody_online() {
    // C D E C D E ... : each symbol fully determines the next.
    let seq: Vec<usize> = (0..30).map(|i| i % 3).collect();
    let mut m = Model::new(Config::new(3, 12));
    let (acc, first, last) = run(&mut m, &seq, 60);
    assert!(acc > 0.95, "accuracy {acc}");
    assert!(last < first * 0.3, "surprise {first} -> {last}");
}

#[test]
fn learns_a_longer_cycle() {
    let cycle = [0usize, 1, 2, 3, 2, 1, 4, 0];
    let seq: Vec<usize> = (0..64).map(|i| cycle[i % cycle.len()]).collect();
    let mut m = Model::new(Config::new(5, 24));
    let (acc, _, last) = run(&mut m, &seq, 150);
    assert!(acc > 0.9, "accuracy {acc}");
    assert!(last < 0.5, "surprise {last}");
}

#[test]
fn surprise_spikes_on_a_violation() {
    let seq: Vec<usize> = (0..30).map(|i| i % 3).collect();
    let mut m = Model::new(Config::new(3, 12));
    run(&mut m, &seq, 60);
    m.reset_state();
    m.observe(0);
    m.observe(1);
    let expected = m.observe(2).surprise;
    m.observe(0);
    let wrong = m.observe(0).surprise; // should have been 1
    assert!(wrong > expected * 5.0, "{wrong} vs {expected}");
}

#[test]
fn deterministic_for_a_seed() {
    let mut a = Model::new(Config::new(4, 8));
    let mut b = Model::new(Config::new(4, 8));
    for i in 0..50 {
        assert_eq!(a.observe(i % 4).next, b.observe(i % 4).next);
    }
}
