//! `cargo run --release -p foretone-core --example throughput [-- --min N]`
//! Measures `observe` calls per second for both models at 32 hidden units.
//! With `--min N` exits non-zero if the single-layer model is below N observes/s
//! (used by check.sh as the performance budget).
use foretone_core::{Config, HierConfig, HierModel, Model};
use std::time::Instant;

const VOCAB: usize = 12;
const HIDDEN: usize = 32;
const N: usize = 200_000;

fn pattern(i: usize) -> usize {
    // A deterministic cycle with a little variety, so inference is never trivially idle.
    [0, 4, 7, 4, 0, 5, 9, 5, 2, 7, 11, 7][(i * 7 + i / 12) % 12] % VOCAB
}

fn rate(mut f: impl FnMut(usize) -> f32) -> f64 {
    for i in 0..5_000 {
        f(pattern(i)); // warm-up
    }
    let mut sink = 0.0;
    let t = Instant::now();
    for i in 0..N {
        sink += f(pattern(i));
    }
    let secs = t.elapsed().as_secs_f64();
    assert!(sink.is_finite());
    N as f64 / secs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let min = args
        .iter()
        .position(|a| a == "--min")
        .map(|i| args[i + 1].parse::<f64>().expect("--min needs a number"));

    let mut m = Model::new(Config::new(VOCAB, HIDDEN));
    let single = rate(|s| m.observe(s).surprise);
    let mut h = HierModel::new(HierConfig::new(VOCAB, HIDDEN, HIDDEN / 2));
    let hier = rate(|s| h.observe(s).surprise);

    println!("model        hidden  observe/s");
    println!("single       {HIDDEN:>6}  {single:>9.0}");
    println!("hierarchical {HIDDEN:>6}  {hier:>9.0}");
    if let Some(min) = min {
        if single < min {
            eprintln!("FAIL: single-layer {single:.0}/s is below budget {min:.0}/s");
            std::process::exit(1);
        }
    }
}
