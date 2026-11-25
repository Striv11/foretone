//! Prints the native output the Node smoke test must reproduce.
//! `cargo run --release -p foretone-wasm --example reference`
use foretone_core::{Config, Model};

fn main() {
    let tune = [0usize, 1, 2, 3, 2, 1, 0, 0];
    let mut m = Model::new(Config::new(4, 16));
    for i in 0..160 {
        let s = m.observe(tune[i % tune.len()]);
        let p: Vec<String> = m.predict().iter().map(|v| format!("{v:.6}")).collect();
        println!("{:.6} {}", s.surprise, p.join(" "));
    }
}
