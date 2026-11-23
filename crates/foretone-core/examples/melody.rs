//! `cargo run --release -p foretone-core --example melody`
use foretone_core::{argmax, Config, Model};

fn main() {
    let names = ["C", "D", "E", "G"];
    let tune = [0usize, 1, 2, 3, 2, 1, 0, 0]; // C D E G E D C C
    let mut m = Model::new(Config::new(4, 16));
    for epoch in 1..=60 {
        let mut total = 0.0;
        let mut hits = 0;
        for (i, &s) in tune.iter().cycle().take(32).enumerate() {
            let guess = argmax(m.predict());
            total += m.observe(s).surprise;
            hits += (i > 0 && guess == s) as usize;
        }
        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "epoch {epoch:>2}: mean surprise {:.3} nats, next-note hits {hits}/31",
                total / 32.0
            );
        }
    }
    let seq: Vec<&str> = (0..8)
        .map(|_| {
            let n = argmax(m.predict());
            m.observe(n);
            names[n]
        })
        .collect();
    println!("free-running dream: {}", seq.join(" "));
}
