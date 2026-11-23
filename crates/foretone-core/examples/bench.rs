//! `cargo run --release -p foretone-core --example bench`
//! Mean surprise (nats) on the 2nd playthrough of each song, fresh model per song.
use foretone_core::{Config, HierConfig, HierModel, Model};

pub const SONGS: [(&str, &[usize]); 5] = [
    (
        "scale up/down",
        &[0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0],
    ),
    (
        "arpeggio",
        &[0, 2, 4, 7, 4, 2, 0, 2, 4, 7, 4, 2, 0, 0, 0, 0],
    ),
    ("twinkle", &[0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0, 0, 0]),
    ("ode-ish", &[2, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 2, 1, 1, 1]),
    (
        "call/response",
        &[0, 2, 4, 2, 0, 2, 4, 2, 5, 4, 3, 2, 1, 0, 0, 0],
    ),
];

fn day1(song: &[usize]) -> f32 {
    // Best-tuned Day-1 rates from a small sweep (fair baseline; defaults are slower).
    let mut cfg = Config::new(8, 16);
    cfg.lr_w = 1.0;
    cfg.lr_a = 0.2;
    let mut m = Model::new(cfg);
    let mut second = 0.0;
    for rep in 0..2 {
        for &s in song {
            let st = m.observe(s);
            if rep == 1 {
                second += st.surprise;
            }
        }
    }
    second / song.len() as f32
}

fn hier(song: &[usize]) -> f32 {
    let mut m = HierModel::new(HierConfig::new(8, 16, 6));
    let mut second = 0.0;
    for rep in 0..2 {
        for &s in song {
            let st = m.observe(s);
            if rep == 1 {
                second += st.surprise;
            }
        }
    }
    second / song.len() as f32
}

fn main() {
    println!("{:<15} {:>8} {:>8}", "song", "day1", "hier");
    let (mut a, mut b) = (0.0, 0.0);
    for (name, song) in SONGS {
        let (x, y) = (day1(song), hier(song));
        println!("{name:<15} {x:>8.3} {y:>8.3}");
        a += x;
        b += y;
    }
    let n = SONGS.len() as f32;
    println!("{:<15} {:>8.3} {:>8.3}", "mean", a / n, b / n);
}
