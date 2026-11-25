//! Exercises the exported C ABI natively, the same way the JS wrapper drives it.
use foretone_core::{argmax, Config, Model};
use foretone_wasm::*;

fn tune() -> Vec<usize> {
    [0, 1, 2, 3, 2, 1, 0, 0]
        .iter()
        .cycle()
        .take(160)
        .copied()
        .collect()
}

#[test]
fn ffi_matches_native_model_bit_for_bit() {
    let mut native = Model::new(Config::new(4, 16));
    unsafe {
        let h = ft_new(4, 16, 42, 0);
        assert!(!h.is_null());
        for s in tune() {
            let a = native.observe(s).surprise;
            let b = ft_observe(h, s as u32);
            assert_eq!(a.to_bits(), b.to_bits());
            let p = std::slice::from_raw_parts(ft_predict(h), ft_vocab(h) as usize);
            assert_eq!(p, native.predict());
        }
        ft_free(h);
    }
}

#[test]
fn seed_halves_are_combined() {
    let mut cfg = Config::new(4, 8);
    cfg.seed = (7u64 << 32) | 9;
    let mut native = Model::new(cfg);
    unsafe {
        let h = ft_new(4, 8, 9, 7);
        assert_eq!(
            native.observe(1).surprise.to_bits(),
            ft_observe(h, 1).to_bits()
        );
        ft_free(h);
    }
}

#[test]
fn invalid_inputs_are_errors_not_panics() {
    unsafe {
        assert!(ft_new(1, 8, 0, 0).is_null());
        assert!(ft_new(4, 0, 0, 0).is_null());
        assert!(ft_new(4, 1 << 20, 0, 0).is_null());
        let h = ft_new(4, 8, 1, 0);
        assert_eq!(ft_observe(h, 4), -1.0);
        assert_eq!(ft_observe(std::ptr::null_mut(), 0), -1.0);
        assert_eq!(ft_vocab(std::ptr::null()), 0);
        assert!(ft_predict(std::ptr::null()).is_null());
        assert_eq!(ft_export(std::ptr::null_mut()), -1);
        assert_eq!(ft_import(h, std::ptr::null(), 0), -1);
        let junk = [1u8, 2, 3];
        assert_eq!(ft_import(h, junk.as_ptr(), junk.len()), -1);
        ft_free(h);
        ft_free(std::ptr::null_mut());
    }
}

#[test]
fn export_import_continues_identically_and_failed_import_keeps_model() {
    unsafe {
        let a = ft_new(4, 16, 42, 0);
        for s in tune() {
            ft_observe(a, s as u32);
        }
        let n = ft_export(a) as usize;
        let bytes = std::slice::from_raw_parts(ft_export_ptr(a), n).to_vec();

        let b = ft_new(4, 16, 1, 0);
        assert_eq!(ft_import(b, bytes.as_ptr(), bytes.len()), 0);
        // truncated import fails and leaves b usable and unchanged
        assert_eq!(ft_import(b, bytes.as_ptr(), bytes.len() - 1), -1);
        for s in tune().into_iter().take(20) {
            assert_eq!(
                ft_observe(a, s as u32).to_bits(),
                ft_observe(b, s as u32).to_bits()
            );
        }
        ft_free(a);
        ft_free(b);
    }
}

#[test]
fn reset_clears_state_but_keeps_learning() {
    let mut s = Session::new(4, 16, 42).unwrap();
    for x in tune().into_iter().cycle().take(1200) {
        s.observe(x);
    }
    s.reset();
    assert!(s.predict().iter().all(|&p| (p - 0.25).abs() < 1e-6));
    for x in [0, 1, 2] {
        s.observe(x);
    }
    assert_eq!(
        argmax(s.predict()),
        3,
        "weights survive reset: C D E is followed by G"
    );
}

#[test]
fn alloc_roundtrip() {
    unsafe {
        let p = ft_alloc(0);
        assert!(!p.is_null());
        ft_dealloc(p, 0);
        let p = ft_alloc(64);
        *p.add(63) = 7;
        ft_dealloc(p, 64);
    }
}
