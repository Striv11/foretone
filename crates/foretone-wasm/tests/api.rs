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

#[test]
fn latent_and_dream_through_the_abi() {
    unsafe {
        let h = ft_new(4, 16, 42, 0);
        assert_eq!(ft_hidden(h), 16);
        let zeros = std::slice::from_raw_parts(ft_latent(h), 16).to_vec();
        assert!(zeros.iter().all(|&v| v == 0.0), "fresh state is silent");
        for s in tune().into_iter().cycle().take(800) {
            ft_observe(h, s as u32);
        }
        let r = std::slice::from_raw_parts(ft_latent(h), 16);
        assert!(r.iter().any(|&v| v != 0.0) && r.iter().all(|v| v.is_finite()));

        // 800 notes = 100 whole cycles, so the dream should continue the tune from C.
        let before_p = std::slice::from_raw_parts(ft_predict(h), 4).to_vec();
        let n = ft_export(h);
        let before_bytes = std::slice::from_raw_parts(ft_export_ptr(h), n as usize).to_vec();
        let mut out = [0u32; 8];
        assert_eq!(ft_dream(h, out.as_mut_ptr(), 8), 8);
        assert_eq!(out, [0, 1, 2, 3, 2, 1, 0, 0]);

        // Dreaming is read-only.
        assert_eq!(std::slice::from_raw_parts(ft_predict(h), 4), &before_p[..]);
        let n2 = ft_export(h);
        assert_eq!(
            std::slice::from_raw_parts(ft_export_ptr(h), n2 as usize),
            &before_bytes[..]
        );

        // Bad arguments are errors, not panics.
        assert_eq!(ft_dream(h, out.as_mut_ptr(), 5000), -1);
        assert_eq!(ft_dream(h, std::ptr::null_mut(), 4), -1);
        assert_eq!(ft_dream(std::ptr::null(), out.as_mut_ptr(), 4), -1);
        assert_eq!(ft_hidden(std::ptr::null()), 0);
        assert!(ft_latent(std::ptr::null()).is_null());
        ft_free(h);
    }
}
