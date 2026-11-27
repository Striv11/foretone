//! WebAssembly bindings for `foretone-core`.
//!
//! Design: a plain C ABI (`extern "C"`) instead of wasm-bindgen. The surface is tiny
//! (numbers and byte buffers), so the module stays dependency-free, needs no extra
//! build tool, and loads with the standard `WebAssembly.instantiate` in browsers and
//! Node. `web/foretone.mjs` wraps it in an ergonomic class.
//!
//! Memory protocol: the host calls `ft_alloc(n)` to get a buffer inside wasm memory,
//! writes into it, and passes it back; every buffer is released with `ft_dealloc`.
//! Models are opaque handles released with `ft_free`. Fallible calls return a negative
//! value instead of panicking across the boundary.
//!
//! The safe [`Session`] type carries the logic so it can be tested natively; the
//! `ft_*` functions only translate pointers.

use foretone_core::{Config, Model};

/// A model plus the scratch buffer used for exports.
pub struct Session {
    model: Model,
    export: Vec<u8>,
}

impl Session {
    /// Returns `None` for sizes the model rejects.
    pub fn new(vocab: usize, hidden: usize, seed: u64) -> Option<Self> {
        if vocab < 2 || hidden < 1 || vocab > 4096 || hidden > 4096 {
            return None;
        }
        let mut cfg = Config::new(vocab, hidden);
        cfg.seed = seed;
        Some(Session {
            model: Model::new(cfg),
            export: Vec::new(),
        })
    }

    pub fn vocab(&self) -> usize {
        self.model.config().vocab
    }

    /// Observe a symbol; returns its surprise in nats, or `None` if out of range.
    pub fn observe(&mut self, symbol: usize) -> Option<f32> {
        (symbol < self.vocab()).then(|| self.model.observe(symbol).surprise)
    }

    pub fn predict(&self) -> &[f32] {
        self.model.predict()
    }

    pub fn latent(&self) -> &[f32] {
        self.model.latent()
    }

    /// Free-running generation of `n` symbols (does not change the model).
    pub fn dream(&self, n: usize) -> Vec<usize> {
        self.model.dream(n)
    }

    pub fn reset(&mut self) {
        self.model.reset_state();
    }

    /// Serialize into the internal buffer and return it.
    pub fn export(&mut self) -> &[u8] {
        self.export = self.model.to_bytes();
        &self.export
    }

    /// Replace the model with a serialized one. On error the model is untouched.
    pub fn import(&mut self, bytes: &[u8]) -> bool {
        match Model::from_bytes(bytes) {
            Ok(m) => {
                self.model = m;
                true
            }
            Err(_) => false,
        }
    }
}

/// Allocate `n` bytes in wasm memory for the host to fill.
#[no_mangle]
pub extern "C" fn ft_alloc(n: usize) -> *mut u8 {
    let mut v = vec![0u8; n.max(1)].into_boxed_slice();
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from `ft_alloc(n)` with the same `n`, and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn ft_dealloc(ptr: *mut u8, n: usize) {
    if !ptr.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            ptr,
            n.max(1),
        )));
    }
}

/// Create a model. Returns null if the sizes are invalid. The seed is passed as two
/// 32-bit halves because 64-bit integers need BigInt on the JS side.
#[no_mangle]
pub extern "C" fn ft_new(vocab: u32, hidden: u32, seed_lo: u32, seed_hi: u32) -> *mut Session {
    let seed = (seed_hi as u64) << 32 | seed_lo as u64;
    match Session::new(vocab as usize, hidden as usize, seed) {
        Some(s) => Box::into_raw(Box::new(s)),
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `h` must be a live handle from `ft_new` (or null); it is invalid afterwards.
#[no_mangle]
pub unsafe extern "C" fn ft_free(h: *mut Session) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Observe a symbol. Returns the surprise in nats, or -1 if the handle is null or the
/// symbol is out of range.
///
/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_observe(h: *mut Session, symbol: u32) -> f32 {
    h.as_mut()
        .and_then(|s| s.observe(symbol as usize))
        .unwrap_or(-1.0)
}

/// Number of symbols in the model's vocabulary (0 for a null handle).
///
/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_vocab(h: *const Session) -> u32 {
    h.as_ref().map_or(0, |s| s.vocab() as u32)
}

/// Pointer to `ft_vocab` f32 probabilities for the next symbol. Valid until the next
/// call that mutates the model.
///
/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_predict(h: *const Session) -> *const f32 {
    h.as_ref()
        .map_or(std::ptr::null(), |s| s.predict().as_ptr())
}

/// Pointer to `ft_hidden` f32 latent activations. Valid until the next mutating call.
///
/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_latent(h: *const Session) -> *const f32 {
    h.as_ref().map_or(std::ptr::null(), |s| s.latent().as_ptr())
}

/// Number of latent units (0 for a null handle).
///
/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_hidden(h: *const Session) -> u32 {
    h.as_ref().map_or(0, |s| s.latent().len() as u32)
}

/// Free-run `n` (<= 4096) symbols into `out[..n]` as u32 indices without changing the
/// model. Returns the count written, or -1 on bad arguments.
///
/// # Safety
/// `h` must be null or a live handle; `out` must be valid for `n` u32 writes.
#[no_mangle]
pub unsafe extern "C" fn ft_dream(h: *const Session, out: *mut u32, n: u32) -> i32 {
    let (Some(s), false) = (h.as_ref(), out.is_null()) else {
        return -1;
    };
    if n > 4096 {
        return -1;
    }
    let seq = s.dream(n as usize);
    for (i, v) in seq.iter().enumerate() {
        *out.add(i) = *v as u32;
    }
    seq.len() as i32
}

/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_reset(h: *mut Session) {
    if let Some(s) = h.as_mut() {
        s.reset();
    }
}

/// Serialize the model. Returns the byte length (or -1); read it via `ft_export_ptr`.
///
/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_export(h: *mut Session) -> i32 {
    h.as_mut().map_or(-1, |s| s.export().len() as i32)
}

/// # Safety
/// `h` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ft_export_ptr(h: *const Session) -> *const u8 {
    h.as_ref().map_or(std::ptr::null(), |s| s.export.as_ptr())
}

/// Load a serialized model from `ptr[..len]`. Returns 0 on success, -1 on error.
///
/// # Safety
/// `h` must be null or a live handle; `ptr` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ft_import(h: *mut Session, ptr: *const u8, len: usize) -> i32 {
    let (Some(s), false) = (h.as_mut(), ptr.is_null()) else {
        return -1;
    };
    if s.import(std::slice::from_raw_parts(ptr, len)) {
        0
    } else {
        -1
    }
}
