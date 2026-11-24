//! Compact little-endian binary codec used for model (de)serialization.
//! Floats are stored as raw IEEE-754 bits, so round trips are bit-exact.

use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    BadMagic,
    WrongKind,
    Truncated,
    TrailingBytes,
    Invalid(&'static str),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::BadMagic => write!(f, "not a foretone model"),
            DecodeError::WrongKind => write!(f, "model kind mismatch"),
            DecodeError::Truncated => write!(f, "data ends unexpectedly"),
            DecodeError::TrailingBytes => write!(f, "unexpected trailing bytes"),
            DecodeError::Invalid(m) => write!(f, "invalid model data: {m}"),
        }
    }
}

impl std::error::Error for DecodeError {}

pub(crate) const MAGIC: &[u8; 4] = b"FTN1";
pub(crate) const KIND_MODEL: u8 = 1;
pub(crate) const KIND_HIER: u8 = 2;

pub(crate) struct Writer(pub Vec<u8>);

impl Writer {
    pub fn new(kind: u8) -> Self {
        let mut v = MAGIC.to_vec();
        v.push(kind);
        Writer(v)
    }
    pub fn u32(&mut self, v: usize) {
        self.0.extend_from_slice(&(v as u32).to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    pub fn f32s(&mut self, vs: &[f32]) {
        vs.iter().for_each(|&v| self.f32(v));
    }
}

pub(crate) struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8], kind: u8) -> Result<Self, DecodeError> {
        if b.len() < 5 || &b[..4] != MAGIC {
            return Err(DecodeError::BadMagic);
        }
        if b[4] != kind {
            return Err(DecodeError::WrongKind);
        }
        Ok(Reader { b, i: 5 })
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.i.checked_add(n).ok_or(DecodeError::Truncated)?;
        let s = self.b.get(self.i..end).ok_or(DecodeError::Truncated)?;
        self.i = end;
        Ok(s)
    }
    pub fn u32(&mut self) -> Result<usize, DecodeError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    }
    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f32, DecodeError> {
        let s = self.take(4)?;
        Ok(f32::from_bits(u32::from_le_bytes([s[0], s[1], s[2], s[3]])))
    }
    pub fn f32s(&mut self, n: usize) -> Result<Vec<f32>, DecodeError> {
        // Check the length first so a corrupt header can't trigger a huge allocation.
        if n.checked_mul(4).is_none_or(|b| b > self.b.len() - self.i) {
            return Err(DecodeError::Truncated);
        }
        (0..n).map(|_| self.f32()).collect()
    }
    pub fn finish(self) -> Result<(), DecodeError> {
        if self.i == self.b.len() {
            Ok(())
        } else {
            Err(DecodeError::TrailingBytes)
        }
    }
}
