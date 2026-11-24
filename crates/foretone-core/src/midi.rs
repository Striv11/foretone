//! Minimal Standard MIDI File (`.mid`) reader, written in-crate.
//!
//! Extracts the note-on events (velocity > 0) of all tracks, merged in time
//! order, as a melody. Chords come out in track order at the same tick.
//! Handles format 0/1, running status, meta and SysEx events.

use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum MidiError {
    NotMidi,
    Truncated,
    BadData(&'static str),
    NoNotes,
}

impl fmt::Display for MidiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MidiError::NotMidi => write!(f, "missing MThd header"),
            MidiError::Truncated => write!(f, "file ends unexpectedly"),
            MidiError::BadData(m) => write!(f, "invalid MIDI data: {m}"),
            MidiError::NoNotes => write!(f, "no notes in file"),
        }
    }
}

impl std::error::Error for MidiError {}

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cur<'a> {
    fn u8(&mut self) -> Result<u8, MidiError> {
        let v = *self.b.get(self.i).ok_or(MidiError::Truncated)?;
        self.i += 1;
        Ok(v)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], MidiError> {
        let end = self.i.checked_add(n).ok_or(MidiError::Truncated)?;
        let s = self.b.get(self.i..end).ok_or(MidiError::Truncated)?;
        self.i = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, MidiError> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn vlq(&mut self) -> Result<u32, MidiError> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            let b = self.u8()?;
            v = (v << 7) | (b & 0x7f) as u32;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(MidiError::BadData("variable-length quantity too long"))
    }
}

/// A note-on event: absolute tick and pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteOn {
    pub tick: u64,
    pub pitch: u8,
}

fn parse_track(data: &[u8]) -> Result<Vec<NoteOn>, MidiError> {
    let mut c = Cur { b: data, i: 0 };
    let mut out = Vec::new();
    let mut tick: u64 = 0;
    let mut status: Option<u8> = None;
    while c.i < data.len() {
        tick += c.vlq()? as u64;
        let first = c.u8()?;
        let st = if first & 0x80 != 0 {
            first
        } else {
            c.i -= 1;
            status.ok_or(MidiError::BadData("data byte without running status"))?
        };
        match st {
            0xFF => {
                let ty = c.u8()?;
                let len = c.vlq()? as usize;
                c.take(len)?;
                if ty == 0x2F {
                    break;
                }
            }
            0xF0 | 0xF7 => {
                let len = c.vlq()? as usize;
                c.take(len)?;
            }
            0x80..=0xEF => {
                status = Some(st);
                let n = if matches!(st & 0xF0, 0xC0 | 0xD0) {
                    1
                } else {
                    2
                };
                let d = c.take(n)?;
                if d.iter().any(|b| b & 0x80 != 0) {
                    return Err(MidiError::BadData("data byte has high bit set"));
                }
                if st & 0xF0 == 0x90 && d[1] > 0 {
                    out.push(NoteOn { tick, pitch: d[0] });
                }
            }
            _ => return Err(MidiError::BadData("unsupported status byte")),
        }
    }
    Ok(out)
}

/// Parse a `.mid` file into its note-on events, sorted by time (stable).
pub fn parse_midi(bytes: &[u8]) -> Result<Vec<NoteOn>, MidiError> {
    let mut c = Cur { b: bytes, i: 0 };
    if c.take(4).map_err(|_| MidiError::NotMidi)? != b"MThd" {
        return Err(MidiError::NotMidi);
    }
    let hlen = c.u32()? as usize;
    if hlen < 6 {
        return Err(MidiError::BadData("header too short"));
    }
    let h = c.take(hlen)?;
    let ntracks = u16::from_be_bytes([h[2], h[3]]);
    let mut notes = Vec::new();
    for _ in 0..ntracks {
        if c.take(4)? != b"MTrk" {
            return Err(MidiError::BadData("expected MTrk chunk"));
        }
        let len = c.u32()? as usize;
        notes.extend(parse_track(c.take(len)?)?);
    }
    if notes.is_empty() {
        return Err(MidiError::NoNotes);
    }
    notes.sort_by_key(|n| n.tick);
    Ok(notes)
}

/// Convenience: just the pitches, in time order.
pub fn midi_melody(bytes: &[u8]) -> Result<Vec<u8>, MidiError> {
    Ok(parse_midi(bytes)?.into_iter().map(|n| n.pitch).collect())
}
