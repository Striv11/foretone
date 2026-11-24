//! Encoders: note-name text, plain characters, and a symbol vocabulary that
//! maps arbitrary tokens onto the dense `0..vocab` indices the models use.

use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    BadNote(String),
    Empty,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::BadNote(t) => write!(f, "not a note name: {t:?}"),
            ParseError::Empty => write!(f, "no notes found"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse one note name such as `C4`, `f#3`, `Bb5` or `C-1` into a MIDI number
/// (middle C = `C4` = 60). The octave is optional and defaults to 4.
pub fn parse_note(tok: &str) -> Result<u8, ParseError> {
    let bad = || ParseError::BadNote(tok.to_string());
    let mut chars = tok.chars().peekable();
    let base: i32 = match chars.next().map(|c| c.to_ascii_uppercase()) {
        Some('C') => 0,
        Some('D') => 2,
        Some('E') => 4,
        Some('F') => 5,
        Some('G') => 7,
        Some('A') => 9,
        Some('B') => 11,
        _ => return Err(bad()),
    };
    let mut acc = 0;
    while let Some(&c) = chars.peek() {
        match c {
            '#' => acc += 1,
            'b' => acc -= 1,
            _ => break,
        }
        chars.next();
    }
    let rest: String = chars.collect();
    let octave: i32 = if rest.is_empty() {
        4
    } else {
        rest.parse().map_err(|_| bad())?
    };
    let midi = (octave + 1) * 12 + base + acc;
    u8::try_from(midi)
        .ok()
        .filter(|&m| m <= 127)
        .ok_or_else(bad)
}

/// Parse whitespace/comma separated note names into MIDI numbers.
pub fn parse_melody(text: &str) -> Result<Vec<u8>, ParseError> {
    let notes: Result<Vec<u8>, _> = text
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(parse_note)
        .collect();
    let notes = notes?;
    if notes.is_empty() {
        return Err(ParseError::Empty);
    }
    Ok(notes)
}

/// Name of a MIDI number using sharps, e.g. `60 -> "C4"`.
pub fn note_name(midi: u8) -> String {
    const N: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!("{}{}", N[(midi % 12) as usize], midi as i32 / 12 - 1)
}

/// Bidirectional map between tokens and dense indices (sorted, so the
/// encoding is deterministic regardless of input order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocab<T: Ord + Clone> {
    items: Vec<T>,
}

impl<T: Ord + Clone> Vocab<T> {
    pub fn from_tokens(tokens: &[T]) -> Self {
        let mut items = tokens.to_vec();
        items.sort();
        items.dedup();
        Vocab { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn encode(&self, t: &T) -> Option<usize> {
        self.items.binary_search(t).ok()
    }

    pub fn decode(&self, i: usize) -> Option<&T> {
        self.items.get(i)
    }

    /// Encode a whole sequence; `None` if any token is outside the vocabulary.
    pub fn encode_all(&self, tokens: &[T]) -> Option<Vec<usize>> {
        tokens.iter().map(|t| self.encode(t)).collect()
    }
}

/// Text mode: characters as symbols.
pub fn char_vocab(text: &str) -> Vocab<char> {
    Vocab::from_tokens(&text.chars().collect::<Vec<_>>())
}
