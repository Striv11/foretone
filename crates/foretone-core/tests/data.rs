use foretone_core::*;

#[test]
fn note_names_parse_to_midi() {
    assert_eq!(parse_note("C4"), Ok(60));
    assert_eq!(parse_note("c"), Ok(60));
    assert_eq!(parse_note("A4"), Ok(69));
    assert_eq!(parse_note("F#3"), Ok(54));
    assert_eq!(parse_note("Bb3"), Ok(58));
    assert_eq!(parse_note("C-1"), Ok(0));
    assert_eq!(parse_note("G9"), Ok(127));
    assert!(parse_note("G#9").is_err());
    assert!(parse_note("H4").is_err());
    assert!(parse_note("C4x").is_err());
    assert!(parse_note("").is_err());
    assert_eq!(
        parse_melody("C4 D4, E4\n G4").unwrap(),
        vec![60, 62, 64, 67]
    );
    assert_eq!(parse_melody("  "), Err(ParseError::Empty));
    assert!(matches!(parse_melody("C4 zz"), Err(ParseError::BadNote(t)) if t == "zz"));
}

#[test]
fn note_name_round_trips() {
    for m in 0..=127u8 {
        assert_eq!(parse_note(&note_name(m)), Ok(m));
    }
}

#[test]
fn vocab_is_sorted_and_invertible() {
    let v = Vocab::from_tokens(&[67u8, 60, 64, 60]);
    assert_eq!(v.len(), 3);
    assert_eq!(v.encode(&60), Some(0));
    assert_eq!(v.encode(&67), Some(2));
    assert_eq!(v.encode(&61), None);
    assert_eq!(v.decode(1), Some(&64));
    assert_eq!(v.encode_all(&[64, 60]), Some(vec![1, 0]));
    assert_eq!(v.encode_all(&[64, 1]), None);
}

#[test]
fn text_mode_learns_a_repeating_phrase() {
    let text = "abcabdabcabd".repeat(20);
    let vocab = char_vocab(&text);
    assert_eq!(vocab.len(), 4);
    let seq: Vec<usize> = text.chars().map(|c| vocab.encode(&c).unwrap()).collect();
    let mut m = Model::new(Config::new(vocab.len(), 24));
    let (mut hits, mut n) = (0, 0);
    for (i, &s) in seq.iter().enumerate() {
        let guess = argmax(m.predict());
        m.observe(s);
        if i >= seq.len() / 2 {
            n += 1;
            hits += (guess == s) as usize;
        }
    }
    assert!(hits as f32 / n as f32 > 0.9, "{hits}/{n}");
}

// ---- MIDI -----------------------------------------------------------------

fn vlq(mut v: u32) -> Vec<u8> {
    let mut out = vec![(v & 0x7f) as u8];
    v >>= 7;
    while v > 0 {
        out.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    out.reverse();
    out
}

fn track(events: &[u8]) -> Vec<u8> {
    let mut t = b"MTrk".to_vec();
    let mut body = events.to_vec();
    body.extend([0x00, 0xFF, 0x2F, 0x00]);
    t.extend((body.len() as u32).to_be_bytes());
    t.extend(body);
    t
}

fn file(format: u16, tracks: &[Vec<u8>]) -> Vec<u8> {
    let mut f = b"MThd".to_vec();
    f.extend(6u32.to_be_bytes());
    f.extend(format.to_be_bytes());
    f.extend((tracks.len() as u16).to_be_bytes());
    f.extend(480u16.to_be_bytes());
    tracks.iter().for_each(|t| f.extend(t));
    f
}

#[test]
fn midi_extracts_notes_with_running_status_and_meta() {
    let mut e = vec![0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]; // tempo meta
    e.extend([0x00, 0xC0, 0x05]); // program change (1 data byte)
    e.extend([0x00, 0x90, 60, 100]); // note on C4
    e.extend(vlq(480));
    e.extend([60, 0]); // running status, velocity 0 = note off
    e.extend([0x00, 62, 90]); // running status note on D4
    e.extend(vlq(300));
    e.extend([0x80, 62, 0]); // explicit note off
    e.extend([0x00, 0xF0, 0x03, 0x01, 0x02, 0xF7]); // sysex
    e.extend(vlq(1000));
    e.extend([0x90, 64, 80]); // E4 at a 2-byte vlq delta
    let notes = parse_midi(&file(0, &[track(&e)])).unwrap();
    assert_eq!(
        notes,
        vec![
            NoteOn { tick: 0, pitch: 60 },
            NoteOn {
                tick: 480,
                pitch: 62
            },
            NoteOn {
                tick: 1780,
                pitch: 64
            }
        ]
    );
}

#[test]
fn midi_merges_tracks_by_time() {
    let t1 = track(&[0x00, 0x90, 60, 90, 0x83, 0x60, 0x90, 64, 90]); // ticks 0, 480
    let t2 = track(&[0x78, 0x91, 62, 90]); // tick 120
    assert_eq!(midi_melody(&file(1, &[t1, t2])).unwrap(), vec![60, 62, 64]);
}

#[test]
fn midi_rejects_bad_input() {
    assert_eq!(parse_midi(b"RIFF"), Err(MidiError::NotMidi));
    assert_eq!(parse_midi(b""), Err(MidiError::NotMidi));
    let good = file(0, &[track(&[0x00, 0x90, 60, 90])]);
    for cut in 5..good.len() {
        assert!(parse_midi(&good[..cut]).is_err(), "cut {cut}");
    }
    assert_eq!(parse_midi(&file(0, &[track(&[])])), Err(MidiError::NoNotes));
    // data byte with no running status
    assert!(parse_midi(&file(0, &[track(&[0x00, 0x3C, 0x40])])).is_err());
    // overlong variable-length quantity
    assert!(parse_midi(&file(0, &[track(&[0xFF, 0xFF, 0xFF, 0xFF, 0x7F])])).is_err());
}

#[test]
fn midi_melody_feeds_the_model() {
    let mut ev = Vec::new();
    for p in [60u8, 62, 64, 62].iter().cycle().take(200) {
        ev.extend([0x00, 0x90, *p, 90, 0x60, 0x80, *p, 0]);
    }
    let melody = midi_melody(&file(0, &[track(&ev)])).unwrap();
    let vocab = Vocab::from_tokens(&melody);
    let seq = vocab.encode_all(&melody).unwrap();
    let mut m = Model::new(Config::new(vocab.len(), 16));
    let mut late = 0.0;
    for (i, &s) in seq.iter().enumerate() {
        let st = m.observe(s);
        if i >= 192 {
            late += st.surprise / 8.0;
        }
    }
    assert!(late < 0.3, "{late}");
}

// ---- serialization ---------------------------------------------------------

fn train(m: &mut Model, n: usize) {
    for &s in [0usize, 1, 2, 3, 2, 1].iter().cycle().take(n) {
        m.observe(s);
    }
}

#[test]
fn model_round_trips_bit_exactly_and_continues_identically() {
    let mut a = Model::new(Config::new(4, 12));
    train(&mut a, 50);
    let bytes = a.to_bytes();
    let mut b = Model::from_bytes(&bytes).unwrap();
    assert_eq!(b.to_bytes(), bytes);
    assert_eq!(a.predict(), b.predict());
    for &s in &[0usize, 1, 2, 3, 3, 0] {
        let (x, y) = (a.observe(s), b.observe(s));
        assert_eq!(x.surprise.to_bits(), y.surprise.to_bits());
        assert_eq!(x.next, y.next);
    }
    assert_eq!(a.to_bytes(), b.to_bytes());
}

#[test]
fn hier_round_trips_bit_exactly_and_continues_identically() {
    let mut a = HierModel::new(HierConfig::new(4, 12, 6));
    for &s in [0usize, 1, 2, 3, 2, 1].iter().cycle().take(50) {
        a.observe(s);
    }
    let bytes = a.to_bytes();
    let mut b = HierModel::from_bytes(&bytes).unwrap();
    assert_eq!(b.to_bytes(), bytes);
    for &s in &[0usize, 1, 2, 3, 3, 0] {
        assert_eq!(a.observe(s).next, b.observe(s).next);
    }
    assert_eq!(a.to_bytes(), b.to_bytes());
}

#[test]
fn nan_and_special_floats_survive() {
    let mut a = Model::new(Config::new(4, 4));
    train(&mut a, 5);
    let mut bytes = a.to_bytes();
    // overwrite the first weight with a NaN payload and check it is preserved
    let off = 5 + 12 + 16 + 8; // header, sizes, 4 f32 rates, seed
    bytes[off..off + 4].copy_from_slice(&0x7fc0_1234u32.to_le_bytes());
    let b = Model::from_bytes(&bytes).unwrap();
    assert_eq!(b.to_bytes(), bytes);
}

#[test]
fn decoding_rejects_corrupt_data() {
    let mut m = Model::new(Config::new(4, 8));
    train(&mut m, 10);
    let bytes = m.to_bytes();
    assert_eq!(
        Model::from_bytes(b"nope!").err(),
        Some(DecodeError::BadMagic)
    );
    assert_eq!(Model::from_bytes(b"").err(), Some(DecodeError::BadMagic));
    for cut in 0..bytes.len() {
        assert!(Model::from_bytes(&bytes[..cut]).is_err(), "cut {cut}");
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert_eq!(
        Model::from_bytes(&extra).err(),
        Some(DecodeError::TrailingBytes)
    );
    // wrong kind
    assert_eq!(
        HierModel::from_bytes(&bytes).err(),
        Some(DecodeError::WrongKind)
    );
    // absurd sizes must error, not allocate
    let mut huge = bytes.clone();
    huge[5..9].copy_from_slice(&u32::MAX.to_le_bytes());
    huge[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Model::from_bytes(&huge).is_err());
    let mut tiny = bytes.clone();
    tiny[5..9].copy_from_slice(&1u32.to_le_bytes());
    assert_eq!(
        Model::from_bytes(&tiny).err(),
        Some(DecodeError::Invalid("sizes"))
    );
}
