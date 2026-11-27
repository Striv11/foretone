// Pure playground logic (no DOM, no audio) so it can be unit-tested in Node.
export const NOTE_NAMES = ["C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4","C5"];
const KEYS = "awsedftgyhujk";

export const isBlack = (i) => NOTE_NAMES[i].includes("#");
export const midiOf = (i) => 60 + i;
export const freqOf = (i) => 440 * Math.pow(2, (midiOf(i) - 69) / 12);

/** Map a keyboard key to a note index, or -1. Ignores modified/repeated keys. */
export function keyToNote(key) {
  if (typeof key !== "string" || key.length !== 1) return -1;
  return KEYS.indexOf(key.toLowerCase());
}

/** Parse "C4 D4 E4" style text into indices; returns {notes, bad} (bad = unknown tokens). */
export function parseNotes(text) {
  const notes = [], bad = [];
  for (const t of text.trim().split(/[\s,]+/).filter(Boolean)) {
    const i = NOTE_NAMES.indexOf(t[0].toUpperCase() + t.slice(1));
    (i < 0 ? bad : notes).push(i < 0 ? t : i);
  }
  return { notes, bad };
}

/** SVG polyline points for values scaled to [0,max] in a w×h box. */
export function sparkline(values, w, h, max = Math.log(NOTE_NAMES.length) * 2) {
  if (values.length === 0) return "";
  const step = values.length > 1 ? w / (values.length - 1) : 0;
  return values
    .map((v, i) => `${(i * step).toFixed(1)},${(h - Math.min(Math.max(v, 0), max) / max * h).toFixed(1)}`)
    .join(" ");
}

const SHORTCUTS = { "1": "dream", "2": "phrase", "3": "clear" };
/** Map a non-note shortcut key (1 dream, 2 new phrase, 3 reset) to an action name, or null. */
export const shortcut = (key) => SHORTCUTS[key] ?? null;

/** CSS colour for a latent activation in [-1,1]: warm for positive, cool for negative. */
export function heatColor(v) {
  const a = Math.min(Math.abs(Number.isFinite(v) ? v : 0), 1);
  return `hsl(${v < 0 ? 210 : 340} 70% ${(92 - a * 52).toFixed(0)}%)`;
}

/** Prediction error of one step: 1 - probability the model gave the note (0 = foreseen). */
export const predictionError = (e) => 1 - e.expected;

export class Playground {
  /** @param model object with observe(i)->surprise, predict()->Float32Array, reset() */
  constructor(model, maxHistory = 64) {
    this.model = model;
    this.maxHistory = maxHistory;
    this.history = [];
    this.total = 0;
  }
  /** Current next-note distribution. */
  expectation() { return this.model.predict(); }
  /** Play note i: returns {note, surprise, expected (prob it was given), top (most expected index)}. */
  play(i) {
    if (!Number.isInteger(i) || i < 0 || i >= NOTE_NAMES.length) throw new RangeError("bad note");
    const p = this.model.predict();
    let top = 0;
    for (let k = 1; k < p.length; k++) if (p[k] > p[top]) top = k;
    const surprise = this.model.observe(i);
    const e = { note: i, surprise, expected: p[i], top };
    if (typeof this.model.latent === "function") e.latent = this.model.latent();
    this.history.push(e);
    if (this.history.length > this.maxHistory) this.history.shift();
    this.total++;
    return e;
  }
  /** Latent activations after the latest note (empty if the model has none). */
  latent() { return this.model.latent ? this.model.latent() : new Float32Array(0); }
  /** Free-run n notes from the current state; the model does not learn from them. */
  dream(n = 8) { return this.model.dream ? this.model.dream(n) : []; }
  errors() { return this.history.map(predictionError); }
  /** Forget the current context but keep learned weights. */
  newPhrase() { this.model.reset(); }
  clear() { this.history = []; this.total = 0; this.model.reset(); }
  surprises() { return this.history.map((e) => e.surprise); }
}
