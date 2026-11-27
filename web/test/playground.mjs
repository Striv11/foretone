import assert from "node:assert/strict";
import { Playground, keyToNote, parseNotes, sparkline, freqOf, isBlack, NOTE_NAMES, shortcut, heatColor, predictionError } from "../playground.mjs";

// Fake model: predicts uniformly until it has seen a symbol, then predicts that symbol again.
function fake() {
  let last = -1;
  const n = NOTE_NAMES.length;
  return {
    predict() { const p = new Float32Array(n).fill(0.01); if (last >= 0) p[last] = 1 - 0.01 * (n - 1); return p; },
    observe(i) { const s = -Math.log(this.predict()[i]); last = i; return s; },
    reset() { last = -1; },
  };
}

assert.equal(keyToNote("a"), 0);
assert.equal(keyToNote("K"), 12);
assert.equal(keyToNote("z"), -1);
assert.equal(keyToNote("Enter"), -1);
assert.equal(keyToNote(undefined), -1);

assert.deepEqual(parseNotes(" C4 d4,E4  "), { notes: [0, 2, 4], bad: [] });
assert.deepEqual(parseNotes("C4 H9 C#4"), { notes: [0, 1], bad: ["H9"] });
assert.deepEqual(parseNotes(""), { notes: [], bad: [] });

assert.ok(Math.abs(freqOf(9) - 440) < 1e-9);
assert.ok(isBlack(1) && !isBlack(0));

assert.equal(sparkline([], 100, 10), "");
assert.equal(sparkline([0, 100], 100, 10, 2), "0.0,10.0 100.0,0.0");
assert.equal(sparkline([1], 100, 10, 2), "0.0,5.0");

const pg = new Playground(fake(), 3);
const a = pg.play(0);
assert.equal(a.top, 0);
const b = pg.play(0);
assert.ok(b.surprise < a.surprise, "repeat is less surprising");
assert.ok(b.expected > 0.8);
const c = pg.play(5);
assert.ok(c.surprise > 3 && c.top === 0, "violation spikes surprise");
pg.play(5);
assert.equal(pg.history.length, 3, "history capped");
assert.equal(pg.total, 4);
assert.throws(() => pg.play(13), RangeError);
assert.throws(() => pg.play(1.5), RangeError);
pg.newPhrase();
assert.ok(Math.abs(pg.expectation()[0] - 0.01) < 1e-6);
pg.clear();
assert.equal(pg.history.length, 0);
console.log("playground tests ok");

// Day 6: shortcuts, heat colours, error trace, dream, latent history.
assert.equal(shortcut("1"), "dream");
assert.equal(shortcut("3"), "clear");
assert.equal(shortcut("a"), null);
assert.equal(shortcut(undefined), null);
assert.notEqual(heatColor(0.9), heatColor(-0.9));
assert.equal(heatColor(NaN), heatColor(0), "NaN is drawn as silent");
assert.equal(heatColor(5), heatColor(1), "clamped");
assert.equal(predictionError({ expected: 0.25 }), 0.75);
{
  const m = fake();
  const plain = new Playground(m);
  assert.deepEqual(plain.dream(4), [], "model without dream support");
  assert.equal(plain.latent().length, 0);
  m.latent = () => new Float32Array([0.5, -0.5]);
  m.dream = (n) => Array(n).fill(2);
  const q = new Playground(m);
  const e1 = q.play(0), e2 = q.play(0);
  assert.deepEqual(Array.from(e1.latent), [0.5, -0.5]);
  assert.deepEqual(q.dream(3), [2, 2, 2]);
  assert.equal(q.errors().length, 2);
  assert.ok(q.errors()[1] < q.errors()[0], "repeat is better predicted");
  assert.equal(q.total, 2, "dreaming does not count as playing");
}
