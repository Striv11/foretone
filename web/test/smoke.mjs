// Node smoke test: replay a melody through the .wasm and compare with native output.
// Usage: node web/test/smoke.mjs <foretone_wasm.wasm> <native-reference.txt>
import { readFileSync } from "node:fs";
import { Foretone } from "../foretone.mjs";
import assert from "node:assert/strict";

const [wasmPath, refPath] = process.argv.slice(2);
const TOL = Number(process.env.TOL ?? 1e-3);
const tune = [0, 1, 2, 3, 2, 1, 0, 0];
const seq = Array.from({ length: 160 }, (_, i) => tune[i % tune.length]);

const m = await Foretone.create(readFileSync(wasmPath), { vocab: 4, hidden: 16, seed: 42n });
const lines = seq.map((s) => {
  const surprise = m.observe(s);
  const p = m.predict();
  return `${surprise.toFixed(6)} ${Array.from(p, (v) => v.toFixed(6)).join(" ")}`;
});
const native = readFileSync(refPath, "utf8").trim().split("\n");
// Not bit-exact: wasm uses Rust's libm for tanh/exp, native uses the system libm.
// Divergence is small but compounds through online learning, so compare loosely.
assert.equal(lines.length, native.length);
let worst = 0;
lines.forEach((l, i) => {
  const a = l.split(" ").map(Number), b = native[i].split(" ").map(Number);
  a.forEach((v, j) => (worst = Math.max(worst, Math.abs(v - b[j]))));
});
console.log(`max |wasm - native| = ${worst.toExponential(2)}`);
assert.ok(worst < TOL, "wasm output differs from native");

// learned something
const dream = m.predict();
assert.equal(dream.indexOf(Math.max(...dream)), 0, "after C C expects C D... sanity");

// export/import round trip continues identically
const bytes = m.export();
const m2 = await Foretone.create(readFileSync(wasmPath), { vocab: 4, hidden: 16, seed: 1n });
m2.import(bytes);
for (const s of seq.slice(0, 16)) assert.equal(m.observe(s), m2.observe(s));
assert.throws(() => m2.import(bytes.slice(0, 10)), /invalid model data/);
assert.throws(() => m.observe(99), RangeError);
m.reset();
assert.ok(m.predict().every((p) => Math.abs(p - 0.25) < 1e-6));
m.free(); m2.free();
console.log(`ok: ${seq.length} observations match native; export ${bytes.length} bytes`);
