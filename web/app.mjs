import { Foretone } from "./foretone.mjs";
import { Playground, NOTE_NAMES, isBlack, freqOf, keyToNote, parseNotes, sparkline } from "./playground.mjs";

const $ = (id) => document.getElementById(id);
const status = (msg, err = false) => { $("status").textContent = msg; $("status").className = err ? "err" : ""; };
let pg = null, audio = null;

function beep(i) {
  if (!$("sound").checked) return;
  audio ??= new (window.AudioContext || window.webkitAudioContext)();
  const t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
  o.type = "triangle"; o.frequency.value = freqOf(i);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
  o.connect(g).connect(audio.destination); o.start(t); o.stop(t + 0.55);
}

function render(last) {
  const p = pg.expectation(), bars = $("bars");
  bars.replaceChildren(...NOTE_NAMES.map((n, i) => {
    const d = document.createElement("div");
    d.className = "bar" + (last && last.top === i ? " hit" : "");
    d.title = `${n}: ${(p[i] * 100).toFixed(1)}%`;
    const b = document.createElement("i"); b.style.height = `${p[i] * 100}%`;
    d.append(b, n); return d;
  }));
  const s = pg.surprises();
  $("empty").hidden = s.length > 0; $("spark").hidden = s.length === 0;
  $("spark").innerHTML = `<polyline points="${sparkline(s, 300, 60)}"/>`;
  if (last) status(`${NOTE_NAMES[last.note]}: surprise ${last.surprise.toFixed(2)} nats` +
    ` (model gave it ${(last.expected * 100).toFixed(0)}%; expected ${NOTE_NAMES[last.top]})`);
}

function play(i) {
  if (!pg) return;
  beep(i);
  const el = $("keys").children[i];
  el.classList.add("active"); setTimeout(() => el.classList.remove("active"), 120);
  render(pg.play(i));
}

async function playSeq() {
  const { notes, bad } = parseNotes($("seq").value);
  if (bad.length) return status(`Unknown notes: ${bad.join(", ")} (use C4…C5, e.g. C#4)`, true);
  if (!notes.length) return status("Type some notes first, e.g. C4 D4 E4", true);
  $("playseq").disabled = true;
  for (const n of notes) { play(n); await new Promise((r) => setTimeout(r, 260)); }
  $("playseq").disabled = false;
}

function buildKeys() {
  $("keys").replaceChildren(...NOTE_NAMES.map((n, i) => {
    const b = document.createElement("button");
    b.className = "key" + (isBlack(i) ? " black" : "");
    b.innerHTML = `${n}<small>${"awsedftgyhujk"[i].toUpperCase()}</small>`;
    b.setAttribute("aria-label", `Play ${n}`);
    b.addEventListener("pointerdown", (e) => { e.preventDefault(); play(i); });
    b.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); play(i); } });
    return b;
  }));
}

buildKeys();
document.addEventListener("keydown", (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.target.tagName === "INPUT") return;
  const i = keyToNote(e.key);
  if (i >= 0) play(i);
});
$("playseq").addEventListener("click", playSeq);
$("phrase").addEventListener("click", () => { pg?.newPhrase(); render(null); status("Context cleared; learned weights kept."); });
$("clear").addEventListener("click", () => { pg?.clear(); render(null); status("Model reset."); });

try {
  const res = await fetch("pkg/foretone_wasm.wasm");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const model = await Foretone.create(await res.arrayBuffer(), { vocab: NOTE_NAMES.length, hidden: 24 });
  pg = new Playground(model);
  render(null);
  status("Ready. Play C D E G E D C a few times, then change a note.");
} catch (e) {
  status(`Could not load the WebAssembly model (${e.message}). The page needs to be served over http(s).`, true);
}
