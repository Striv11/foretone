import { Foretone } from "./foretone.mjs";
import { Playground, NOTE_NAMES, isBlack, freqOf, keyToNote, parseNotes, sparkline, shortcut, heatColor } from "./playground.mjs";

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
  $("err").hidden = s.length === 0;
  $("err").innerHTML = `<polyline points="${sparkline(pg.errors(), 300, 40, 1)}"/>`;
  renderHeat();
  if (last) status(`${NOTE_NAMES[last.note]}: surprise ${last.surprise.toFixed(2)} nats` +
    ` (model gave it ${(last.expected * 100).toFixed(0)}%; expected ${NOTE_NAMES[last.top]})`);
}

function renderHeat() {
  const cols = pg.history.map((e) => e.latent).filter((l) => l && l.length).slice(-24);
  const heat = $("heat");
  if (!cols.length) return heat.replaceChildren();
  const rows = cols[0].length;
  heat.style.setProperty("--rows", rows);
  heat.style.gridTemplateColumns = `repeat(${cols.length}, 1fr)`;
  heat.replaceChildren(...cols.flatMap((l) => Array.from(l, (v) => {
    const c = document.createElement("i"); c.style.background = heatColor(v); return c;
  })));
}

function play(i, learn = true) {
  if (!pg) return;
  beep(i);
  if (!learn) return flash(i);
  flash(i);
  render(pg.play(i));
}

function flash(i) {
  const el = $("keys").children[i];
  el.classList.add("active"); setTimeout(() => el.classList.remove("active"), 120);
}

let busy = false;
async function dream() {
  if (!pg || busy) return;
  if (!pg.total) return status("Play a melody first; it dreams what it has learned.", true);
  busy = true; $("dream").disabled = true;
  try {
    const seq = pg.dream(16);
    $("dreamout").textContent = seq.map((i) => NOTE_NAMES[i]).join(" ");
    for (const n of seq) { play(n, false); await new Promise((r) => setTimeout(r, 260)); }
  } finally { busy = false; $("dream").disabled = false; }
}

function act(name) {
  if (name === "dream") dream();
  else if (name === "phrase") $("phrase").click();
  else if (name === "clear") $("clear").click();
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
  const a = shortcut(e.key);
  if (a) return act(a);
  const i = keyToNote(e.key);
  if (i >= 0) play(i);
});
$("dream").addEventListener("click", dream);
$("playseq").addEventListener("click", playSeq);
$("phrase").addEventListener("click", () => { pg?.newPhrase(); render(null); status("Context cleared; learned weights kept."); });
$("clear").addEventListener("click", () => { pg?.clear(); render(null); status("Model reset."); $("dreamout").textContent = ""; });

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
