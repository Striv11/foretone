// Thin wrapper over the foretone-wasm C ABI. Works in browsers and Node (ES module).
export class Foretone {
  /** @param {BufferSource|WebAssembly.Module} wasm */
  static async create(wasm, { vocab, hidden = 16, seed = 42n }) {
    const { instance } = await WebAssembly.instantiate(wasm, {});
    return new Foretone(instance.exports, vocab, hidden, BigInt(seed));
  }
  constructor(x, vocab, hidden, seed) {
    this.x = x;
    this.h = x.ft_new(vocab, hidden, Number(seed & 0xffffffffn), Number(seed >> 32n));
    if (!this.h) throw new Error("invalid model size");
    this.vocab = x.ft_vocab(this.h);
  }
  /** Observe symbol index; returns surprise in nats. */
  observe(symbol) {
    const s = this.x.ft_observe(this.h, symbol);
    if (s < 0) throw new RangeError("symbol out of range");
    return s;
  }
  /** Probabilities of the next symbol (a copy). */
  predict() {
    const p = this.x.ft_predict(this.h);
    return new Float32Array(this.x.memory.buffer, p, this.vocab).slice();
  }
  /** Latent activations r (a copy), for visualization. */
  latent() {
    const n = this.x.ft_hidden(this.h);
    return new Float32Array(this.x.memory.buffer, this.x.ft_latent(this.h), n).slice();
  }
  /** Free-run n symbols from the current state without changing the model. */
  dream(n) {
    const p = this.x.ft_alloc(4 * n);
    try {
      const k = this.x.ft_dream(this.h, p, n);
      if (k < 0) throw new RangeError("bad dream length");
      return Array.from(new Uint32Array(this.x.memory.buffer, p, k));
    } finally {
      this.x.ft_dealloc(p, 4 * n);
    }
  }
  reset() { this.x.ft_reset(this.h); }
  /** Serialize to bytes (a copy). */
  export() {
    const n = this.x.ft_export(this.h);
    return new Uint8Array(this.x.memory.buffer, this.x.ft_export_ptr(this.h), n).slice();
  }
  import(bytes) {
    const p = this.x.ft_alloc(bytes.length);
    try {
      new Uint8Array(this.x.memory.buffer, p, bytes.length).set(bytes);
      if (this.x.ft_import(this.h, p, bytes.length) !== 0) throw new Error("invalid model data");
      this.vocab = this.x.ft_vocab(this.h);
    } finally {
      this.x.ft_dealloc(p, bytes.length);
    }
  }
  free() { if (this.h) this.x.ft_free(this.h); this.h = 0; }
}
