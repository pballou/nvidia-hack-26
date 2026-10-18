/* AudioWorkletProcessor for voice input (console/static/voice.js): the microphone's mono Float32 samples, with a
 * short pre-roll ring so a take can start a moment before the key went down.
 *
 * The node is created with channelCount 1 / channelCountMode 'explicit', so the graph already mixes the input down
 * to mono; any extra channels are averaged here anyway. The node outputs silence.
 *
 * Two modes. Between takes ("warm", not recording) nothing is posted: the newest `ringFrames` samples
 * (processorOptions, about 400 ms) are kept in a ring and older ones are overwritten. If processing stalls for more
 * than GAP_MS (a suspended context, a hidden page), the ring is emptied: what it held is no longer "just now".
 * While recording, samples are batched into chunks of `chunkFrames` (processorOptions, default 2048) and posted.
 *
 * Messages in:
 *   {type: 'start', id, preRollMs, maxAgeMs}  begin take `id`. First posts {type: 'preroll', id, samples, ms}: the
 *       ring's newest samples, at most preRollMs long, and none older than maxAgeMs of wall time (samples may be
 *       empty). Then {type: 'chunk', id, samples} as audio arrives; the first chunk goes out after the first render
 *       quantum, so the page knows at once that frames are arriving. The ring is emptied.
 *   {type: 'end', id, discard?}  end take `id`: posts what is buffered as a last chunk (unless discard), then
 *       {type: 'ended', id}, and goes back to filling an empty ring. Audio is never sent twice.
 *   {type: 'stop'}  ends processing for good (the mic is being released).
 * Every Float32Array is posted with its buffer transferred.
 */
const GAP_MS = 250;

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.size = Math.max(128, Math.floor(o.chunkFrames || 2048));
    this.buf = new Float32Array(this.size);
    this.n = 0;
    this.cap = Math.max(0, Math.floor(o.ringFrames || 0));
    this.ring = new Float32Array(Math.max(1, this.cap));
    this.ringLen = 0;
    this.ringPos = 0;  // where the next sample goes
    this.lastAt = 0;   // Date.now() at the last render quantum with input
    this.take = null;  // the id of the take being recorded, or null
    this.firstPending = false;
    this.stopped = false;
    this.port.onmessage = (e) => this.onMessage(e.data || {});
  }

  onMessage(d) {
    if (d.type === 'start') {
      this.n = 0;  // a take that never got its 'end' is dropped
      const age = this.lastAt ? Date.now() - this.lastAt : Infinity;
      const maxAge = Number(d.maxAgeMs) || 0;
      const fresh = Number.isFinite(age) ? Math.max(0, maxAge - age) : 0;  // ms of the ring still young enough
      const want = Math.min(this.ringLen, Math.floor((Math.max(0, Number(d.preRollMs) || 0) * sampleRate) / 1000),
        Math.floor((fresh * sampleRate) / 1000));
      const pre = this.ringTail(want);
      this.ringLen = 0;
      this.take = d.id;
      this.firstPending = true;
      this.port.postMessage({ type: 'preroll', id: d.id, samples: pre, ms: (pre.length * 1000) / sampleRate },
        [pre.buffer]);
    } else if (d.type === 'end') {
      if (this.take !== null && this.take === d.id && !d.discard) this.flush();
      this.n = 0;
      this.take = null;
      this.ringLen = 0;
      this.port.postMessage({ type: 'ended', id: d.id });
    } else if (d.type === 'stop') {
      this.take = null;
      this.stopped = true;
    }
  }

  /** The newest `k` samples of the ring, oldest first. */
  ringTail(k) {
    const out = new Float32Array(Math.max(0, k));
    if (!k) return out;
    const start = (this.ringPos - k + this.cap) % this.cap;
    const first = Math.min(k, this.cap - start);
    out.set(this.ring.subarray(start, start + first), 0);
    if (first < k) out.set(this.ring.subarray(0, k - first), first);
    return out;
  }

  flush() {
    if (!this.n) return;
    const out = this.buf.slice(0, this.n);
    this.n = 0;
    this.port.postMessage({ type: 'chunk', id: this.take, samples: out }, [out.buffer]);
  }

  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels || !channels.length || !channels[0]) return true;  // no input connected yet
    const now = Date.now();
    if (this.lastAt && now - this.lastAt > GAP_MS) this.ringLen = 0;  // stale: from before a stall
    this.lastAt = now;
    const frames = channels[0].length;
    const k = channels.length;
    const recording = this.take !== null;
    for (let i = 0; i < frames; i += 1) {
      let s = channels[0][i];
      if (k > 1) {
        for (let c = 1; c < k; c += 1) s += channels[c][i];
        s /= k;
      }
      if (recording) {
        this.buf[this.n] = s;
        this.n += 1;
        if (this.n === this.size) {
          const out = this.buf;
          this.buf = new Float32Array(this.size);
          this.n = 0;
          this.firstPending = false;
          this.port.postMessage({ type: 'chunk', id: this.take, samples: out }, [out.buffer]);
        }
      } else if (this.cap) {
        this.ring[this.ringPos] = s;
        this.ringPos = (this.ringPos + 1) % this.cap;
        if (this.ringLen < this.cap) this.ringLen += 1;
      }
    }
    if (recording && this.firstPending) {  // frames are arriving for this take: say so now, not a chunk later
      this.firstPending = false;
      this.flush();
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
