/* AudioWorkletProcessor for voice input (console/static/voice.js): posts the microphone's mono Float32 samples.
 *
 * The node is created with channelCount 1 / channelCountMode 'explicit', so the graph already mixes the input down
 * to mono; any extra channels are averaged here anyway. Samples are batched into chunks of `chunkFrames`
 * (processorOptions, default 2048) and posted as {type: 'chunk', samples: Float32Array} (the buffer is
 * transferred). Messages in: {type: 'flush'} posts the partial chunk, then {type: 'flushed'};
 * {type: 'stop'} ends processing. The node outputs silence.
 */
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.size = Math.max(128, Math.floor(o.chunkFrames || 2048));
    this.buf = new Float32Array(this.size);
    this.n = 0;
    this.stopped = false;
    this.port.onmessage = (e) => {
      const type = e.data && e.data.type;
      if (type === 'flush') {
        this.flush();
        this.port.postMessage({ type: 'flushed' });
      } else if (type === 'stop') {
        this.flush();
        this.stopped = true;
      }
    };
  }

  flush() {
    if (!this.n) return;
    const out = this.buf.slice(0, this.n);
    this.n = 0;
    this.port.postMessage({ type: 'chunk', samples: out }, [out.buffer]);
  }

  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels || !channels.length || !channels[0]) return true;  // no input connected yet
    const frames = channels[0].length;
    const k = channels.length;
    for (let i = 0; i < frames; i += 1) {
      let s = channels[0][i];
      if (k > 1) {
        for (let c = 1; c < k; c += 1) s += channels[c][i];
        s /= k;
      }
      this.buf[this.n] = s;
      this.n += 1;
      if (this.n === this.size) {
        const out = this.buf;
        this.buf = new Float32Array(this.size);
        this.n = 0;
        this.port.postMessage({ type: 'chunk', samples: out }, [out.buffer]);
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
