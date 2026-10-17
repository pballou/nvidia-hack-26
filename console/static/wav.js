/* WAV helpers for voice input: pure functions, no DOM, no network. Exposed as window.AcmeWav.
 *
 *   encodeWav(float32Samples, sampleRate) -> ArrayBuffer   PCM16 mono WAV (44-byte RIFF header + samples)
 *   resample(float32Samples, fromRate, toRate = 16000) -> Float32Array
 *       Downsampling averages each output sample's span of input (a box filter with fractional edges), which is a
 *       simple anti-alias low-pass; upsampling interpolates linearly. Same rate returns a copy.
 *   rms(float32Samples) -> number   root mean square, 0 for empty input
 *
 * console/asr.py accepts only 16 kHz mono PCM16, so the browser records at 16 kHz when it can and resamples otherwise.
 */
'use strict';

(() => {
  const TARGET_RATE = 16000;

  function writeAscii(view, offset, text) {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  }

  function encodeWav(samples, sampleRate) {
    const n = samples ? samples.length : 0;
    const rate = Math.round(Number(sampleRate));
    if (!Number.isFinite(rate) || rate <= 0) throw new RangeError(`bad sample rate: ${sampleRate}`);
    const dataBytes = n * 2;
    const buf = new ArrayBuffer(44 + dataBytes);
    const v = new DataView(buf);
    writeAscii(v, 0, 'RIFF');
    v.setUint32(4, 36 + dataBytes, true);  // RIFF chunk size: everything after this field
    writeAscii(v, 8, 'WAVE');
    writeAscii(v, 12, 'fmt ');
    v.setUint32(16, 16, true);             // fmt chunk size (PCM)
    v.setUint16(20, 1, true);              // audio format 1 = integer PCM
    v.setUint16(22, 1, true);              // mono
    v.setUint32(24, rate, true);
    v.setUint32(28, rate * 2, true);       // byte rate = rate * channels * 2 bytes
    v.setUint16(32, 2, true);              // block align = channels * 2 bytes
    v.setUint16(34, 16, true);             // bits per sample
    writeAscii(v, 36, 'data');
    v.setUint32(40, dataBytes, true);
    for (let i = 0, off = 44; i < n; i += 1, off += 2) {
      let s = Number(samples[i]);
      if (Number.isNaN(s)) s = 0;          // NaN -> silence
      s = s < -1 ? -1 : (s > 1 ? 1 : s);
      v.setInt16(off, Math.round(s < 0 ? s * 0x8000 : s * 0x7fff), true);
    }
    return buf;
  }

  function resample(samples, fromRate, toRate = TARGET_RATE) {
    const src = samples || new Float32Array(0);
    const from = Number(fromRate);
    const to = Number(toRate);
    if (!(from > 0) || !(to > 0)) throw new RangeError(`bad sample rates: ${fromRate} -> ${toRate}`);
    if (from === to || src.length === 0) return Float32Array.from(src);
    const ratio = from / to;  // input samples per output sample
    const n = src.length;
    if (ratio > 1) {
      // Area average: output i covers input [i*ratio, (i+1)*ratio), partial samples weighted by overlap.
      const outLen = Math.floor(n / ratio);
      const out = new Float32Array(outLen);
      for (let i = 0; i < outLen; i += 1) {
        const start = i * ratio;
        const end = Math.min(start + ratio, n);
        let pos = start;
        let j = Math.floor(start);
        let acc = 0;
        while (pos < end && j < n) {
          const next = Math.min(j + 1, end);
          acc += src[j] * (next - pos);
          pos = next;
          j += 1;
        }
        out[i] = pos > start ? acc / (pos - start) : 0;
      }
      return out;
    }
    // Upsampling (a mic slower than 16 kHz): linear interpolation.
    const outLen = Math.floor(n / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i += 1) {
      const x = i * ratio;
      const j = Math.floor(x);
      const f = x - j;
      const a = src[j];
      const b = j + 1 < n ? src[j + 1] : a;
      out[i] = a + (b - a) * f;
    }
    return out;
  }

  function rms(samples) {
    const n = samples ? samples.length : 0;
    if (!n) return 0;
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += samples[i] * samples[i];
    return Math.sqrt(sum / n);
  }

  const api = Object.freeze({ encodeWav, resample, rms, TARGET_RATE });
  if (typeof window !== 'undefined') window.AcmeWav = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
