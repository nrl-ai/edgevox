// AudioWorklet that captures mono float32 from the mic, downsamples to
// 16 kHz, and posts int16 PCM frames of exactly RESAMPLED_FRAME samples back
// to the main thread. The server expects 512-sample (32 ms) frames.

const TARGET_SR = 16000;
const RESAMPLED_FRAME = 512;

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._inputSr = sampleRate;
    this._ratio = TARGET_SR / this._inputSr;
    this._resampleBuf = []; // accumulated float32 samples at 16 kHz
  }

  // Linear-interpolation downsample. Cheap and good enough for VAD/STT.
  _resample(input) {
    const outLen = Math.floor(input.length * this._ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i / this._ratio;
      const left = Math.floor(srcIdx);
      const right = Math.min(left + 1, input.length - 1);
      const frac = srcIdx - left;
      out[i] = input[left] * (1 - frac) + input[right] * frac;
    }
    return out;
  }

  process(inputs) {
    const channelData = inputs[0]?.[0];
    if (!channelData) return true;

    const resampled = this._resample(channelData);
    for (let i = 0; i < resampled.length; i++) {
      this._resampleBuf.push(resampled[i]);
    }

    while (this._resampleBuf.length >= RESAMPLED_FRAME) {
      const frame = this._resampleBuf.splice(0, RESAMPLED_FRAME);
      const pcm = new Int16Array(RESAMPLED_FRAME);
      for (let i = 0; i < RESAMPLED_FRAME; i++) {
        const s = Math.max(-1, Math.min(1, frame[i]));
        pcm[i] = s < 0 ? s * 32768 : s * 32767;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
