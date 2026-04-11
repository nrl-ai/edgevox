// Mic capture wrapper. Loads the AudioWorklet, downsamples to 16 kHz int16,
// and forwards 512-sample frames to ``onFrame``.

import workletUrl from "../worklets/recorder.worklet.js?url";

export interface MicCapture {
  start: () => Promise<void>;
  stop: () => void;
  isRunning: () => boolean;
}

export function createMicCapture(onFrame: (pcm: ArrayBuffer) => void): MicCapture {
  let ctx: AudioContext | null = null;
  let stream: MediaStream | null = null;
  let node: AudioWorkletNode | null = null;
  let running = false;

  return {
    async start() {
      if (running) return;
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      // Browsers ignore the requested sampleRate hint on most platforms — the
      // worklet handles whatever rate the device gives us.
      ctx = new AudioContext();
      await ctx.audioWorklet.addModule(workletUrl);
      const source = ctx.createMediaStreamSource(stream);
      node = new AudioWorkletNode(ctx, "recorder-processor");
      node.port.onmessage = (event) => onFrame(event.data as ArrayBuffer);
      source.connect(node);
      // Don't connect node → destination, otherwise the user hears themselves.
      running = true;
    },
    stop() {
      if (!running) return;
      running = false;
      try {
        node?.disconnect();
      } catch {
        // ignore
      }
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close();
      ctx = null;
      stream = null;
      node = null;
    },
    isRunning() {
      return running;
    },
  };
}
