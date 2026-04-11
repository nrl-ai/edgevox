// FIFO WAV player: queue WAV blobs from the server and play them back-to-back.
// We use HTMLAudioElement instead of AudioContext.decodeAudioData because the
// server already encoded a complete WAV per sentence — Audio handles the
// container parsing for us and works on every modern browser.

export class WavQueuePlayer {
  private queue: Blob[] = [];
  private playing = false;
  private current: HTMLAudioElement | null = null;
  private onIdle: (() => void) | null = null;

  enqueue(blob: Blob) {
    this.queue.push(blob);
    void this.tick();
  }

  /** Drop any queued/playing audio immediately. */
  flush() {
    this.queue = [];
    if (this.current) {
      this.current.pause();
      this.current.src = "";
      this.current = null;
    }
    this.playing = false;
  }

  isBusy() {
    return this.playing || this.queue.length > 0;
  }

  setOnIdle(cb: (() => void) | null) {
    this.onIdle = cb;
  }

  private async tick() {
    if (this.playing) return;
    const next = this.queue.shift();
    if (!next) {
      this.onIdle?.();
      return;
    }
    this.playing = true;
    const url = URL.createObjectURL(next);
    const audio = new Audio(url);
    this.current = audio;
    try {
      await audio.play();
      await new Promise<void>((resolve) => {
        audio.onended = () => resolve();
        audio.onerror = () => resolve();
      });
    } finally {
      URL.revokeObjectURL(url);
      this.current = null;
      this.playing = false;
      void this.tick();
    }
  }
}
