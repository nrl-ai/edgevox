// Typed WebSocket wrapper around the EdgeVox server protocol. Binary frames
// from the server are always WAV bytes for the most recent ``bot_sentence``
// announcement.

export type ServerMessage =
  | { type: "ready"; session_id: string; language: string; languages: string[]; voice: string; voices: string[]; tts_sample_rate: number; sample_rate: number; frame_size: number }
  | { type: "state"; value: "listening" | "transcribing" | "thinking" | "speaking" }
  | { type: "level"; value: number }
  | { type: "user_text"; text: string; latency: number }
  | { type: "bot_token"; text: string }
  | { type: "bot_sentence"; text: string; audio_id: number; sample_rate: number; bytes: number }
  | { type: "bot_text"; text: string; latency: number }
  | { type: "metrics"; stt: number; llm: number; ttft: number; tts: number; total: number; audio_duration: number }
  | { type: "info"; message: string }
  | { type: "error"; message: string }
  | { type: "language_changed"; language: string; voice?: string; voices?: string[] }
  | { type: "voice_changed"; voice: string };

export interface WsHandlers {
  onJson: (msg: ServerMessage) => void;
  onAudio: (blob: Blob) => void;
  onOpen?: () => void;
  onClose?: (ev: CloseEvent) => void;
  onError?: (ev: Event) => void;
}

export class EdgeVoxWs {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: WsHandlers;

  constructor(url: string, handlers: WsHandlers) {
    this.url = url;
    this.handlers = handlers;
  }

  connect() {
    const ws = new WebSocket(this.url);
    ws.binaryType = "blob";
    ws.onopen = () => this.handlers.onOpen?.();
    ws.onclose = (ev) => this.handlers.onClose?.(ev);
    ws.onerror = (ev) => this.handlers.onError?.(ev);
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data) as ServerMessage;
          this.handlers.onJson(msg);
        } catch {
          // ignore malformed
        }
      } else if (ev.data instanceof Blob) {
        this.handlers.onAudio(ev.data);
      }
    };
    this.ws = ws;
  }

  sendAudio(buffer: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(buffer);
    }
  }

  sendControl(payload: object) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  close() {
    this.ws?.close();
    this.ws = null;
  }

  isOpen() {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
