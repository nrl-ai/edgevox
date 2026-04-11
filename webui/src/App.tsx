import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Mic, MicOff, RotateCcw, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  ConversationView,
  type Message,
} from "@/components/ConversationView";
import {
  StatusIndicator,
  type BotState,
} from "@/components/StatusIndicator";
import { MicMeter } from "@/components/MicMeter";
import { MetricsBar, type Metrics } from "@/components/MetricsBar";
import { createMicCapture } from "@/lib/audio-capture";
import { WavQueuePlayer } from "@/lib/audio-playback";
import { EdgeVoxWs, type ServerMessage } from "@/lib/ws-client";

function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

function ts(): string {
  return new Date().toLocaleTimeString("en-GB", { hour12: false });
}

const COMMANDS: Record<string, string> = {
  "/help": "Show all available commands",
  "/reset": "Reset conversation history",
  "/lang ": "Switch language (en, vi, fr, ko, ...)",
  "/langs": "List all supported languages",
  "/say ": "TTS preview — speak text directly",
};

export default function App() {
  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [state, setState] = useState<BotState>("idle");
  const [level, setLevel] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [info, setInfo] = useState<{
    sessionId: string;
    language: string;
    languages: string[];
    voice: string;
    voices: string[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cmdValue, setCmdValue] = useState("");
  const [infoMsg, setInfoMsg] = useState<string | null>(null);

  const wsRef = useRef<EdgeVoxWs | null>(null);
  const playerRef = useRef<WavQueuePlayer>(new WavQueuePlayer());
  const micRef = useRef<ReturnType<typeof createMicCapture> | null>(null);
  const pendingBotIdRef = useRef<string | null>(null);
  const expectedAudioRef = useRef<Map<number, string>>(new Map());
  const lastAudioIdRef = useRef<number | null>(null);

  const ensurePendingBotMessage = useCallback(() => {
    if (pendingBotIdRef.current) return pendingBotIdRef.current;
    const id = `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    pendingBotIdRef.current = id;
    setMessages((prev) => [...prev, { id, role: "bot", text: "", pending: true, timestamp: ts() }]);
    return id;
  }, []);

  const finalizeBotMessage = useCallback(() => {
    const id = pendingBotIdRef.current;
    if (!id) return;
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, pending: false } : m))
    );
    pendingBotIdRef.current = null;
  }, []);

  const handleJson = useCallback(
    (msg: ServerMessage) => {
      switch (msg.type) {
        case "ready":
          setInfo({
            sessionId: msg.session_id,
            language: msg.language,
            languages: msg.languages,
            voice: msg.voice,
            voices: msg.voices,
          });
          setState("listening");
          break;
        case "state":
          setState(msg.value);
          break;
        case "level":
          setLevel(msg.value);
          break;
        case "user_text": {
          const id = `user-${Date.now()}`;
          setMessages((prev) => [
            ...prev,
            { id, role: "user", text: msg.text, timestamp: ts() },
          ]);
          break;
        }
        case "bot_token": {
          const id = ensurePendingBotMessage();
          setMessages((prev) =>
            prev.map((m) =>
              m.id === id ? { ...m, text: m.text + msg.text } : m
            )
          );
          break;
        }
        case "bot_sentence":
          expectedAudioRef.current.set(msg.audio_id, msg.text);
          lastAudioIdRef.current = msg.audio_id;
          break;
        case "bot_text":
          finalizeBotMessage();
          break;
        case "metrics":
          setMetrics(msg);
          break;
        case "error":
          setError(msg.message);
          setTimeout(() => setError(null), 5000);
          break;
        case "info":
          setInfoMsg(msg.message);
          setTimeout(() => setInfoMsg(null), 3000);
          break;
        case "language_changed":
          setInfo((prev) => prev ? {
            ...prev,
            language: msg.language,
            voice: msg.voice || prev.voice,
            voices: msg.voices || prev.voices,
          } : prev);
          break;
        case "voice_changed":
          setInfo((prev) => prev ? { ...prev, voice: msg.voice } : prev);
          break;
        default:
          break;
      }
    },
    [ensurePendingBotMessage, finalizeBotMessage]
  );

  const handleAudio = useCallback((blob: Blob) => {
    playerRef.current.enqueue(blob);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.isOpen()) return;
    setError(null);
    const ws = new EdgeVoxWs(wsUrl(), {
      onJson: handleJson,
      onAudio: handleAudio,
      onOpen: () => setConnected(true),
      onClose: () => {
        setConnected(false);
        setState("idle");
      },
      onError: () => setError("WebSocket error"),
    });
    ws.connect();
    wsRef.current = ws;
  }, [handleJson, handleAudio]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  const startMic = useCallback(async () => {
    if (recording) return;
    if (!wsRef.current?.isOpen()) connect();
    try {
      const mic = createMicCapture((pcm) => {
        wsRef.current?.sendAudio(pcm);
      });
      await mic.start();
      micRef.current = mic;
      setRecording(true);
    } catch (e) {
      setError(`mic error: ${(e as Error).message}`);
    }
  }, [recording, connect]);

  const stopMic = useCallback(() => {
    micRef.current?.stop();
    micRef.current = null;
    setRecording(false);
  }, []);

  const interrupt = useCallback(() => {
    wsRef.current?.sendControl({ type: "interrupt" });
    playerRef.current.flush();
    finalizeBotMessage();
  }, [finalizeBotMessage]);

  const reset = useCallback(() => {
    wsRef.current?.sendControl({ type: "reset" });
    playerRef.current.flush();
    setMessages([]);
    setMetrics(null);
    pendingBotIdRef.current = null;
  }, []);

  // Process slash commands or send text to chat
  const handleCommand = useCallback(
    (input: string) => {
      const trimmed = input.trim();
      if (!trimmed || !connected) return;

      if (trimmed === "/help") {
        const helpLines = Object.entries(COMMANDS)
          .map(([cmd, desc]) => `  ${cmd.padEnd(12)} ${desc}`)
          .join("\n");
        const id = `sys-${Date.now()}`;
        setMessages((prev) => [
          ...prev,
          { id, role: "bot", text: `Available commands:\n${helpLines}` },
        ]);
        return;
      }
      if (trimmed === "/reset") {
        reset();
        return;
      }
      if (trimmed === "/langs") {
        const langs = info?.languages?.join(", ") || "—";
        const id = `sys-${Date.now()}`;
        setMessages((prev) => [
          ...prev,
          { id, role: "bot", text: `Supported languages: ${langs}` },
        ]);
        return;
      }
      if (trimmed.startsWith("/lang ")) {
        const code = trimmed.slice(6).trim();
        wsRef.current?.sendControl({ type: "set_language", language: code });
        return;
      }
      if (trimmed.startsWith("/say ")) {
        const text = trimmed.slice(5).trim();
        wsRef.current?.sendControl({ type: "say", text });
        return;
      }
      if (trimmed.startsWith("/")) {
        setError(`Unknown command: ${trimmed.split(" ")[0]}`);
        return;
      }

      // Plain text → type-to-chat (skip STT, go straight to LLM)
      wsRef.current?.sendControl({ type: "text_input", text: trimmed });
    },
    [connected, reset, info]
  );

  useEffect(() => {
    connect();
    return () => {
      stopMic();
      disconnect();
      playerRef.current.flush();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const statusText = useMemo(() => {
    if (error) return <span className="text-neon-red font-mono text-xs">✘ {error}</span>;
    if (connected) return <span className="text-neon-green font-mono text-xs">✓ Connected</span>;
    return <span className="text-neon-orange font-mono text-xs">○ Disconnected</span>;
  }, [connected, error]);

  return (
    <div className="h-screen w-full flex flex-col bg-[#0a0e14]">
      {/* ── Header ── */}
      <header className="bg-[#111820] border-b border-[#1e3a2e] px-4 py-2 flex items-center justify-between font-mono">
        <div className="flex items-center gap-3">
          <span className="text-neon-green font-bold text-sm">EdgeVox</span>
          <span className="text-muted-foreground text-xs">/ web</span>
          {statusText}
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {info && (
            <>
              <span className="text-muted-foreground">[{info.language.toUpperCase()}]</span>
              <span className="text-neon-blue">{info.voice}</span>
              <span className="hidden sm:inline">{info.sessionId}</span>
            </>
          )}
        </div>
      </header>

      {/* ── Status Bar ── */}
      <div className="bg-[#111820] border-b border-[#1e3a2e] px-4 py-2 flex items-center justify-center gap-4">
        <StatusIndicator state={state} />
        {info && (
          <span className="text-muted-foreground font-mono text-xs">
            [{info.language.toUpperCase()}]
          </span>
        )}
        {infoMsg && (
          <span className="text-neon-cyan font-mono text-xs">
            {infoMsg}
          </span>
        )}
      </div>

      {/* ── Main area ── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat panel */}
        <main className="flex-[3] min-w-0 border border-[#1e3a2e] rounded-lg m-1 bg-[#0a0e14] overflow-hidden">
          <ConversationView messages={messages} />
        </main>

        {/* Side panel */}
        <aside className="hidden md:flex flex-col w-72 max-w-[280px] min-w-[220px] border border-[#1e3a2e] rounded-lg m-1 mr-1 bg-[#0d1117] font-mono text-xs overflow-y-auto">
          {/* Audio level */}
          <div className="px-3 py-2">
            <MicMeter level={level} />
          </div>
          <div className="border-t border-[#1e3a2e]" />

          {/* Models */}
          <div className="px-3 py-2">
            <div className="text-neon-green font-bold mb-1">■ Models</div>
            <div>
              <span className="text-muted-foreground">STT </span>
              <span className="text-neon-cyan">{info ? "whisper" : "..."}</span>
            </div>
            <div>
              <span className="text-muted-foreground">LLM </span>
              <span className="text-neon-purple">{info ? "gemma" : "..."}</span>
            </div>
            <div>
              <span className="text-muted-foreground">TTS </span>
              <span className="text-neon-blue">{info?.voice || "..."}</span>
            </div>
          </div>
          <div className="border-t border-[#1e3a2e]" />

          {/* Latency / Metrics */}
          <div className="px-3 py-2">
            <MetricsBar metrics={metrics} />
          </div>
          <div className="border-t border-[#1e3a2e]" />

          {/* Settings */}
          <div className="px-3 py-2">
            <div className="text-neon-green font-bold mb-2">■ Settings</div>
            <div className="mb-2">
              <label className="text-muted-foreground block mb-1">Language</label>
              <select
                className="w-full bg-[#0a0e14] border border-[#1e3a2e] rounded px-2 py-1 text-foreground text-xs font-mono focus:border-neon-green focus:outline-none"
                value={info?.language || "en"}
                onChange={(e) => {
                  wsRef.current?.sendControl({ type: "set_language", language: e.target.value });
                }}
              >
                {(info?.languages || ["en"]).map((lang) => (
                  <option key={lang} value={lang}>
                    {lang}
                  </option>
                ))}
              </select>
            </div>
            <div className="mb-2">
              <label className="text-muted-foreground block mb-1">Voice</label>
              <select
                className="w-full bg-[#0a0e14] border border-[#1e3a2e] rounded px-2 py-1 text-foreground text-xs font-mono focus:border-neon-green focus:outline-none"
                value={info?.voice || ""}
                onChange={(e) => {
                  wsRef.current?.sendControl({ type: "set_voice", voice: e.target.value });
                }}
              >
                {(info?.voices || []).map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="border-t border-[#1e3a2e]" />

          {/* Controls */}
          <div className="px-3 py-2">
            <div className="text-neon-green font-bold mb-2">■ Controls</div>
            <div className="flex flex-col gap-1.5">
              {!recording ? (
                <Button onClick={startMic} disabled={!connected} size="sm" className="w-full justify-start">
                  <Mic className="size-3" />
                  Start
                </Button>
              ) : (
                <Button variant="outline" onClick={stopMic} size="sm" className="w-full justify-start border-neon-red text-neon-red">
                  <MicOff className="size-3" />
                  Stop
                </Button>
              )}
              <Button variant="outline" onClick={interrupt} disabled={!connected} size="sm" className="w-full justify-start">
                <Square className="size-3" />
                Interrupt
              </Button>
              <Button variant="ghost" onClick={reset} disabled={!connected} size="sm" className="w-full justify-start">
                <RotateCcw className="size-3" />
                Reset
              </Button>
            </div>
          </div>
        </aside>
      </div>

      {/* ── Command Input (matches TUI #command-input) ── */}
      <div className="bg-[#111820] border-t border-[#1e3a2e] px-4 py-2">
        <input
          type="text"
          className="w-full bg-[#0d1117] border border-[#1e3a2e] rounded px-3 py-2 text-foreground font-mono text-sm placeholder-muted-foreground focus:border-neon-green focus:outline-none"
          placeholder="Type / for commands, or text to chat"
          value={cmdValue}
          onChange={(e) => setCmdValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && cmdValue.trim()) {
              handleCommand(cmdValue);
              setCmdValue("");
            }
          }}
        />
      </div>

      {/* ── Footer ── */}
      <footer className="bg-[#111820] border-t border-[#1e3a2e] px-4 py-1.5 flex items-center justify-between font-mono text-xs">
        <div className="flex items-center gap-4">
          <span>
            <span className="bg-[#1e3a2e] text-neon-cyan px-1.5 py-0.5 rounded-sm">Enter</span>
            <span className="text-muted-foreground ml-1">send</span>
          </span>
          <span>
            <span className="bg-[#1e3a2e] text-neon-cyan px-1.5 py-0.5 rounded-sm">/help</span>
            <span className="text-muted-foreground ml-1">commands</span>
          </span>
          <span>
            <span className="bg-[#1e3a2e] text-neon-cyan px-1.5 py-0.5 rounded-sm">/lang</span>
            <span className="text-muted-foreground ml-1">language</span>
          </span>
          <span>
            <span className="bg-[#1e3a2e] text-neon-cyan px-1.5 py-0.5 rounded-sm">/say</span>
            <span className="text-muted-foreground ml-1">TTS preview</span>
          </span>
        </div>

        {/* Mobile controls */}
        <div className="flex md:hidden items-center gap-2">
          {!recording ? (
            <Button onClick={startMic} disabled={!connected} size="sm">
              <Mic className="size-3" />
            </Button>
          ) : (
            <Button variant="outline" onClick={stopMic} size="sm" className="border-neon-red text-neon-red">
              <MicOff className="size-3" />
            </Button>
          )}
          <Button variant="outline" onClick={interrupt} disabled={!connected} size="sm">
            <Square className="size-3" />
          </Button>
          <Button variant="ghost" onClick={reset} disabled={!connected} size="sm">
            <RotateCcw className="size-3" />
          </Button>
        </div>
      </footer>
    </div>
  );
}
