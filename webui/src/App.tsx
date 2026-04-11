import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Mic, MicOff, RotateCcw, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
    voice: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<EdgeVoxWs | null>(null);
  const playerRef = useRef<WavQueuePlayer>(new WavQueuePlayer());
  const micRef = useRef<ReturnType<typeof createMicCapture> | null>(null);
  const pendingBotIdRef = useRef<string | null>(null);
  // Map audio_id → text (so binary frames are paired with their announcement)
  const expectedAudioRef = useRef<Map<number, string>>(new Map());
  const lastAudioIdRef = useRef<number | null>(null);

  const ensurePendingBotMessage = useCallback(() => {
    if (pendingBotIdRef.current) return pendingBotIdRef.current;
    const id = `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    pendingBotIdRef.current = id;
    setMessages((prev) => [...prev, { id, role: "bot", text: "", pending: true }]);
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
            voice: msg.voice,
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
            { id, role: "user", text: msg.text },
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
          break;
        case "info":
          // {"info":"history cleared"} etc — surface briefly
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

  // Connect on mount, clean up on unmount.
  useEffect(() => {
    connect();
    return () => {
      stopMic();
      disconnect();
      playerRef.current.flush();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const headerBadge = useMemo(() => {
    if (error) return <Badge variant="danger">{error}</Badge>;
    if (connected) return <Badge variant="success">Connected</Badge>;
    return <Badge variant="warn">Disconnected</Badge>;
  }, [connected, error]);

  return (
    <div className="min-h-screen w-full flex flex-col">
      <header className="border-b border-border px-6 py-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold tracking-tight">
            EdgeVox <span className="text-muted-foreground font-normal">/ web</span>
          </h1>
          {headerBadge}
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono">
          {info && (
            <>
              <Badge variant="outline">{info.language}</Badge>
              <Badge variant="outline">{info.voice}</Badge>
              <span className="hidden sm:inline">{info.sessionId}</span>
            </>
          )}
        </div>
      </header>

      <main className="flex-1 px-6 py-6">
        <Card className="h-[calc(100vh-220px)] min-h-[420px] overflow-hidden">
          <CardContent className="h-full p-0">
            <ConversationView messages={messages} />
          </CardContent>
        </Card>
      </main>

      <footer className="border-t border-border px-6 py-4 flex flex-wrap items-center gap-4 justify-between">
        <div className="flex items-center gap-3">
          <StatusIndicator state={state} />
          <MicMeter level={level} />
        </div>

        <MetricsBar metrics={metrics} />

        <div className="flex items-center gap-2">
          {!recording ? (
            <Button onClick={startMic} disabled={!connected}>
              <Mic className="size-4" />
              Start
            </Button>
          ) : (
            <Button variant="secondary" onClick={stopMic}>
              <MicOff className="size-4" />
              Stop
            </Button>
          )}
          <Button variant="outline" onClick={interrupt} disabled={!connected}>
            <Square className="size-4" />
            Interrupt
          </Button>
          <Button variant="ghost" onClick={reset} disabled={!connected}>
            <RotateCcw className="size-4" />
            Reset
          </Button>
        </div>
      </footer>
    </div>
  );
}
