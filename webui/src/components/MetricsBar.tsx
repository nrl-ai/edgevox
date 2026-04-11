import { useRef } from "react";

export interface Metrics {
  stt: number;
  llm: number;
  ttft: number;
  tts: number;
  total: number;
  audio_duration: number;
}

const SPARK = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"];
const HIST_LEN = 20;

function fmt(n: number | undefined) {
  return typeof n === "number" ? `${n.toFixed(2)}s` : " —  ";
}

function ttfsColor(n: number | undefined): string {
  if (n === undefined) return "text-muted-foreground";
  if (n < 1.0) return "text-neon-green";
  if (n < 2.0) return "text-neon-orange";
  return "text-neon-red";
}

export function MetricsBar({ metrics }: { metrics: Metrics | null }) {
  const historyRef = useRef<number[]>([]);

  if (metrics?.ttft !== undefined) {
    historyRef.current.push(metrics.ttft);
    if (historyRef.current.length > HIST_LEN)
      historyRef.current = historyRef.current.slice(-HIST_LEN);
  }

  const hist = historyRef.current;
  const spark = hist
    .map((v) => SPARK[Math.round(Math.min(v / 5.0, 1.0) * (SPARK.length - 1))])
    .join("");
  const avg = hist.length > 0 ? hist.reduce((a, b) => a + b, 0) / hist.length : 0;

  const ttft = metrics?.ttft;

  return (
    <div className="font-mono text-xs leading-relaxed">
      <div className="flex flex-wrap items-center gap-x-2">
        <span className="text-neon-green font-bold">■ Latency</span>
        <span className="text-neon-cyan">STT {fmt(metrics?.stt)}</span>
        <span className="text-neon-purple">LLM {fmt(metrics?.llm)}</span>
        <span className="text-neon-blue">TTS {fmt(metrics?.tts)}</span>
        <span className="text-foreground">Total {fmt(metrics?.total)}</span>
        <span className="text-muted-foreground">TTFS</span>
        <span className={`font-bold ${ttfsColor(ttft)}`}>{fmt(ttft)}</span>
      </div>
      {hist.length > 0 && (
        <div className="flex items-center gap-1">
          <span className="text-neon-orange">{spark}</span>
          <span className="text-muted-foreground">avg {avg.toFixed(2)}s</span>
        </div>
      )}
    </div>
  );
}
