import { Badge } from "@/components/ui/badge";

export interface Metrics {
  stt: number;
  llm: number;
  ttft: number;
  tts: number;
  total: number;
  audio_duration: number;
}

function fmt(n: number | undefined) {
  return typeof n === "number" ? `${n.toFixed(2)}s` : "—";
}

export function MetricsBar({ metrics }: { metrics: Metrics | null }) {
  return (
    <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
      <Badge variant="secondary">STT {fmt(metrics?.stt)}</Badge>
      <Badge variant="secondary">TTFT {fmt(metrics?.ttft)}</Badge>
      <Badge variant="secondary">LLM {fmt(metrics?.llm)}</Badge>
      <Badge variant="secondary">TTS {fmt(metrics?.tts)}</Badge>
      <Badge variant="default">Total {fmt(metrics?.total)}</Badge>
    </div>
  );
}
