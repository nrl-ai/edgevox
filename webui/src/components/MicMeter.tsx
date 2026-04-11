import { useRef } from "react";

const SPARK = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"];
const HISTORY_LEN = 22;

export function MicMeter({ level }: { level: number }) {
  const historyRef = useRef<number[]>([]);

  const clamped = Math.min(1, Math.max(0, level));
  historyRef.current.push(clamped);
  if (historyRef.current.length > HISTORY_LEN)
    historyRef.current = historyRef.current.slice(-HISTORY_LEN);

  const spark = historyRef.current
    .map((v) => SPARK[Math.round(v * (SPARK.length - 1))])
    .join("");
  const pad = SPARK[0].repeat(HISTORY_LEN - spark.length);

  const color =
    clamped > 0.6
      ? "text-neon-red"
      : clamped > 0.3
        ? "text-neon-orange"
        : "text-neon-green";

  return (
    <span className="font-mono text-sm">
      <span className="text-neon-cyan font-bold">● Audio </span>
      <span className={color}>{pad}{spark}</span>
      <span className="text-muted-foreground"> {Math.round(clamped * 100)}%</span>
    </span>
  );
}
