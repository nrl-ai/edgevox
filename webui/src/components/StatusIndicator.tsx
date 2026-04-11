import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";

export type BotState = "idle" | "listening" | "transcribing" | "thinking" | "speaking";

const STATE_TEMPLATE: Record<BotState, string> = {
  idle: "{anim} Idle",
  listening: "{anim} Listening — speak now",
  transcribing: "{anim} Transcribing...",
  thinking: "{anim} Thinking...",
  speaking: "{anim} Speaking...",
};

const STATE_ANIMS: Record<BotState, string> = {
  idle: "○",
  listening: "●○",
  transcribing: "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
  thinking: "⣾⣽⣻⢿⡿⣟⣯⣷",
  speaking: "▁▂▃▄▅▆▇█▇▅▃",
};

const STATE_COLOR: Record<BotState, string> = {
  idle: "text-muted-foreground",
  listening: "text-neon-green",
  transcribing: "text-neon-cyan",
  thinking: "text-neon-purple",
  speaking: "text-neon-blue",
};

export function StatusIndicator({ state }: { state: BotState }) {
  const [frame, setFrame] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    timerRef.current = setInterval(() => setFrame((f) => f + 1), 150);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const anims = STATE_ANIMS[state];
  const char = anims[frame % anims.length];
  const label = STATE_TEMPLATE[state].replace("{anim}", char);

  return (
    <span className={cn("font-mono text-sm font-bold", STATE_COLOR[state])}>
      {label}
    </span>
  );
}
