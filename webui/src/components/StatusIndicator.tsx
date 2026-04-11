import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

export type BotState = "idle" | "listening" | "transcribing" | "thinking" | "speaking";

const STATE_LABEL: Record<BotState, string> = {
  idle: "Idle",
  listening: "Listening",
  transcribing: "Transcribing",
  thinking: "Thinking",
  speaking: "Speaking",
};

const STATE_COLOR: Record<BotState, string> = {
  idle: "bg-muted-foreground",
  listening: "bg-emerald-500",
  transcribing: "bg-sky-500",
  thinking: "bg-violet-500",
  speaking: "bg-amber-500",
};

export function StatusIndicator({ state }: { state: BotState }) {
  return (
    <Badge variant="outline" className="gap-2 px-3 py-1.5">
      <span
        className={cn(
          "inline-block h-2.5 w-2.5 rounded-full",
          STATE_COLOR[state],
          state !== "idle" && "animate-pulse-soft"
        )}
      />
      <span>{STATE_LABEL[state]}</span>
    </Badge>
  );
}
