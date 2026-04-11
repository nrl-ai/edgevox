import { cn } from "@/lib/utils";

export function MicMeter({ level }: { level: number }) {
  const segments = 16;
  const active = Math.round(Math.min(1, Math.max(0, level)) * segments);
  return (
    <div className="flex items-center gap-[2px]">
      {Array.from({ length: segments }).map((_, i) => {
        const lit = i < active;
        const hue =
          i < segments * 0.6
            ? "bg-emerald-500"
            : i < segments * 0.85
              ? "bg-amber-400"
              : "bg-red-500";
        return (
          <span
            key={i}
            className={cn(
              "h-3 w-1 rounded-sm",
              lit ? hue : "bg-muted"
            )}
          />
        );
      })}
    </div>
  );
}
