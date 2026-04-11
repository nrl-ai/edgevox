import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";

export interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  pending?: boolean;
  timestamp?: string;
}

const LOGO = [
  " ███████ ██████   ██████  ███████ ██    ██  ██████  ██   ██",
  " ██      ██   ██ ██       ██      ██    ██ ██    ██  ██ ██",
  " █████   ██   ██ ██   ███ █████   ██    ██ ██    ██   ███",
  " ██      ██   ██ ██    ██ ██       ██  ██  ██    ██  ██ ██",
  " ███████ ██████   ██████  ███████   ████    ██████  ██   ██",
];

export function ConversationView({ messages }: { messages: Message[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col p-4 font-mono text-sm leading-relaxed">
        {/* ASCII splash */}
        <div className="mb-1">
          {LOGO.map((line, i) => (
            <div key={i} className="text-neon-green font-bold whitespace-pre">
              {line}
            </div>
          ))}
          <div className="text-neon-cyan">{"  Sub-second local voice AI  v0.1.0"}</div>
        </div>

        {messages.length === 0 && (
          <p className="text-muted-foreground py-4">
            Press <span className="text-neon-green font-bold">Start</span> and say something.
          </p>
        )}

        {messages.map((m) => (
          <div key={m.id}>
            {m.role === "user" && (
              <>
                {/* TUI-style separator */}
                <div className="text-[#1e3a2e] select-none whitespace-pre">
                  {"─".repeat(60)}
                </div>
                <div>
                  <span className="text-neon-green font-bold"> ▶ You </span>
                  <span className="text-muted-foreground"> {m.timestamp || ""} </span>
                </div>
                <div className="text-foreground pl-3 whitespace-pre-wrap">
                  {"   "}{m.text}
                </div>
              </>
            )}
            {m.role === "bot" && (
              <>
                <div>
                  <span className="text-neon-cyan font-bold"> ◀ Bot</span>
                </div>
                <div className={`text-foreground pl-3 whitespace-pre-wrap ${m.pending ? "opacity-70" : ""}`}>
                  {"   "}{m.text || (m.pending ? "… thinking" : "")}
                </div>
              </>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </ScrollArea>
  );
}
