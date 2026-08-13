import { useCallback, useState } from "react";
import "./App.css";
import ConnectionBar from "./components/ConnectionBar";
import ChatPane from "./components/ChatPane";
import MemoryPanel from "./components/MemoryPanel";
import { streamChat } from "./api";
import type { ChatMessage, EngramConfig, ScoredMemory } from "./types";

const CONFIG_KEY = "engram-ui-config";

function loadConfig(): EngramConfig {
  try {
    const raw = localStorage.getItem(CONFIG_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore malformed storage
  }
  return { baseUrl: "http://localhost:8000", apiKey: "", userId: "user_1" };
}

type MemoryPhase = "idle" | "recalling" | "thinking" | "streaming";

export default function App() {
  const [config, setConfig] = useState<EngramConfig>(loadConfig);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [memories, setMemories] = useState<ScoredMemory[]>([]);
  const [phase, setPhase] = useState<MemoryPhase>("idle");
  const [busy, setBusy] = useState(false);

  const updateConfig = useCallback((next: EngramConfig) => {
    setConfig(next);
    localStorage.setItem(CONFIG_KEY, JSON.stringify(next));
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", text };
      const assistantId = crypto.randomUUID();
      const assistantMsg: ChatMessage = { id: assistantId, role: "assistant", text: "", streaming: true };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setBusy(true);
      setPhase("recalling");
      setMemories([]);

      const patchAssistant = (patch: Partial<ChatMessage>) => {
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)));
      };

      try {
        for await (const event of streamChat(config, text)) {
          if (event.type === "memories") {
            setMemories(event.memories);
            setPhase("thinking");
          } else if (event.type === "delta") {
            setPhase("streaming");
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: m.text + event.text } : m))
            );
          } else if (event.type === "error") {
            patchAssistant({ text: event.message, streaming: false, error: true });
          } else if (event.type === "done") {
            patchAssistant({ streaming: false });
          }
        }
      } catch (err) {
        patchAssistant({
          text: err instanceof Error ? err.message : "Something went wrong talking to Engram.",
          streaming: false,
          error: true,
        });
      } finally {
        setBusy(false);
        setPhase("idle");
      }
    },
    [config]
  );

  const canChat = Boolean(config.apiKey && config.userId && config.baseUrl);

  return (
    <div className="app-shell">
      <ConnectionBar config={config} onChange={updateConfig} />
      <div className="main-layout">
        <ChatPane messages={messages} onSend={handleSend} disabled={!canChat || busy} />
        <MemoryPanel memories={memories} phase={phase} />
      </div>
    </div>
  );
}
