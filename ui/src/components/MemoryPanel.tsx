import type { ScoredMemory } from "../types";

interface Props {
  memories: ScoredMemory[];
  phase: "idle" | "recalling" | "thinking" | "streaming";
}

function relativeTime(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.round(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export default function MemoryPanel({ memories, phase }: Props) {
  return (
    <aside className="memory-panel">
      <div className="memory-panel-header">
        <h2>Recalled memories</h2>
        {phase === "recalling" && <span className="phase-label pulsing">searching…</span>}
      </div>

      {memories.length === 0 && phase !== "recalling" && (
        <div className="memory-empty">Nothing recalled yet — send a message to see what Engram remembers.</div>
      )}

      {phase === "recalling" && (
        <div className="memory-skeletons">
          {[0, 1, 2].map((i) => (
            <div key={i} className="memory-card skeleton" style={{ animationDelay: `${i * 80}ms` }} />
          ))}
        </div>
      )}

      <div className="memory-list">
        {memories.map((m, i) => (
          <div key={m.memory_id} className="memory-card" style={{ animationDelay: `${i * 60}ms` }}>
            <div className="memory-card-top">
              <span className={`source-badge source-${m.source}`}>
                {m.source === "graph_expansion" ? "graph" : "vector"}
              </span>
              <span className="memory-time">{relativeTime(m.timestamp)}</span>
            </div>
            <p className="memory-text">{m.text}</p>
            <div className="score-row" title={`semantic ${m.semantic_score.toFixed(2)} · temporal ${m.temporal_score.toFixed(2)} · frequency ${m.frequency_score.toFixed(2)}`}>
              <div className="score-bar">
                <div className="score-bar-fill" style={{ width: `${Math.round(m.final_score * 100)}%` }} />
              </div>
              <span className="score-value">{m.final_score.toFixed(2)}</span>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
