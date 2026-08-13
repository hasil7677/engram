import { useEffect, useState } from "react";
import type { EngramConfig } from "../types";
import { checkHealth } from "../api";

interface Props {
  config: EngramConfig;
  onChange: (config: EngramConfig) => void;
}

export default function ConnectionBar({ config, onChange }: Props) {
  const [open, setOpen] = useState(!config.apiKey);
  const [status, setStatus] = useState<"checking" | "up" | "down">("checking");

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      const ok = await checkHealth(config.baseUrl);
      if (!cancelled) setStatus(ok ? "up" : "down");
    };
    poll();
    const id = setInterval(poll, 8000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [config.baseUrl]);

  return (
    <div className="connection-bar">
      <div className="brand">
        <span className="brand-mark">◈</span> Engram
      </div>

      <button className={`status-pill status-${status}`} onClick={() => setOpen((o) => !o)} title="Connection settings">
        <span className="status-dot" />
        {status === "up" ? "Connected" : status === "down" ? "Offline" : "Checking…"}
      </button>

      {open && (
        <div className="connection-panel">
          <label>
            Base URL
            <input
              value={config.baseUrl}
              onChange={(e) => onChange({ ...config, baseUrl: e.target.value })}
              placeholder="http://localhost:8000"
            />
          </label>
          <label>
            API Key
            <input
              value={config.apiKey}
              onChange={(e) => onChange({ ...config, apiKey: e.target.value })}
              placeholder="X-API-Key"
            />
          </label>
          <label>
            User ID
            <input
              value={config.userId}
              onChange={(e) => onChange({ ...config, userId: e.target.value })}
              placeholder="X-User-Id"
            />
          </label>
          <button className="connection-panel-close" onClick={() => setOpen(false)}>
            Done
          </button>
        </div>
      )}
    </div>
  );
}
