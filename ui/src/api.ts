import type { EngramConfig, ScoredMemory } from "./types";

export type ChatEvent =
  | { type: "memories"; memories: ScoredMemory[] }
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

/**
 * Raw fetch() + ReadableStream reading of the SSE response -- no EventSource,
 * since EventSource can't send the X-API-Key/X-User-Id headers or a POST body.
 */
export async function* streamChat(config: EngramConfig, text: string, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const resp = await fetch(`${config.baseUrl}/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": config.apiKey,
      "X-User-Id": config.userId,
    },
    body: JSON.stringify({ text }),
    signal,
  });

  if (!resp.ok || !resp.body) {
    const detail = await resp.text().catch(() => "");
    throw new Error(`Chat request failed (${resp.status}): ${detail || resp.statusText}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sepIndex: number;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);
      const event = parseSseEvent(rawEvent);
      if (event) yield event;
    }
  }
}

function parseSseEvent(raw: string): ChatEvent | null {
  let eventType = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;

  const data = JSON.parse(dataLines.join("\n"));
  switch (eventType) {
    case "memories":
      return { type: "memories", memories: data.memories };
    case "delta":
      return { type: "delta", text: data.text };
    case "done":
      return { type: "done" };
    case "error":
      return { type: "error", message: data.message };
    default:
      return null;
  }
}

export async function checkHealth(baseUrl: string): Promise<boolean> {
  try {
    const resp = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(3000) });
    return resp.ok;
  } catch {
    return false;
  }
}
