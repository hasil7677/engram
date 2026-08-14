# Engram chat UI

A small React + TypeScript + Vite front end for the Engram memory service. It talks to
`POST /v1/chat` and — the reason it exists — shows the memories Engram retrieved for each
reply *next to* that reply. Watching the recall happen is a much faster way to tell whether
memory is actually working than inferring it from `curl` output.

## Run it

The API has to be up first (from the repo root):

```bash
docker compose up --build
```

Then, in `ui/`:

```bash
npm install
npm run dev
```

Vite serves on <http://localhost:5173>. That origin is not incidental — `app/main.py` allows
CORS for exactly `http://localhost:5173` and `http://127.0.0.1:5173`, so if you serve the UI
from a different port the browser will block the requests.

Fill in the connection bar at the top: **base URL** (defaults to `http://localhost:8000`),
**API key**, and **user ID** (defaults to `user_1`). The API key is a tenant key you've
registered:

```bash
docker compose exec api python scripts/register_tenant.py mykey123 tenant_demo
```

Settings persist to `localStorage` under `engram-ui-config`, and the bar polls `/health` to
show whether the service is reachable. The composer stays disabled until all three fields are
filled.

## What's in here

| File | Role |
|---|---|
| `src/App.tsx` | State root — config, message list, retrieved memories, request phase |
| `src/api.ts` | SSE client for `/v1/chat` and the `/health` check |
| `src/components/ConnectionBar.tsx` | Base URL / API key / user ID, plus health indicator |
| `src/components/ChatPane.tsx` | Transcript and composer |
| `src/components/MessageBubble.tsx` | One message, including the streaming and error states |
| `src/components/MemoryPanel.tsx` | Retrieved memories with their relevance scores |

## Two things worth knowing

**The SSE client is hand-rolled on `fetch`, not `EventSource`.** `EventSource` can't send
custom headers or a POST body, and Engram's auth is `X-API-Key` + `X-User-Id` headers on a
POST — so `api.ts` reads `response.body` as a stream and splits events on `\n\n` itself. The
stream carries four event types: `memories` (sent first, before generation starts), `delta`,
`done`, and `error`.

**The memory panel shows the score breakdown, not just the text.** Each memory renders its
`final_score` as a bar, with the `semantic` / `temporal` / `frequency` components in the
hover title. Those three are what Engram's ranking actually combines, so when a reply uses
the wrong memory this is where you see why. The panel also tracks the request phase —
`recalling` → `thinking` → `streaming` — which makes it obvious whether latency is coming
from retrieval or from generation.

## Build

```bash
npm run build     # tsc -b && vite build, output to dist/
npm run lint      # oxlint
npm run preview   # serve the built dist/
```

Note that `npm run preview` serves on a different port than the dev server, which the API's
CORS allowlist does not include — the built bundle is checked in under `dist/`, but for
against-a-live-API use, `npm run dev` is the supported path.
