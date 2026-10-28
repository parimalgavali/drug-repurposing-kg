# RepurposeKG web

Vue 3 + TypeScript + Vite, Cytoscape.js for graphs. Types in `src/api/schema.ts` are generated from `docs/openapi.json`.

```bash
npm ci
npm run dev          # :5173, proxies /api to $API_URL (default http://localhost:8000)
npm run gen:api      # regenerate src/api/schema.ts after `make openapi`
npm run typecheck && npm test
npm run e2e          # Playwright smoke test; needs the API on :8000, see below
```

The e2e test runs against the fixture graph and replays saved LLM responses:

```bash
PYTHONPATH=. uv run python scripts/demo_fixture.py   # empty Neo4j → fixture + predictions
LLM_PROVIDER=recorded LLM_RECORDINGS=web/e2e/recordings make api
```

Re-record with `LLM_PROVIDER=anthropic LLM_RECORD=1` after changing the question or the prompts.
