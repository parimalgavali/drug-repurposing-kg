# Task 7: API

**Goal:** one FastAPI service exposes search, candidates, explanations and Q&A.

**Deliverables**
- `api/main.py` with lifespan-managed Neo4j driver and LLM provider; settings via pydantic-settings.
- Endpoints: `GET /api/search?q=&type=` (full-text) · `GET /api/diseases/{id}/candidates?limit=20` (from `PREDICTED_INDICATION`, with score, rank, trial counts, known indications for context) · `GET /api/pairs/{drug_id}/{disease_id}/explain` (task 5 JSON) · `POST /api/ask` (SSE events: `entities`, `cypher`, `rows`, `token`, `subgraph`, `done`, `error`) · `GET /api/health`.
- `Dockerfile` for the API; `docker-compose.yml` brings up neo4j + api (+ web after task 8).
- OpenAPI schema committed to `docs/openapi.json` for the frontend types.

**Tests:** httpx `TestClient` tests against the fixture graph (Neo4j service in CI) and `RecordedProvider`; SSE event order; 404 for unknown ids; `/ask` with an injected malicious question returns an `error` event, not a write.

**Done when:** `docker compose up` → `curl` the full flow (search → candidates → explain → ask) works; CI green.
