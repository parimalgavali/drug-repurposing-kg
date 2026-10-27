# Task 8: Frontend & release

**Goal:** one screen that a reviewer understands in 30 seconds and can demo in 2 minutes.

**Deliverables**
- `web/` (Vue 3 + TypeScript + Vite, types generated from `docs/openapi.json`):
  - Disease search with autocomplete → ranked candidate list (score bar, "in N trials" badge, known indications shown greyed for context).
  - Clicking a candidate shows the Cytoscape.js subgraph of the top explanation paths, coloured by node type, with a legend.
  - Question box streams the answer; "Show Cypher" expands the query that ran; entities in the answer are clickable in the graph.
  - A permanent "Research demo, not medical advice" banner.
- Playwright smoke test (search → candidate → graph → ask, with recorded LLM) in CI.
- README: 60-second demo GIF, architecture image, model-card summary table, "run it in one command", resume bullets.

**Done when:** a stranger can `docker compose up` and complete the demo flow; Playwright green in CI.
