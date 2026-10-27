# Roadmap

One task = one GitHub issue = one branch `task-N-slug` = one PR = one one-page spec in `docs/specs/`. A task is done when its "Done when" column is true and the PR is merged. Order: 0 → 8; 9 only after 8.

| # | Task | Spec | Issue | Done when |
| --- | --- | --- | --- | --- |
| 0 | Planning & docs | [task-0](specs/task-0-docs.md) | [#1](https://github.com/parimalgavali/drug-repurposing-kg/issues/1) | planning doc, DECISIONS, DATA, ARCHITECTURE, ROADMAP, specs merged |
| 1 | Data layer | [task-1](specs/task-1-data-layer.md) | [#2](https://github.com/parimalgavali/drug-repurposing-kg/issues/2) | `uv run python scripts/fetch_data.py all` downloads + verifies PrimeKG; parquet conversion; committed fixture subgraph; data card; CI (ruff + pytest) green |
| 2 | Neo4j graph | [task-2](specs/task-2-neo4j.md) | [#3](https://github.com/parimalgavali/drug-repurposing-kg/issues/3) | `make kg` imports full PrimeKG into Neo4j in < 15 min; constraints + full-text index; 10 documented example Cypher queries with tests on the fixture |
| 3 | Splits & baselines | [task-3](specs/task-3-splits-baselines.md) | [#4](https://github.com/parimalgavali/drug-repurposing-kg/issues/4) | disease-held-out + random splits saved; popularity, shared-neighbour and PyKEEN baselines evaluated; first `model_card.md` |
| 4 | Heterogeneous GNN | [task-4](specs/task-4-gnn.md) | [#5](https://github.com/parimalgavali/drug-repurposing-kg/issues/5) | PyG model trained on CPU; compared with task-3 baselines on both splits; predictions written to Neo4j as `PREDICTED_INDICATION` |
| 5 | Explanations & trial check | [task-5](specs/task-5-explain-validate.md) | [#6](https://github.com/parimalgavali/drug-repurposing-kg/issues/6) | top metapaths per pair via Cypher; ClinicalTrials.gov hit rate of top-k predictions vs random/popularity baselines in the model card |
| 6 | GraphRAG | [task-6](specs/task-6-graphrag.md) | [#7](https://github.com/parimalgavali/drug-repurposing-kg/issues/7) | entity linker, intent router, templates, guarded text-to-Cypher; ≥ 80% execution accuracy on 40 gold questions; offline tests on recorded LLM responses |
| 7 | API | [task-7](specs/task-7-api.md) | [#8](https://github.com/parimalgavali/drug-repurposing-kg/issues/8) | FastAPI endpoints + SSE `/api/ask`; `docker compose up` → full flow via curl; CI green |
| 8 | Frontend & release | [task-8](specs/task-8-frontend-release.md) | [#9](https://github.com/parimalgavali/drug-repurposing-kg/issues/9) | one-screen Vue app with candidates, graph view, Q&A; Playwright smoke in CI; demo GIF + resume bullets in README |
| 9 | Stretch | [task-9](specs/task-9-stretch.md) | [#10](https://github.com/parimalgavali/drug-repurposing-kg/issues/10) | RDF/SPARQL export (semantic-web view) + SHACL; German labels via Wikidata; Hetionet cross-check |

Rough effort: tasks 0–8 ≈ 5–7 focused sessions.

## Secrets needed and when

| Secret | First needed | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | task 6 (recording fixtures), task 7+ at runtime | LLM provider. Ollama works without it |

No other accounts: PrimeKG and ClinicalTrials.gov need no login.

## Out of scope

Auth, multi-user, cloud deployment, GPU training, building a KG from literature, clinical use. See `DECISIONS.md` #13.
