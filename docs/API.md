# API

FastAPI service in `api/`. Everything is read-only: graph queries go through `KGClient.read`, which runs inside read-only Neo4j transactions. Research demo, not medical advice.

```bash
docker compose up --build          # neo4j + api on :8000 (load the graph first, see GRAPH.md)
make api                           # or locally, with Neo4j already running
make openapi                       # regenerate docs/openapi.json (a test fails if it is stale)
```

Settings come from the environment or `.env`: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `LLM_PROVIDER` (`anthropic` | `ollama` | `recorded`, default `recorded`), `CORS_ORIGINS`. If the LLM provider cannot start (for example, no `ANTHROPIC_API_KEY`), every endpoint except `/ask` still works.

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/health` | `{status, neo4j, llm}`; 503 if Neo4j is unreachable |
| GET | `/api/search?q=&type=&limit=10` | full-text + rapidfuzz hits `{id, name, label, confidence}`; `type` is a node label |
| GET | `/api/diseases/{id}/candidates?limit=20` | `{disease, candidates, known}`: `PREDICTED_INDICATION` by rank with score and trial counts, plus known indications/off-label uses |
| GET | `/api/pairs/{drug_id}/{disease_id}/explain` | task 5 evidence: top metapaths by DWPC, example paths, Cytoscape JSON |
| POST | `/api/ask` `{question}` | Server-Sent Events, see below |

Unknown ids, or an id with the wrong type (a drug id where a disease id is expected), return 404.

## `/ask` events

`entities` → `cypher` → `rows` → `token`… → `subgraph` → `done`

- `entities`: linked mentions `[{slot, mention, id, name, label, confidence}]`
- `cypher`: `{queries, attempts}`. These are the templates or the guarded text2cypher query that ran.
- `rows`: `{rows, total}`, capped at 50 rows
- `token`: `{text}`: chunks of the grounded answer, with node ids cited as `[id]`
- `subgraph`: Cytoscape JSON. For pair questions it shows the evidence paths; otherwise the linked entity plus the returned nodes.
- `done`: `{intent, clarification}`
- `error`: `{message, detail}`. This ends the stream early. It is sent, for example, when every text2cypher attempt is rejected by the guard.

An ambiguous mention sends `entities`, then one `token` holding the clarifying question, then `done` with `clarification: true`.

## Example (full graph, `LLM_PROVIDER=anthropic`)

```bash
B=http://localhost:8000/api
curl "$B/search?q=asthma&type=Disease&limit=2"
# [{"id":33675,"name":"asthma","label":"Disease","confidence":1.0},
#  {"id":39773,"name":"allergic asthma","label":"Disease","confidence":0.9}]
curl "$B/diseases/33675/candidates?limit=3"
# candidates: Zileuton (rank 1), Fluticasone propionate (2), Salmeterol (3); 45 known drugs
curl "$B/pairs/15479/33675/explain"          # Salbutamol → asthma
curl -N -X POST "$B/ask" -H 'content-type: application/json' \
     -d '{"question": "Which genes does Ibuprofen target?"}'
# event: entities … event: cypher … event: rows (10) … 25 × event: token
# "Ibuprofen [14287] targets these 10 genes: BCL2 [2429], … PTGS1 [9031], PTGS2 [4425] …"
# event: subgraph … event: done
```

## German names

`Node`, `SearchHit` and `PathNode` carry `name_de`, and `Candidate` and `KnownDrug` carry `drug_de`. Each is the Wikidata German name, or `null` when there is none. Cytoscape nodes get `data.label_de` when available.
- **Search:** still matches the PrimeKG (English) names.
- **`POST /api/ask {"question": …, "lang": "de"}`:** the prompt asks for a German answer and passes the German names of the subgraph nodes. Every `[id]` citation is kept, and the disclaimer is German.
- **`lang` omitted or `"en"`:** the prompt is unchanged, so the recorded LLM responses stay valid.

## Semantic web

| Endpoint | Returns |
| --- | --- |
| `GET /api/diseases/{id}/rdf?k=20` | Turtle export of the disease-centred subgraph |
| `GET /api/sparql?query=…`, `POST /api/sparql {"query": …}` | `{vars, rows, truncated}` for `SELECT`, `{boolean}` for `ASK`; 400 for anything else, 503 if no export is loaded |

`/api/sparql` serves `RDF_PATH` (default `data/rdf/export.ttl`, written by `make rdf`). See [SEMANTIC_WEB.md](SEMANTIC_WEB.md).
