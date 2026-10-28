# Architecture

Principle: one Python process, one graph database, one screen. Offline jobs (fetch, load, train, validate) are plain scripts beside the API.

## Components

```
┌──────────────────────────────────────────────────────────────────────┐
│ web/  (Vue 3 + Vite + TypeScript, one route)                         │
│   Left: disease search → ranked candidate drugs (score, trials badge)│
│   Right: Cytoscape.js subgraph of explanation paths                  │
│   Bottom: question box → streamed answer + Cypher shown + subgraph   │
└───────────────┬──────────────────────────────────────────────────────┘
                │ REST + SSE (/api)
┌───────────────▼──────────────────────────────────────────────────────┐
│ api/  (FastAPI, Python 3.12)                                         │
│   routers/   search, diseases/{id}/candidates, pairs/{d}/{x}/explain,│
│              ask (SSE), health                                       │
│ kg/          Neo4j driver wrapper, schema, read-only query runner    │
│ graphrag/    entity linker, intent router, Cypher templates,         │
│              text-to-Cypher + guards, answer synthesis               │
│ llm/         Provider protocol → AnthropicProvider | OllamaProvider  │
│              | RecordedProvider (tests)                              │
│ explain/     metapath enumeration + ranking for a (drug, disease)    │
└──────┬──────────────────────────────┬────────────────────────────────┘
       │ Bolt                         │ HTTPS
  Neo4j 5 Community               Anthropic API / local Ollama
  (:Drug, :Disease, :Gene, …,
   INDICATION, TARGET, …,
   PREDICTED_INDICATION{score,rank,model_version})
```

Offline, not part of the API runtime:

- `scripts/fetch_data.py`: downloads PrimeKG with checksum checks (task 1).
- `kg/build_import.py`: `kg.csv` → per-label node CSVs + per-type relationship CSVs → `neo4j-admin database import full`, then constraints and full-text index (task 2).
- `ml/`: `splits.py`, `baselines.py`, `kge.py` (PyKEEN), `gnn.py` (PyG), `evaluate.py`, `model_card.md`, `write_predictions.py` (tasks 3–4).
- `validation/ctgov.py`: ClinicalTrials.gov checks for the top-k predictions (task 5).
- `graphrag/eval/questions.jsonl` + `eval.py`: gold questions with expected results (task 6).

## Graph model in Neo4j

- One label per PrimeKG node type: `Drug, Disease, Gene, Phenotype, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Exposure`. Every node has `id` (PrimeKG `node_index`, unique), `source_id`, `source`, `name`.
- Relationship type = PrimeKG `relation` in upper snake case, with `drug_protein` split into `TARGET`/`ENZYME`/`TRANSPORTER`/`CARRIER` (33 types, e.g. `INDICATION`, `CONTRAINDICATION`, `OFF_LABEL_USE`, `DISEASE_PROTEIN`, `PROTEIN_PROTEIN`, `DRUG_EFFECT`, `DISEASE_DISEASE`). One direction per undirected edge, with a fixed type order (see `data/README.md`).
- `Drug` and `Disease` carry selected text features (description, mechanism, definition) for the UI and the LLM.
- Full-text index `entityNames` on `name` for all labels, plus `entityNamesDe` on `name_de` for drugs and diseases → entity linking in English or German.
- `PREDICTED_INDICATION` is written by `ml/write_predictions.py`, top-50 per disease.

## GraphRAG flow (`POST /api/ask`, SSE)

```
question
  → entity linker (full-text index + rapidfuzz re-rank, typed)
  → intent router (LLM, structured output): repurpose | explain_pair | neighbours | free_query
  → template Cypher (first three)   |   text-to-Cypher (free_query)
                                       → guard: allowlist labels/rels, add LIMIT,
                                         EXPLAIN, read-only tx, 5 s timeout, 1 retry
  → rows + paths → answer synthesis (LLM, cites node ids) → SSE: tokens, cypher, subgraph
```

Every response includes the Cypher that ran. The UI shows it next to the answer.

## Link prediction

- Positives: `INDICATION` edges. Splits: (a) **disease-held-out**: diseases grouped by PrimeKG's disease grouping, 20% of groups held out, all their indication edges removed from training; (b) random 80/10/10 on indication edges. Seeds fixed and splits saved to `data/processed/splits/`.
- Training graph: all relations except held-out indications. Drug–drug synergy and anatomy expression are excluded by default (ADR 6).
- Metrics: filtered ranking over all drugs per test disease (MRR, Hits@1/10/50) plus AUROC/AUPRC with 1:1 sampled negatives. Reported per split and per model in `ml/model_card.md`.

## Configuration

`.env` (never committed): `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `LLM_PROVIDER` (`anthropic|ollama|recorded`), `ANTHROPIC_API_KEY`, `OLLAMA_HOST`.

## Testing

- Unit tests run on the committed fixture subgraph, with no network or LLM.
- Neo4j integration tests are skipped unless `NEO4J_URI` is set. CI starts a Neo4j service container.
- LLM paths use `RecordedProvider` fixtures, re-recorded with `--record` when a key is present.
