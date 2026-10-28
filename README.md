# RepurposeKG

**Drug-repurposing hypotheses from a biomedical knowledge graph.** PrimeKG (129k nodes, 4M relationships) is stored in Neo4j. A graph neural network predicts missing drug → disease links, and the evidence is shown as graph paths and ClinicalTrials.gov counts. An LLM answers questions about the graph with validated Cypher.

> Research demo, **not medical advice**. Uses PrimeKG, which includes DrugBank-derived content licensed CC BY-NC 4.0, so this project is **non-commercial**.

## Status

Done so far: the data layer and Neo4j graph (tasks 1–2), baselines and a heterogeneous GNN with predictions stored in Neo4j (tasks 3–4), and evidence paths plus a ClinicalTrials.gov check (task 5). GraphRAG is next. See the [roadmap](docs/ROADMAP.md) and the [model card](ml/model_card.md).

```bash
uv sync --extra data --extra graph
make data                 # PrimeKG → data/processed/*.parquet
make kg NEO4J_ADMIN="sudo -u neo4j neo4j-admin"   # see docs/GRAPH.md
make ml-baselines ml-gnn   # task 3–4: models, model card, PREDICTED_INDICATION
make validate              # task 5: CT.gov trial check (~1 h, cached)
uv run --extra graph python -m explain.paths --drug Salbutamol --disease asthma
make lint test
```

## Docs

- [Planning & investigation](docs/planning/2026-10-28-direction.md): goal, user loop, verified data facts, risks
- [Decisions](docs/DECISIONS.md): ADRs
- [Data](docs/DATA.md): PrimeKG and ClinicalTrials.gov, licenses, counts, checksums
- [Architecture](docs/ARCHITECTURE.md): components, graph model, GraphRAG flow, evaluation
- [Graph](docs/GRAPH.md): Neo4j labels/relationship counts, build steps, example queries
- [Roadmap](docs/ROADMAP.md) and [task specs](docs/specs/)

## Stack

Python 3.12 (uv) · Neo4j 5 · PyTorch Geometric · PyKEEN · FastAPI · Anthropic Claude / Ollama · Vue 3 + TypeScript + Cytoscape.js

## License

Code: MIT (see `LICENSE`). Data is not redistributed; see [docs/DATA.md](docs/DATA.md) for the terms of each source.
