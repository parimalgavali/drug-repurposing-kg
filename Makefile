# RepurposeKG. Local Neo4j (Debian package): make kg NEO4J_ADMIN="sudo -u neo4j neo4j-admin"
# Docker: make kg NEO4J_ADMIN="docker compose run --rm neo4j neo4j-admin" IMPORT_PREFIX=/import
NEO4J_ADMIN ?= neo4j-admin
IMPORT_PREFIX ?=
PREFIX_ARG = $(if $(IMPORT_PREFIX),--prefix $(IMPORT_PREFIX),)

.PHONY: pages-data ml-blend labels-de labels-de-fetch rdf data kg kg-import kg-schema kg-fixture ml-baselines ml-gnn validate validate-hetionet gnnexplainer api openapi lint test

data:
	uv run python scripts/fetch_data.py all
	uv run python scripts/prepare.py

# Neo4j must be stopped for the offline bulk import; start it again before kg-schema.
kg-import:
	uv run --extra graph python -m kg.build_import --run --admin "$(NEO4J_ADMIN)" $(PREFIX_ARG)

kg-schema:
	uv run --extra graph python -m kg.schema

kg: kg-import kg-schema

# Load the committed mini fixture into an empty, running database (no PrimeKG download needed).
kg-fixture:
	uv run --extra graph python -m kg.load
	uv run --extra graph python -m kg.schema

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

# Splits + all task-3 models on both splits (KGE: 1 seed, ~10 min per model per split on CPU).
ml-baselines:
	uv run --extra ml python -m ml.run --models popularity,shared_neighbour,similar_disease --split all
	uv run --extra ml python -m ml.run --models rotate,complex --split all --seeds 0

# Task 4: text features, GNN (+ ablation) on both splits, then predictions into Neo4j.
ml-gnn:
	uv run --extra ml python -m ml.text_features
	uv run --extra ml python -m ml.run --models gnn,gnn_notext --split all
	uv run --extra ml --extra graph python -m ml.write_predictions --model gnn_notext --split random

# Task 12: GNN + similar-disease blend from the saved task-4 checkpoints (α picked on validation).
ml-blend:
	uv run --extra ml python -m ml.run --models gnn_blend --split all

# Task 5: ClinicalTrials.gov check of held-out-disease predictions (rate-limited, cached).
validate:
	uv run --extra ml --extra graph python -m validation.run

gnnexplainer:
	uv run --extra ml --extra graph python -m explain.gnnexplainer

validate-hetionet:
	uv run --extra ml --extra graph python -m validation.hetionet fetch
	uv run --extra ml --extra graph python -m validation.hetionet

# Task 7: the API (LLM_PROVIDER=anthropic for live answers; default replays recordings).
api:
	uv run --extra graph --extra llm --extra api --extra rdf uvicorn api.main:app --reload --port 8000

openapi:
	uv run --extra graph --extra llm --extra api --extra rdf python -m api.openapi

labels-de-fetch:
	uv run --extra data --extra graph python -m kg.labels_de fetch

labels-de:
	uv run --extra data --extra graph python -m kg.labels_de apply

rdf:
	uv run --extra graph --extra rdf python -m semweb.export --disease $${DISEASE:-33675}

# Upload the static snapshot exported by scripts/export_static.py for .github/workflows/pages.yml.
pages-data:
	tar -czf site-data.tar.gz -C site-data .
	gh release view static-snapshot >/dev/null 2>&1 || gh release create static-snapshot \
		--title "Static snapshot data for GitHub Pages" \
		--notes "API responses exported from the full PrimeKG graph by scripts/export_static.py. Derived from PrimeKG (DrugBank parts CC BY-NC): non-commercial use only."
	gh release upload static-snapshot site-data.tar.gz --clobber
