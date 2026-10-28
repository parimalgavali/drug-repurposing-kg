# RepurposeKG. Local Neo4j (Debian package): make kg NEO4J_ADMIN="sudo -u neo4j neo4j-admin"
# Docker: make kg NEO4J_ADMIN="docker compose run --rm neo4j neo4j-admin" IMPORT_PREFIX=/import
NEO4J_ADMIN ?= neo4j-admin
IMPORT_PREFIX ?=
PREFIX_ARG = $(if $(IMPORT_PREFIX),--prefix $(IMPORT_PREFIX),)

.PHONY: data kg kg-import kg-schema kg-fixture ml-baselines ml-gnn lint test

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
