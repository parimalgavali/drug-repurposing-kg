# Task 2: Neo4j graph

**Goal:** the full PrimeKG lives in Neo4j with a clean schema, and anyone can reproduce that with one command.

**Deliverables**
- `docker-compose.yml` with `neo4j:5-community` (APOC core only; heap/pagecache sized for 4M edges). Docs also describe a non-Docker path (Debian package) for machines where Docker Hub is rate-limited.
- `kg/build_import.py`: parquet → per-label node CSVs and per-type relationship CSVs in `neo4j-admin` header format → `neo4j-admin database import full --overwrite-destination`.
- `kg/schema.cypher`: uniqueness constraint on `id` per label; full-text index `entityNames` over `name` for all labels; range index on `Disease.name`, `Drug.name`.
- `kg/client.py`: thin driver wrapper with `read(query, params, timeout)` that always uses a read-only transaction.
- `kg/queries/`: 10 documented example queries as `.cypher` files, e.g. drugs indicated for a disease, shared targets of two drugs, genes linking a drug and a disease within 3 hops, diseases sharing phenotypes.
- `Makefile` targets `kg-import`, `kg-schema`, `kg` (both).
- `docs/GRAPH.md`: label/relationship table with counts, and a Neo4j Browser screenshot.

**Tests:** import builder on the fixture (header format, one direction per edge); integration tests (skipped without `NEO4J_URI`) load the fixture via `UNWIND` and run all 10 example queries; CI adds a Neo4j service container.

**Done when:** `make kg` imports the full graph in < 15 min on the dev VM; node/relationship counts match `docs/DATA.md` minus the deduplicated reverse edges; all example queries return within 2 s.

**Risks:** Docker Hub rate limits (use the Debian package locally, GHCR or a cached image in CI); memory (set `dbms.memory.*` explicitly).

**Outcome:** bulk import of the full graph takes ~7 s (`neo4j-admin`, 129,375 nodes / 4,050,249 relationships); all 10 example queries run in < 20 ms warm. Local dev uses the Debian package (`stable 5` component) because Docker Hub rate-limited anonymous pulls. `kg-fixture` (UNWIND loader) added so CI and contributors can test without the 1 GB download. See `docs/GRAPH.md`.
