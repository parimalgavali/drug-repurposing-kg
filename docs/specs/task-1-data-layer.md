# Task 1: Data layer

**Goal:** one command fetches and verifies PrimeKG, converts it to fast columnar files, and produces a small committed fixture that all later tests use.

**Deliverables**
- `pyproject.toml` (Python 3.12, uv; extras added per task: `data`, `kg`, `ml`, `llm`, `api`; dev: ruff line-length 100, pytest).
- `scripts/fetch_data.py` with sub-commands `primekg`, `features`, `all`. It streams downloads from Dataverse with resume and a progress bar, verifies sha256 against constants, and writes to `data/raw/primekg/`.
- `scripts/prepare.py`: `kg.csv` → `data/processed/edges.parquet`. It keeps one direction per undirected edge using a fixed type order (assert the count is 4,050,249), adds a `rel_type` column (upper snake case), and writes `nodes.parquet` plus `drug_features.parquet` / `disease_features.parquet`.
- `scripts/make_fixture.py`: deterministic (seeded) 2-hop sample around ~5 diseases with known indications (asthma, psoriasis, Parkinson disease, type 2 diabetes, rheumatoid arthritis), capped at ~2k edges → `data/fixtures/primekg_mini/{nodes,edges}.csv`, committed.
- `data/README.md` data card: source, license caveats, counts, checksums, date fetched.
- `.github/workflows/ci.yml`: `uv sync`, `ruff check`, `ruff format --check`, `pytest`.

**Tests:** dedup direction logic (fixture with both directions → one), relation-name normalisation, checksum mismatch raises, fetch with mocked HTTP, fixture contains at least one indication per seed disease.

**Done when:** `uv run python scripts/fetch_data.py all && uv run python scripts/prepare.py` succeeds on a clean machine and prints node/edge counts matching `docs/DATA.md`; CI green.

**Risks:** Dataverse file ids change in a new version (pin version 2 and fail loudly with the DOI page URL); 1 GB download on slow links (resume support).
