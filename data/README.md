# Data card

Raw files are downloaded by `scripts/fetch_data.py` into `data/raw/primekg/` (git-ignored). `scripts/prepare.py` writes the canonical tables to `data/processed/` (git-ignored). Only the small fixture in `data/fixtures/` is committed.

## PrimeKG v2

| Field | Value |
| --- | --- |
| Source | Harvard Dataverse, `doi:10.7910/DVN/IXA7BM`, version 2 (released 2022-05-02) |
| Citation | Chandak P., Huang K., Zitnik M. *Building a knowledge graph to enable precision medicine.* Scientific Data 10, 67 (2023) |
| License | Labelled CC0 1.0 on Dataverse. It integrates 20 sources with their own terms: DrugBank-derived content is CC BY-NC 4.0 and the Mayo Clinic text in `disease_features` is copyrighted. **Non-commercial use; raw files are not redistributed.** |
| Fetched | 2026-10-28 |

| File | Dataverse id | Bytes | sha256 |
| --- | --- | --- | --- |
| `kg.csv` | 6180620 | 981,751,236 | `0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69` |
| `nodes.tab` | 6180617 | 8,893,757 | `18ebcf24887e8529b95c7a7c96c81089cc7c5ba22d53a5c05d56480331961af8` |
| `drug_features.tab` | 6180619 | 10,030,011 | `d9db67a3ae9e4e1bea6f50ac55c9e3152c30f9a128b45cec0d5b003fabdc30c6` |
| `disease_features.tab` | 6180618 | 113,534,270 | `bdd18b39aae9c80e37c3716c2bfabaf854f6a5a7e44c8d2e5b082f0dd27124cb` |

## Processed tables (`data/processed/`)

| Table | Rows | Columns |
| --- | --- | --- |
| `nodes.parquet` | 129,375 | `node_index` (int, key), `node_id`, `node_type`, `label` (Neo4j label), `node_name`, `node_source` |
| `edges.parquet` | 4,050,249 | `rel_type`, `src_index`, `src_type`, `dst_index`, `dst_type` |
| `drug_features.parquet` | 7,957 | DrugBank text and descriptors (description, indication, mechanism_of_action, ATC levels, …) |
| `disease_features.parquet` | 17,080 | MONDO/UMLS/Orphanet definitions, Mayo Clinic text |

**Transformations**
- `kg.csv` lists each undirected edge twice. One row is kept per `(rel_type, pair)`. The source is the node whose type comes first in `drug, disease, gene/protein, effect/phenotype, exposure, pathway, biological_process, molecular_function, cellular_component, anatomy`; for same-type relations it's the smaller `node_index`. The result is exactly half of the 8,100,498 rows, and `prepare.py` asserts this.
- `rel_type` is PrimeKG's `relation` in upper snake case (`off-label use` → `OFF_LABEL_USE`). The exception is `drug_protein`, which is split by its display relation into `TARGET`, `ENZYME`, `TRANSPORTER` and `CARRIER`, because one drug–gene pair can have several of them. This gives 33 relationship types.
- `disease_features.tab` has 44,133 rows for 17,080 diseases, because grouped MONDO diseases repeat. The first row per `node_index` is kept. `drug_features.tab` has one row per drug.
- `.tab` files are read with Python's `csv` module, because quoted fields contain newlines.

**Caveats**
- A missing `INDICATION` edge means "not curated", not "does not work". Evaluation negatives are unlabelled pairs, so metrics are lower bounds.
- Disease nodes with `node_source = MONDO_grouped` merge several MONDO ids (`node_id` like `11764_11658_…`).

## Fixture `data/fixtures/primekg_mini/` (committed)

`scripts/make_fixture.py` (seed 7) samples around five diseases: asthma, psoriasis, Parkinson disease, type 2 diabetes mellitus and rheumatoid arthritis. It takes their indicated or contraindicated drugs, associated genes, drug targets, phenotypes, a few other diseases those drugs treat, and one pathway per gene. It then keeps the induced edges, excluding drug–drug and anatomy expression, capped at 2,500. Every indication, contraindication and off-label edge is kept. The result has **498 nodes and 2,328 edges in the same columns as the processed tables**. Regenerate it with `uv run python scripts/make_fixture.py` after `prepare.py`.
