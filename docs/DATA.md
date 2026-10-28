# Data

Nothing large is committed. `scripts/fetch_data.py` downloads everything below into `data/raw/` (git-ignored) and verifies checksums; derived files go to `data/processed/`. CI uses the small fixture subgraph in `data/fixtures/`.

## PrimeKG (core)

| Field | Value |
| --- | --- |
| Source | Harvard Dataverse, `doi:10.7910/DVN/IXA7BM`, version 2, released 2022-05-02 |
| Paper | Chandak P., Huang K., Zitnik M. "Building a knowledge graph to enable precision medicine." *Scientific Data* 10, 67 (2023) |
| License | Dataset labelled **CC0 1.0** on Dataverse. It integrates 20 resources with their own terms; DrugBank-derived drug nodes/features are CC BY-NC 4.0, Mayo Clinic text in `disease_features` is copyrighted. → **Non-commercial use only; we do not redistribute the raw files.** |
| Access | `https://dataverse.harvard.edu/api/access/datafile/<id>`, no login |

| File | Dataverse id | Size | sha256 (first/last 8) | Use |
| --- | --- | --- | --- | --- |
| `kg.csv` | 6180620 | 981,751,236 B | `0a5a010f…1cfb5c69` | edges (both directions) with node names/types |
| `nodes.tab` | 6180617 | 8,893,757 B | `18ebcf24…31961af8` | 129,375 nodes |
| `drug_features.tab` | 6180619 | 10,030,011 B | `d9db67a3…dabdc30c6` | DrugBank description, mechanism, ATC, … |
| `disease_features.tab` | 6180618 | 113,534,270 B | `bdd18b39…dd27124cb` | MONDO/UMLS/Orphanet definitions, Mayo text |

Full checksums are in `primekg/fetch.py` and in the [data card](../data/README.md).

**Node types:** gene/protein 27,671 · biological_process 28,642 · disease 17,080 · effect/phenotype 15,311 · anatomy 14,035 · molecular_function 11,169 · drug 7,957 · cellular_component 4,176 · pathway 2,516 · exposure 818.

**Relations (undirected edge counts; `kg.csv` stores each twice):** drug_drug 2,672,628 · anatomy_protein_present 1,518,203 · protein_protein 642,150 · disease_phenotype_positive 150,317 · bioprocess_protein 144,805 · bioprocess_bioprocess 105,772 · cellcomp_protein 83,402 · disease_protein 80,411 · molfunc_protein 69,530 · drug_effect 64,784 · disease_disease 64,388 · pathway_protein 42,646 · phenotype_phenotype 37,472 · **contraindication 30,675** · anatomy_anatomy 28,064 · molfunc_molfunc 27,148 · anatomy_protein_absent 19,887 · drug_protein (target 16,380 / enzyme 5,317 / transporter 3,092 / carrier 864) · cellcomp_cellcomp 9,690 · **indication 9,388** · pathway_pathway 5,070 · exposure_exposure 4,140 · phenotype_protein 3,330 · **off-label use 2,568** · exposure_disease 2,304 · exposure_bioprocess 1,625 · exposure_protein 1,212 · disease_phenotype_negative 1,193 · exposure_molfunc 45 · exposure_cellcomp 10.

**Caveats**
- Undirected: every edge appears as (x, y) and (y, x). The loader keeps one direction per relation with a fixed type order (e.g. drug → disease).
- `x_source` for diseases is `MONDO` or `MONDO_grouped` (PrimeKG merges near-duplicate diseases); keep the grouped id as the node key.
- Indication labels come from DrugBank / DrugCentral-style curation. A missing edge does not mean "does not work". Negatives in evaluation are therefore "unlabelled", and metrics are lower bounds.

## ClinicalTrials.gov API v2 (external validation)

| Field | Value |
| --- | --- |
| Endpoint | `https://clinicaltrials.gov/api/v2/studies?query.cond=<disease>&query.intr=<drug>&countTotal=true` |
| License | US government public data. No key needed. Be polite: ≤ 3 requests/s, cache every response in `data/processed/ctgov_cache.jsonl` |
| Use | For each (drug, disease) pair: number of registered studies, max phase, statuses. Reported as a weak signal (name matching is approximate) |

## Hetionet (stretch only)

`hetio/hetionet` on GitHub, v1.0 (2017). Its original content is CC0; the JSON and Neo4j formats carry each source's license per node and edge. It's used only for the task 9b cross-check (`validation/hetionet.py`), which reads just the 755 `CtD` (Compound–treats–Disease) edges from `hetnet/tsv/hetionet-v1.0-edges.sif.gz`. The file is pinned to commit `8a6cc0c` and checked by sha256.

Disease Ontology ids are mapped to MONDO with MONDO's curated exact-match set (`mondo_exactmatch_doid.sssom.tsv`, CC BY 4.0, mapping set 2026-10-06), pinned to commit `65a4751` and also checked by sha256. Both files go to `data/raw/hetionet/` (git-ignored) via `make validate-hetionet`.

## Fixture subgraph (committed)

`data/fixtures/primekg_mini/` (task 1): a deterministic 2,328-edge / 498-node sample around five diseases (asthma, psoriasis, Parkinson disease, type 2 diabetes mellitus, rheumatoid arthritis). It uses the processed-table columns (see the data card) and is used by every test and by CI.
