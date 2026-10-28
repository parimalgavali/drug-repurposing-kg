# Task 5: Explanations & clinical-trial check

**Goal:** every prediction comes with checkable evidence, plus one independent signal that predictions are better than chance.

**Deliverables**
- `explain/paths.py`: for a (drug, disease) pair, enumerate paths of length ≤ 3 through the training graph (Cypher, bounded), group them by metapath (e.g. `Drug-TARGET-Gene-ASSOCIATED_WITH-Disease`), and rank them by a degree-weighted path count (DWPC, as in Hetionet). Returns the top 3 metapaths with up to 5 example paths each, as JSON nodes and edges ready for Cytoscape.
- `validation/ctgov.py`: for each pair, query ClinicalTrials.gov API v2 (`query.cond`, `query.intr`, `countTotal=true`) with polite rate limiting and a JSONL cache, and record the study count, max phase and statuses.
- `validation/run.py`: for the disease-held-out test diseases, compare the share of top-20 predicted pairs with ≥ 1 registered study against (a) the popularity baseline's top-20 and (b) 20 random drugs. Report this with a binomial confidence interval in the model card.
- Trial counts for served predictions are stored as properties on `PREDICTED_INDICATION` (`ct_studies`, `ct_max_phase`).

**Tests:** metapath grouping and DWPC on a toy graph; CT.gov client with recorded HTTP responses; cache hit avoids network.

**Done when:** `uv run python -m explain.paths --drug <name> --disease <name>` prints paths for a known indication (e.g. a drug indicated for asthma); the model card has the trial-validation table.

**Risks:** name mismatch with CT.gov (also try PrimeKG synonyms if present; report match rate); trials ≠ efficacy (wording in UI and card).

## Outcome

- `explain.paths`: Salbutamol → asthma returns 864 paths in about 1 s. The top metapaths go through `bronchial disease` and sibling bronchodilators. The integration test checks DWPC ordering and Cytoscape JSON on the fixture.
- Trial check (30 held-out diseases, top 20, novel pairs): GNN 15.3 % [12.7–18.4], similar_disease 14.8 %, popularity 13.2 %, random 0.5 %. All methods beat random by a wide margin. The GNN vs. baseline intervals overlap. Full table in `ml/model_card.md`.
- Deviation: only a sample is annotated with `ct_studies`/`ct_max_phase` (300 served relationships), because CT.gov allows about 0.7 queries/s. Annotating all 854k is infeasible, so the UI should look up on demand via `validation.ctgov` with its cache.
- Synonyms: lookups use PrimeKG names only. CT.gov's own condition search expands synonyms, which is why disease names match well. Drug-name 400s are counted as misses.
