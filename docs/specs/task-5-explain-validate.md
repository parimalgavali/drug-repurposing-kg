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
