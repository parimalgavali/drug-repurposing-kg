# Task 3: Splits & baselines

**Goal:** an honest evaluation harness and the numbers the GNN must beat.

**Deliverables**
- `ml/splits.py`: (a) **disease-held-out**: group diseases by PrimeKG's grouped MONDO id, hold out 20% of groups (seeded) as test and 10% as validation, and remove all their `INDICATION` edges (and `OFF_LABEL_USE`/`CONTRAINDICATION` for the same pairs) from training; (b) **random**: 80/10/10 on `INDICATION` edges. Saved as parquet in `data/processed/splits/<name>/`.
- `ml/evaluate.py`: for each test disease, score **all** drugs, filter known training positives, and compute MRR, Hits@1/10/50 and AUROC/AUPRC (1:1 negatives sampled from unlabelled drugs). Reports mean ± std over 3 seeds.
- `ml/baselines.py`: `popularity` (number of indications per drug); `shared_neighbour` (Adamic–Adar over drug→gene←disease and drug→phenotype←disease two-hop paths).
- `ml/kge.py`: PyKEEN RotatE and ComplEx on the pruned training graph (ADR 6), CPU, early stopping on validation MRR.
- `ml/model_card.md`: table per split × model, training time, hardware, known label caveats ("missing ≠ negative").

**Tests:** no leakage (no test indication pair, in either direction, appears in training edges); evaluator on a toy graph with a known answer; baselines are deterministic.

**Done when:** `uv run python -m ml.run --models popularity,shared_neighbour,rotate,complex --split disease` writes metrics JSON and updates the model card; the shared-neighbour baseline beats popularity on the disease split (otherwise the harness is suspect and gets debugged before task 4).

**Risks:** KGE training time on CPU (cap epochs, record time); disease groups that are too coarse (report test-set size).

**Outcome:** the drug-side Adamic–Adar `shared_neighbour` loses to popularity on both splits (it favours never-indicated compounds with many targets). It is kept and reported. A disease-side variant, `similar_disease` (drugs of diseases sharing genes/phenotypes), clearly beats popularity and becomes the bar for task 4. The disease split leaks some signal through near-duplicate diseases that the BERT grouping leaves ungrouped. That is documented in `ml/model_card.md`, and a stricter grouping is a follow-up. KGE models run with 1 seed per split because of CPU time.
