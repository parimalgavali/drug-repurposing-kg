# Model card: drug → disease INDICATION link prediction

**Task.** Rank all 7,957 PrimeKG drugs for a disease; positives are `INDICATION` edges (9,388). Research demo, not medical advice.

**Splits** (`ml/splits.py`, saved to `data/processed/splits/<split>/seed_<k>/`):
- `disease`: 20% test / 10% validation of indicated disease groups (PrimeKG `group_id_bert`, else the disease itself; 1,363 indicated diseases fall into 1,363 groups). Held-out diseases lose all `INDICATION`, `CONTRAINDICATION` and `OFF_LABEL_USE` edges, so they look like diseases with no known drugs.
- `random`: 80/10/10 over indication pairs; clinical edges of held-out pairs are removed.
- Training graph: everything else minus `DRUG_DRUG` and `ANATOMY_PROTEIN_*` (ADR 6): ~1.17M edges.

**Metrics** (`ml/evaluate.py`): per test disease, score all drugs, drop training/validation positives, rank each test positive (ties count half). MRR and Hits@k over test pairs; AUROC/AUPRC with one sampled negative drug per positive.

**Models**
- `popularity`: number of training indications per drug.
- `shared_neighbour`: Adamic–Adar over drug → gene/phenotype ← disease two-hop paths (`TARGET|ENZYME|TRANSPORTER|CARRIER|DRUG_EFFECT` × `DISEASE_PROTEIN|DISEASE_PHENOTYPE_POSITIVE`).
- `similar_disease`: Adamic–Adar disease–disease similarity over shared genes/phenotypes × the similar diseases' training indications (collaborative filter).
- `gnn` (`ml/gnn.py`, `ml/configs/gnn.yaml`): heterogeneous GNN with 2 relation-specific GraphSAGE layers (R-GCN-style `HeteroConv`) and a DistMult decoder for `(drug, INDICATION, disease)`. Inputs are learned ID embeddings for every node type except diseases, plus frozen `all-MiniLM-L6-v2` embeddings of drug descriptions/mechanisms and disease definitions (`ml/text_features.py`). The DrugBank `indication` text is excluded because it names the labels. Training is full-batch BCE with 4 corrupted drugs per pair. Each epoch, 30% of the training indications are hidden from message passing and used as targets. Early stopping is on validation MRR. `gnn_notext` is the ablation without text features.
- `rotate`, `complex`: PyKEEN, dim 64, sLCWA with 4 negatives, Adam 1e-3, batch 8,192, up to 30 epochs with early stopping on validation MRR (checked every 5 epochs). One seed per split because of CPU time.

## Results

Hardware: 8-core CPU, 31 GB RAM, no GPU. "Fit+score s" = fit time + scoring the test diseases.

<!-- metrics:start -->
| Split | Model | MRR | Hits@1 | Hits@10 | Hits@50 | AUROC | AUPRC | Fit+score s | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disease | popularity | 0.110 ± 0.007 | 0.088 ± 0.009 | 0.129 ± 0.004 | 0.271 ± 0.006 | 0.897 ± 0.008 | 0.882 ± 0.008 | 0 | 3 |
| disease | shared_neighbour | 0.011 ± 0.002 | 0.002 ± 0.002 | 0.021 ± 0.003 | 0.070 ± 0.006 | 0.615 ± 0.012 | 0.612 ± 0.014 | 0 | 3 |
| disease | similar_disease | 0.384 ± 0.032 | 0.339 ± 0.028 | 0.452 ± 0.043 | 0.607 ± 0.026 | 0.913 ± 0.005 | 0.911 ± 0.006 | 1 | 3 |
| disease | rotate | 0.015 ± 0.000 | 0.001 ± 0.000 | 0.043 ± 0.000 | 0.110 ± 0.000 | 0.822 ± 0.000 | 0.805 ± 0.000 | 84 | 1 |
| disease | complex | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.008 ± 0.000 | 0.577 ± 0.000 | 0.537 ± 0.000 | 74 | 1 |
| random | popularity | 0.131 ± 0.006 | 0.109 ± 0.005 | 0.149 ± 0.009 | 0.291 ± 0.016 | 0.912 ± 0.004 | 0.901 ± 0.004 | 0 | 3 |
| random | shared_neighbour | 0.011 ± 0.002 | 0.001 ± 0.002 | 0.020 ± 0.003 | 0.078 ± 0.007 | 0.618 ± 0.006 | 0.616 ± 0.002 | 0 | 3 |
| random | similar_disease | 0.365 ± 0.007 | 0.313 ± 0.008 | 0.454 ± 0.009 | 0.628 ± 0.007 | 0.928 ± 0.004 | 0.925 ± 0.005 | 1 | 3 |
| random | rotate | 0.032 ± 0.000 | 0.011 ± 0.000 | 0.066 ± 0.000 | 0.159 ± 0.000 | 0.854 ± 0.000 | 0.822 ± 0.000 | 121 | 1 |
| random | complex | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | 0.576 ± 0.000 | 0.539 ± 0.000 | 87 | 1 |
<!-- metrics:end -->

## Comparison with TxGNN (published, not reproduced)

TxGNN (Huang et al., *Nature Medicine* 2024) trains on PrimeKG too. For indications it reports AUPRC 0.913 in the setting where test diseases already have some drugs (best baseline: HAN at 0.873). In the zero-shot setting, it reports a 19.0% AUPRC gain over the next-best method. Results are averaged over 5 splits. Its numbers aren't directly comparable with ours, because:
- its zero-shot split holds out whole disease areas, which is stricter than our `group_id_bert` grouping;
- its negatives and AUPRC protocol differ from our 1:1 sampled negatives;
- it trains on GPU with metric-learning disease pooling.
Our AUPRC column is closest in spirit. The ranking metrics (MRR, Hits@k) have no TxGNN counterpart.

## Findings and caveats

- **Drug-side shared neighbours lose to popularity.** Most of the 7,957 drugs never have an indication (only 1,801 do), and experimental compounds with many targets/side effects dominate the Adamic–Adar scores. Restricting the same score to drugs with ≥ 1 training indication lifts its AUROC from ~0.62 to 0.90, but its MRR (0.06) is still below popularity. Popularity is the bar that matters.
- **`similar_disease` is the bar for task 4.** Transferring drugs from diseases with similar genes/phenotypes reaches MRR ~0.38 even on the disease split. Some of that is near-duplicate diseases (e.g. subtypes that the BERT grouping leaves ungrouped: 1,145 of the 1,363 indicated diseases have no group), so the disease split is easier than "truly novel disease". A stricter grouping (e.g. via `DISEASE_DISEASE` / MONDO ancestors) is a follow-up; numbers on this split should be read as an upper bound.
- **Missing ≠ negative.** Unlabelled drug–disease pairs are treated as negatives; some are real but unrecorded uses, which is the point of repurposing and depresses all metrics.
- **KGE is undertrained here.** Validation MRR peaked at the first check (5 epochs, ~100 s) and early stopping ended both runs. RotatE reaches MRR 0.015/0.032 (disease/random), and ComplEx is near random (MRR 0.001). With ~1.17M triples and a few CPU minutes, the `INDICATION` relation (9k of 1.17M triples) is barely learned. These numbers are what that budget buys, not a ceiling; task 4's GNN trains on the indication objective directly.
