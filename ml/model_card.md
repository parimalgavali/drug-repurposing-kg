# Model card: drug → disease INDICATION link prediction

**Task.** Rank all 7,957 PrimeKG drugs for a disease; positives are `INDICATION` edges (9,388). Research demo, not medical advice.

**Splits** (`ml/splits.py`, saved to `data/processed/splits/<split>/seed_<k>/`):
- `disease`: 20% test / 10% validation of indicated disease groups (PrimeKG `group_id_bert`, else the disease itself; 1,363 indicated diseases fall into 1,363 groups). Held-out diseases lose all `INDICATION`, `CONTRAINDICATION` and `OFF_LABEL_USE` edges, so they look like diseases with no known drugs.
- `disease_strict`: like `disease`, but indicated diseases that share a `group_id_bert` or a direct `DISEASE_DISEASE` (MONDO parent/child) edge are merged into one connected component, and whole components are held out until 20% / 10% of indicated diseases are reached (545 components; the largest has 223 diseases). No test disease has a direct ontology neighbour with training indications. On `disease`, 38–49% of test pairs had their drug already indicated for such a neighbour in training (`ontology_leakage` in `ml/splits.py`).
- `random`: 80/10/10 over indication pairs; clinical edges of held-out pairs are removed.
- Training graph: everything else minus `DRUG_DRUG` and `ANATOMY_PROTEIN_*` (ADR 6): ~1.17M edges.

**Metrics** (`ml/evaluate.py`): per test disease, score all drugs, drop training/validation positives, rank each test positive (ties count half). MRR and Hits@k over test pairs; AUROC/AUPRC with one sampled negative drug per positive.

**Models**
- `popularity`: number of training indications per drug.
- `shared_neighbour`: Adamic–Adar over drug → gene/phenotype ← disease two-hop paths (`TARGET|ENZYME|TRANSPORTER|CARRIER|DRUG_EFFECT` × `DISEASE_PROTEIN|DISEASE_PHENOTYPE_POSITIVE`).
- `similar_disease`: Adamic–Adar disease–disease similarity over shared genes/phenotypes × the similar diseases' training indications (collaborative filter).
- `gnn` (`ml/gnn.py`, `ml/configs/gnn.yaml`): heterogeneous GNN with 2 relation-specific GraphSAGE layers (R-GCN-style `HeteroConv`) and a DistMult decoder for `(drug, INDICATION, disease)`. Inputs are learned ID embeddings for every node type except diseases, plus frozen `all-MiniLM-L6-v2` embeddings of drug descriptions/mechanisms and disease definitions (`ml/text_features.py`). The DrugBank `indication` text is excluded because it names the labels. Training is full-batch BCE with 4 corrupted drugs per pair. Each epoch, 30% of the training indications are hidden from message passing and used as targets. Early stopping is on validation MRR. `gnn_notext` is the ablation without text features.
- `gnn_pool` (TxGNN-style): `gnn` plus disease-similarity pooling. After message passing, each disease embedding is mixed with the similarity-weighted mean of its 10 most similar training diseases (Adamic–Adar over shared genes/phenotypes), with gate sigmoid(a − |b|·log(1 + visible indications)), so diseases without known drugs rely on their neighbours. Training hides 30% of training *diseases* per epoch (`supervision: disease`) instead of 30% of pairs, so it practises the unseen-disease case. `gnn_dmask` is the ablation with disease-level hiding but no pooling.
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
| disease | gnn | 0.146 ± 0.010 | 0.088 ± 0.010 | 0.265 ± 0.023 | 0.514 ± 0.026 | 0.956 ± 0.007 | 0.958 ± 0.007 | 300 | 3 |
| disease | gnn_notext | 0.083 ± 0.015 | 0.052 ± 0.012 | 0.133 ± 0.026 | 0.252 ± 0.049 | 0.901 ± 0.028 | 0.897 ± 0.029 | 97 | 3 |
| disease | gnn_blend | 0.374 ± 0.036 | 0.325 ± 0.040 | 0.452 ± 0.026 | 0.635 ± 0.008 | 0.940 ± 0.031 | 0.954 ± 0.018 | 4 | 3 |
| disease | gnn_dmask | 0.200 ± 0.066 | 0.134 ± 0.047 | 0.326 ± 0.107 | 0.563 ± 0.115 | 0.961 ± 0.011 | 0.964 ± 0.013 | 428 | 3 |
| disease | gnn_pool | 0.186 ± 0.048 | 0.119 ± 0.028 | 0.310 ± 0.085 | 0.523 ± 0.124 | 0.961 ± 0.008 | 0.962 ± 0.011 | 417 | 3 |
| disease_strict | popularity | 0.109 ± 0.034 | 0.075 ± 0.042 | 0.148 ± 0.039 | 0.272 ± 0.048 | 0.837 ± 0.021 | 0.832 ± 0.023 | 0 | 3 |
| disease_strict | shared_neighbour | 0.013 ± 0.001 | 0.002 ± 0.001 | 0.025 ± 0.002 | 0.078 ± 0.009 | 0.618 ± 0.012 | 0.612 ± 0.015 | 0 | 3 |
| disease_strict | similar_disease | 0.223 ± 0.016 | 0.183 ± 0.010 | 0.287 ± 0.031 | 0.458 ± 0.073 | 0.843 ± 0.019 | 0.845 ± 0.021 | 1 | 3 |
| disease_strict | gnn | 0.099 ± 0.023 | 0.050 ± 0.017 | 0.200 ± 0.045 | 0.410 ± 0.061 | 0.915 ± 0.017 | 0.923 ± 0.016 | 561 | 3 |
| disease_strict | gnn_dmask | 0.158 ± 0.018 | 0.088 ± 0.012 | 0.286 ± 0.021 | 0.471 ± 0.047 | 0.915 ± 0.017 | 0.928 ± 0.015 | 744 | 3 |
| disease_strict | gnn_pool | 0.139 ± 0.008 | 0.085 ± 0.014 | 0.239 ± 0.035 | 0.452 ± 0.038 | 0.916 ± 0.015 | 0.927 ± 0.012 | 434 | 3 |
| disease_strict | gnn_blend | 0.201 ± 0.069 | 0.154 ± 0.069 | 0.287 ± 0.069 | 0.498 ± 0.078 | 0.914 ± 0.015 | 0.927 ± 0.015 | 5 | 3 |
| random | popularity | 0.131 ± 0.006 | 0.109 ± 0.005 | 0.149 ± 0.009 | 0.291 ± 0.016 | 0.912 ± 0.004 | 0.901 ± 0.004 | 0 | 3 |
| random | shared_neighbour | 0.011 ± 0.002 | 0.001 ± 0.002 | 0.020 ± 0.003 | 0.078 ± 0.007 | 0.618 ± 0.006 | 0.616 ± 0.002 | 0 | 3 |
| random | similar_disease | 0.365 ± 0.007 | 0.313 ± 0.008 | 0.454 ± 0.009 | 0.628 ± 0.007 | 0.928 ± 0.004 | 0.925 ± 0.005 | 1 | 3 |
| random | rotate | 0.032 ± 0.000 | 0.011 ± 0.000 | 0.066 ± 0.000 | 0.159 ± 0.000 | 0.854 ± 0.000 | 0.822 ± 0.000 | 121 | 1 |
| random | complex | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | 0.576 ± 0.000 | 0.539 ± 0.000 | 87 | 1 |
| random | gnn | 0.479 ± 0.036 | 0.371 ± 0.040 | 0.693 ± 0.012 | 0.868 ± 0.004 | 0.986 ± 0.002 | 0.989 ± 0.001 | 481 | 3 |
| random | gnn_notext | 0.512 ± 0.010 | 0.408 ± 0.009 | 0.717 ± 0.008 | 0.839 ± 0.005 | 0.976 ± 0.001 | 0.982 ± 0.001 | 481 | 3 |
| random | gnn_blend | 0.493 ± 0.023 | 0.395 ± 0.027 | 0.681 ± 0.023 | 0.865 ± 0.007 | 0.982 ± 0.002 | 0.986 ± 0.001 | 4 | 3 |
| random | gnn_dmask | 0.428 ± 0.013 | 0.322 ± 0.008 | 0.655 ± 0.013 | 0.840 ± 0.008 | 0.986 ± 0.002 | 0.987 ± 0.002 | 1012 | 3 |
| random | gnn_pool | 0.365 ± 0.018 | 0.254 ± 0.013 | 0.582 ± 0.028 | 0.827 ± 0.019 | 0.986 ± 0.002 | 0.986 ± 0.002 | 1207 | 3 |
<!-- metrics:end -->

## Clinical-trial check (ClinicalTrials.gov)

This is an independent check of novel predictions on the disease split (seed 0). For a seeded sample of held-out diseases, each method's top 20 drugs are compared against ClinicalTrials.gov. Drugs with any recorded `INDICATION`/`OFF_LABEL_USE` for that disease are excluded first, so only novel pairs are counted. The table shows the share of those pairs with at least one registered study (`query.cond` = disease name, `query.intr` = drug name). A registered study means someone tested the idea, **not** that the drug works. Name matching is literal, so misses also include naming mismatches. Produced by `validation/run.py` (cache: `artifacts/ctgov_cache.jsonl`).

<!-- ctgov:start -->
| Method | Pairs | With ≥ 1 study | Share (95% Wilson CI) | Max phase ≥ 2 |
| --- | ---: | ---: | --- | ---: |
| gnn (text) | 600 | 92 | 0.153 (0.127–0.184) | 73 |
| similar_disease | 600 | 89 | 0.148 (0.122–0.179) | 78 |
| popularity | 600 | 79 | 0.132 (0.107–0.161) | 55 |
| random drugs | 600 | 3 | 0.005 (0.002–0.015) | 3 |
<!-- ctgov:end -->

The sample is 30 held-out diseases × top 20 novel drugs. 29 of the 30 disease names match studies on CT.gov by themselves; the exception is "obsolete vitamin D deficiency". CT.gov rejected 84 queries with HTTP 400, mostly long IUPAC-style compound names, and those count as "no study".

- All three methods are well above random drugs (≈ 30×), so predictions carry real signal beyond chance.
- GNN ≥ similar_disease > popularity on this check, but the confidence intervals overlap, so this does **not** show the GNN beats the similar-disease baseline on held-out diseases. That agrees with the MRR results above.
- The gnn row is the disease-split, seed-0, text model, because the served random-split model has seen these diseases' indications.
- Served predictions (`gnn_notext-random-seed0`) for the 30 sampled diseases, ranks 1–10, have `ct_studies` / `ct_max_phase` set on `PREDICTED_INDICATION` (300 relationships). The UI looks up any other candidate on demand when it is clicked (`/api/pairs/{drug}/{disease}/trials`, cached on disk).

## Comparison with TxGNN (published, not reproduced)

TxGNN (Huang et al., *Nature Medicine* 2024) trains on PrimeKG too. For indications it reports AUPRC 0.913 in the setting where test diseases already have some drugs (best baseline: HAN at 0.873). In the zero-shot setting, it reports a 19.0% AUPRC gain over the next-best method. Results are averaged over 5 splits. Its numbers aren't directly comparable with ours, because:
- its zero-shot split holds out whole disease areas, which is stricter than our `group_id_bert` grouping;
- its negatives and AUPRC protocol differ from our 1:1 sampled negatives;
- it trains on GPU with metric-learning disease pooling.
Our AUPRC column is closest in spirit. The ranking metrics (MRR, Hits@k) have no TxGNN counterpart.

## Findings and caveats

- **Drug-side shared neighbours lose to popularity.** Most of the 7,957 drugs never have an indication (only 1,801 do), and experimental compounds with many targets/side effects dominate the Adamic–Adar scores. Restricting the same score to drugs with ≥ 1 training indication lifts its AUROC from ~0.62 to 0.90, but its MRR (0.06) is still below popularity. Popularity is the bar that matters.
- **`similar_disease` is the bar for task 4.** Transferring drugs from diseases with similar genes/phenotypes reaches MRR ~0.38 even on the disease split. Some of that is near-duplicate diseases (e.g. subtypes that the BERT grouping leaves ungrouped: 1,145 of the 1,363 indicated diseases have no group), so the disease split is easier than "truly novel disease". A stricter grouping (e.g. via `DISEASE_DISEASE` / MONDO ancestors) is a follow-up; numbers on this split should be read as an upper bound.
- **The GNN wins when a disease has some known drugs, not when it has none.** On the random split it reaches MRR 0.48 / Hits@10 0.69 against 0.37 / 0.45 for `similar_disease`. On the disease split it reaches 0.15, which beats popularity, RotatE and shared-neighbour but stays below `similar_disease` (0.38). Its AUROC/AUPRC is the best on both splits (0.96/0.96 disease, 0.99/0.99 random): it separates indicated drugs from random ones well, but it is weaker at putting the right drug first for a disease it has never seen.
- **Text features matter for unseen diseases only.** Without text, the disease-split MRR falls from 0.15 to 0.08. On the random split, the no-text variant is slightly better on MRR (0.51 vs 0.48) and worse on Hits@50/AUROC. When a disease already has drugs, its ID-free neighbourhood carries the signal and the text adds noise to the top ranks. Removing the learned disease ID embeddings (so a disease is only its text + neighbourhood) raised validation MRR from 0.10 to 0.13 during tuning, and going from hidden 64 to 128 raised it to 0.19.
- **Tuning was light and done on disease-split seed 0 validation only.** Learning rate {1e-3, 3e-3, 5e-3}, layers {2, 3} and hidden {64, 128} were tried; HGT was not. The random-split runs hit the 400-epoch cap while still improving, so those numbers are not a ceiling. Training is full-batch (the whole pruned graph fits in memory and an epoch takes about 1 s on 8 cores), so the spec's `LinkNeighborLoader` was not needed.
- **Served model.** Picked by validation MRR on random-split seed 0, the setting where every disease has training drugs: `gnn_notext` 0.44, `gnn` 0.41, `similar_disease` 0.29. So `ml/write_predictions.py --model gnn_notext` writes that checkpoint's top 50 non-indicated drugs per disease to Neo4j, as `model_version` `gnn_notext-random-seed0`. For a disease with no known drugs, the text model would be the better choice; the API (task 7) can flag that case.
- **Missing ≠ negative.** Unlabelled drug–disease pairs are treated as negatives; some are real but unrecorded uses, which is the point of repurposing and depresses all metrics.
- **KGE is undertrained here.** Validation MRR peaked at the first check (5 epochs, ~100 s) and early stopping ended both runs. RotatE reaches MRR 0.015/0.032 (disease/random), and ComplEx is near random (MRR 0.001). With ~1.17M triples and a few CPU minutes, the `INDICATION` relation (9k of 1.17M triples) is barely learned. These numbers are what that budget buys, not a ceiling; task 4's GNN trains on the indication objective directly.

## Hetionet cross-check

This is a second, independent check, against another curated graph. Hetionet's `CtD` (treats) edges come from a curated set of indications (PharmacotherapyDB). The check asks: of the drug–disease pairs Hetionet records but PrimeKG doesn't, how many does each method put in its top-k novel drugs for that disease?
- **Mapping:** drugs map by DrugBank id. Diseases map DOID → MONDO through MONDO's exact-match set, then to every PrimeKG disease node that contains that MONDO id.
- **Fairness:** all methods use the random split (seed 0), the one the served model was chosen on. Drugs with any recorded PrimeKG indication or off-label use for the disease are excluded and the rest re-ranked.
- **No leakage:** Hetionet-only pairs are absent from PrimeKG, so no method saw them in training.
- **Reproduce:** `make validate-hetionet`.

**Result:** every method is far above random (57 vs 1 hit in the top 20), but the similar-disease baseline finds more Hetionet-only pairs than the served GNN (73 vs 57). The confidence intervals barely overlap. This agrees with the ClinicalTrials.gov check: the GNN's edge only shows up on PrimeKG's own held-out indications.

<!-- hetionet:start -->
Hetionet v1.0 has 755 `CtD` edges; 752 map to PrimeKG drug–disease pairs. Of those, 334 are already PrimeKG indications or off-label uses and 12 are PrimeKG *contraindications*. The remaining **418 Hetionet-only pairs** over 64 diseases are the truth set; every method ranks only drugs without a recorded use.

| Method | Hits in top 20 | Precision@20 (95% CI) | Recall@20 | Hits in top 50 | Recall@50 |
| --- | ---: | --- | ---: | ---: | ---: |
| served GNN | 57 | 0.045 (0.035–0.057) | 0.136 | 101 | 0.242 |
| similar_disease | 73 | 0.057 (0.046–0.071) | 0.175 | 119 | 0.285 |
| popularity | 33 | 0.026 (0.018–0.036) | 0.079 | 90 | 0.215 |
| random drugs | 1 | 0.001 (0.000–0.004) | 0.002 | 4 | 0.010 |
<!-- hetionet:end -->

## GNNExplainer vs metapath explanations

<!-- gnnexplainer:start -->
20 served rank-1 predictions, 100 GNNExplainer epochs each.

| Top-k GNNExplainer edges | Share on metapath paths | Random top-k | Lift | Path edges recovered | Pairs with ≥1 shared edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.060 | 0.0021 | 33.9× | 0.029 | 7/20 |
| 20 | 0.040 | 0.0021 | 21.4× | 0.041 | 7/20 |

Mean drop of the pair's logit when the edges are deleted from message passing (same number of edges each):

| Deleted edges | Mean logit drop |
| --- | ---: |
| metapath path edges | 0.818 |
| top GNNExplainer edges | 1.678 |
| random receptive-field edges | 0.006 |

Relations of the top-20 GNNExplainer edges (all pairs): CONTRAINDICATION 92, OFF_LABEL_USE 78, DRUG_EFFECT 41, ENZYME 30, DISEASE_DISEASE 28, TRANSPORTER 24, PATHWAY_PROTEIN 22, CARRIER 13

Relations of the metapath path edges: DISEASE_PHENOTYPE_POSITIVE 159, INDICATION 111, DISEASE_PROTEIN 91, DRUG_EFFECT 90, ENZYME 28, PHENOTYPE_PROTEIN 23, TRANSPORTER 23, OFF_LABEL_USE 22
<!-- gnnexplainer:end -->

Run with `make gnnexplainer` (needs the full graph in Neo4j and the served checkpoint). Each pair's
GNNExplainer run is restricted to the edges inside the model's 2-layer receptive field of the drug
and the disease. The pair's logit on that subgraph is identical to the full graph's, which a test
checks. Results per pair are in `ml/results/gnnexplainer.json`.

**How faithful are the path explanations?** Only partly.
- **Low overlap, but far above chance.** 4–6% of the top GNNExplainer edges lie on a top metapath
  path, about 20–34× what a random top-k from the receptive field would hit. Only 7 of 20 pairs
  share any edge. GNNExplainer's top 20 recovers about 4% of the path edges.
- **Some of the gap is by design.** 31% of the top-20 GNNExplainer edges (124 of 400) use relations
  the path explainer never walks, mostly `CONTRAINDICATION` (92). The GNN uses a drug's
  contraindications and off-label uses as a profile of the drug; the path explainer only follows
  mechanism-like relations.
- **Deleting the edges.** Removing the path edges from message passing lowers the logit by 0.82 on
  average, against 1.68 for the same number of top GNNExplainer edges and 0.006 for random edges.
  So the path edges matter to the model far more than random edges do. But in 9 of 20 pairs,
  removing them left the logit flat or raised it.
- **Caveat: GNNExplainer's ranking is coarse.** With PyG's defaults (100 epochs, lr 0.01), many
  masks end within 0.005 of the top weight. In 8 of 20 pairs, 12 or more of the top 20 are tied,
  and ties are broken by edge id. The overlap numbers are only as precise as that ranking.

**Examples.**
- **Bleomycin → hyperparathyroidism 2 with jaw tumors.** The top metapath is
  Drug-INDICATION-Disease-DISEASE_PROTEIN-Gene-DISEASE_PROTEIN-Disease, i.e. Bleomycin treats a
  disease that shares genes with this one. GNNExplainer's top edge is BLMH -ENZYME- Bleomycin, and
  no edge is shared. Deleting the path edges drops the logit by 6.4, more than GNNExplainer's edges
  (2.5): here the paths are the more faithful explanation.
- **Phenytoin → myocardial infarction.** The top metapath runs through drugs with the same side
  effects and their off-label uses. GNNExplainer's top edge is Phenytoin -OFF_LABEL_USE- bipolar
  disorder. No edge is shared, and deleting GNNExplainer's edges drops the logit by 8.5, against
  0.7 for the path edges.
- **Tobramycin → serum sickness.** GNNExplainer's top edge is Tobramycin -CONTRAINDICATION-
  diabetes mellitus, which no path can contain. Deleting the path edges raises the logit by 1.1.
- **Deutetrabenazine → developmental malformations-deafness-dystonia syndrome** is one of the
  pairs where the two agree: 3 of GNNExplainer's top 20 edges lie on the paths, which run through
  shared phenotypes.

**Takeaway:** the metapath paths shown in the UI are readable, plausible graph evidence, but not
a faithful account of why the served GNN ranks a drug. They should be read as supporting evidence,
not as the model's reasons.

## GNN + similar-disease blend (unseen diseases)

This was a time-boxed attempt to close the disease-split gap: GNN MRR 0.15 vs `similar_disease` 0.38. `gnn_blend` (`ml/blend.py`, `make ml-blend`) z-scores the text GNN's and `similar_disease`'s scores per disease and ranks by α·gnn + (1 − α)·similar_disease. α ∈ {0, 0.1, …, 1} is picked per seed on **validation** MRR. It reuses the task-4 checkpoints, so nothing is retrained.

**Result: the blend does not close the gap.** On the disease split it ties `similar_disease` on MRR (0.374 ± 0.036 vs 0.384 ± 0.032). It adds about 3 points of Hits@50 (0.635 vs 0.607). Paired test results per seed, with α picked on validation:

| Seed | α picked (validation) | `similar_disease` MRR / Hits@50 | `gnn_blend` MRR / Hits@50 | GNN alone MRR |
| ---: | ---: | --- | --- | ---: |
| 0 | 0.0 | 0.397 / 0.641 | 0.397 / 0.641 | 0.150 |
| 1 | 0.9 | 0.339 / 0.579 | 0.323 / 0.639 | 0.156 |
| 2 | 0.7 | 0.414 / 0.600 | 0.402 / 0.624 | 0.133 |

- **The validation curve is flat.** For α from 0.1 to 0.9, validation MRR moves by at most 0.03, and it only collapses at α = 1 (the GNN alone). The picked α is therefore noise (0.0, 0.9, 0.7), and on seeds 1 and 2 it lands slightly below `similar_disease`.
- **The GNN helps further down the list, not at the top.** Looked at after the fact on test, for information only, a fixed α = 0.7 gives MRR 0.385 (vs 0.383) and Hits@50 0.643 (vs 0.607). The GNN reorders the long tail, but it rarely moves the right drug to rank 1 for a disease it has never seen.
- **Random split:** validation picks α 0.9–1.0, so the blend is essentially the text GNN (MRR 0.493 vs 0.479). `gnn_notext` (0.512) is still the best model when diseases have known drugs.
- **Serving is unchanged.** The UI keeps serving `gnn_notext-random-seed0`. For a truly unseen disease, `similar_disease`, or the blend if recall at 50 matters, is the better ranking. Closing the gap at the top ranks likely needs a stricter disease split and a model change (e.g. TxGNN-style disease-similarity pooling), not score mixing.

## Stricter unseen-disease split and TxGNN-style pooling

**Why a stricter split.** On `disease`, 38–49% of test pairs (per seed) have their drug already indicated in training for a direct MONDO parent or child of the test disease, and 53–55% of test diseases have such a neighbour. `disease_strict` holds out whole connected components of indicated diseases, linked by `DISEASE_DISEASE` edges or a shared `group_id_bert`, which brings both numbers to 0. Grouping by two-hop links instead is not usable: it puts 1,279 of the 1,363 indicated diseases into one component.

**Result: the stricter split lowers every model, and the TxGNN-style model does not beat `similar_disease`.** Mean ± std over 3 seeds, 296 test diseases and 1,959 test pairs on average:

| Model | `disease` MRR | `disease_strict` MRR | `disease_strict` Hits@50 |
| --- | ---: | ---: | ---: |
| popularity | 0.110 | 0.109 ± 0.034 | 0.272 |
| similar_disease | 0.384 | 0.223 ± 0.016 | 0.458 |
| gnn (text, pair-level hiding) | 0.146 | 0.099 ± 0.023 | 0.410 |
| gnn_dmask (disease-level hiding) | 0.200 ± 0.066 | 0.158 ± 0.018 | 0.471 |
| gnn_pool (disease-level hiding + pooling) | 0.186 ± 0.048 | 0.139 ± 0.008 | 0.452 |
| gnn_blend (gnn + similar_disease) | 0.374 | 0.201 ± 0.069 | 0.498 |

- **The old split overstated unseen-disease performance.** `similar_disease` loses 42% of its MRR (0.38 → 0.22), because it worked mostly by copying drugs from near-duplicate training diseases. Popularity doesn't depend on neighbours and stays put.
- **Training like the test helps the GNN, but pooling doesn't add to it.** Hiding whole diseases during training (`gnn_dmask`) raises strict MRR from 0.099 to 0.158 and Hits@50 from 0.41 to 0.47. Adding the similarity pooling on top (`gnn_pool`) gives 0.139, which is within noise of or below `gnn_dmask`. The learned gate gives an unseen disease about 48% weight on its 10 neighbours (sigmoid(a) with a ≈ −0.07). The degree term gets no gradient, because target diseases always have zero visible indications under disease-level hiding.
- **The same holds on the old split.** On `disease`, disease-level hiding raises GNN MRR from 0.146 to 0.200 (`gnn_dmask`), and pooling gives 0.186. Both have large seed variance (per-seed `gnn_dmask` 0.26 / 0.11 / 0.23), and both stay well below `similar_disease` (0.384).
- **The gap to `similar_disease` shrinks but stays.** On strict MRR, the best GNN reaches 71% of `similar_disease` (0.158 vs 0.223), up from 38% on `disease` (0.146 vs 0.384). On Hits@50 `gnn_dmask` is slightly ahead (0.471 vs 0.458), the same pattern as before: the GNN helps further down the list, not at rank 1.
- **The splits are uneven.** One component holds 223 indicated diseases and must go wholly to one side. On seed 2 it lands in validation (2,724 validation pairs, 4,568 training pairs). There, `similar_disease` scores only 0.079 validation MRR against the GNN's 0.125, so validation picks α = 1.0 for the blend and its test MRR falls to 0.104. That one seed explains the blend's large std; on seeds 0 and 1 the blend beats `similar_disease` (0.238 vs 0.220, 0.261 vs 0.244).
- **Disease-level hiding costs accuracy when diseases already have drugs.** On `random`, MRR is 0.428 ± 0.013 for `gnn_dmask` and 0.365 ± 0.018 for `gnn_pool`, against 0.479 for `gnn` and 0.512 for `gnn_notext`. Training only on fully hidden diseases means the model never practises using a disease's own known drugs, and pooling pulls a known disease toward its neighbours.
- **Not TxGNN.** This borrows TxGNN's idea of disease-similarity pooling and zero-shot training, but not its pretraining, its metric-learning module or its exact similarity. Published TxGNN numbers remain not comparable (see above).
- **Serving is unchanged.** The UI keeps serving `gnn_notext-random-seed0`.
