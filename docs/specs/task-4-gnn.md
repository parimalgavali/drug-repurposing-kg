# Task 4: Heterogeneous GNN

**Goal:** a graph neural network that uses the graph's structure and node features, compared fairly with task 3.

**Deliverables**
- `ml/gnn.py` (PyTorch Geometric, CPU): `HeteroData` built from the pruned training graph; node features = learned embeddings plus, for drugs and diseases, a frozen sentence embedding of the description text (`all-MiniLM-L6-v2` or similar, precomputed); encoder = 2-layer R-GCN (or HGT, chosen on validation MRR); decoder = DistMult over `(drug, INDICATION, disease)`; training with `LinkNeighborLoader` and negative sampling.
- Hyper-parameters in `ml/configs/gnn.yaml`; seeds fixed; checkpoint to `artifacts/` (git-ignored).
- Results for both splits added to `ml/model_card.md`, including an ablation without text features. TxGNN's published numbers are listed with the setup differences stated.
- `ml/write_predictions.py`: top-50 non-indicated drugs per disease → Neo4j `PREDICTED_INDICATION {score, rank, model_version}`; the old version is deleted first.

**Tests:** model forward/backward on the fixture in < 10 s; prediction writer is idempotent (integration, skipped without Neo4j); no test edges in the message-passing graph.

**Done when:** the GNN trains in < 2 h on the dev VM (8 CPU cores), the model card has the full comparison table, and predictions are queryable in Neo4j.

**Risks:** the GNN does not beat RotatE or shared-neighbour. That is reported honestly. Then the best model by validation MRR is the one served, whatever it is.
