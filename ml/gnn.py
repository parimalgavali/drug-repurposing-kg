"""Heterogeneous GNN for drug → disease INDICATION prediction (PyTorch Geometric, CPU).

Encoder: learned node embeddings (+ projected frozen text embeddings for drugs/diseases), then
relation-specific GraphSAGE layers combined per node type (an R-GCN-style HeteroConv).
Decoder: DistMult over (drug, INDICATION, disease). Training is full-batch: the pruned graph
(~1.2M edges) fits in memory, so the GNN sees exact neighbourhoods instead of sampled ones.

Leakage guard: each epoch a random `supervision_ratio` of the training indications is removed
from message passing and used as targets, so a pair's own edge never predicts itself.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from ml.evaluate import ScoreFn, evaluate
from ml.splits import Split

CONFIG = Path(__file__).with_name("configs") / "gnn.yaml"
TEXT = Path("artifacts/text_features.npz")
CHECKPOINTS = Path("artifacts/gnn")
IND = ("drug", "INDICATION", "disease")
REV_IND = ("disease", "rev_INDICATION", "drug")


def load_config(path: Path = CONFIG, **overrides) -> dict:
    return yaml.safe_load(path.read_text()) | overrides


def _type(name: str) -> str:
    return name.replace("/", "_")


@dataclass
class Graph:
    data: HeteroData
    index: dict[str, np.ndarray]  # node type -> sorted global node_index; position = local id

    def local(self, node_type: str, global_ids: np.ndarray) -> torch.Tensor:
        idx = self.index[node_type]
        pos = np.searchsorted(idx, global_ids)
        assert (idx[pos] == global_ids).all(), f"unknown {node_type} ids"
        return torch.from_numpy(pos)

    def set_indications(self, pairs: pl.DataFrame) -> None:
        drug = self.local("drug", pairs["drug"].to_numpy())
        disease = self.local("disease", pairs["disease"].to_numpy())
        self.data[IND].edge_index = torch.stack([drug, disease])
        self.data[REV_IND].edge_index = torch.stack([disease, drug])


def build_graph(nodes: pl.DataFrame, split: Split) -> Graph:
    """Message-passing graph from split.graph minus INDICATION edges (set per epoch)."""
    data = HeteroData()
    index = {}
    for (t,), g in nodes.group_by("node_type"):
        index[_type(t)] = np.sort(g["node_index"].to_numpy())
        data[_type(t)].num_nodes = len(index[_type(t)])
    graph = Graph(data, index)
    edges = split.graph.filter(pl.col("rel_type") != "INDICATION")
    for (rel, s, d), g in edges.group_by(["rel_type", "src_type", "dst_type"]):
        s, d = _type(s), _type(d)
        src = graph.local(s, g["src_index"].to_numpy())
        dst = graph.local(d, g["dst_index"].to_numpy())
        data[s, rel, d].edge_index = torch.stack([src, dst])
        data[d, f"rev_{rel}", s].edge_index = torch.stack([dst, src])
    graph.set_indications(split.train)
    return graph


def text_features(graph: Graph, path: Path = TEXT) -> dict[str, torch.Tensor]:
    """Frozen text embeddings aligned to local ids; zeros where a node has no text."""
    raw = np.load(path)
    out = {}
    for t in ("drug", "disease"):
        emb = np.zeros((len(graph.index[t]), raw[f"{t}_emb"].shape[1]), dtype=np.float32)
        ids = raw[f"{t}_index"]
        keep = np.isin(ids, graph.index[t])
        emb[graph.local(t, ids[keep]).numpy()] = raw[f"{t}_emb"][keep]
        out[t] = torch.from_numpy(emb)
    return out


class HeteroGNN(nn.Module):
    def __init__(self, graph: Graph, cfg: dict, text: dict[str, torch.Tensor] | None = None):
        super().__init__()
        h = cfg["hidden"]
        data = graph.data
        # Without a disease ID embedding, unseen diseases are represented only by their text and
        # neighbourhood, which is what the disease-held-out split tests.
        learned = [t for t in data.node_types if t != "disease" or cfg["disease_id_embedding"]]
        self.emb = nn.ModuleDict({t: nn.Embedding(data[t].num_nodes, h) for t in learned})
        self.num_nodes = {t: data[t].num_nodes for t in data.node_types}
        self.hidden = h
        self.text = text or {}
        self.proj = nn.ModuleDict({t: nn.Linear(x.shape[1], h) for t, x in self.text.items()})
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv((-1, -1), h) for et in data.edge_types}, aggr="sum")
            for _ in range(cfg["layers"])
        )
        self.dropout = nn.Dropout(cfg["dropout"])
        self.rel = nn.Parameter(torch.ones(h))
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.1)

    def encode(self, edge_index_dict) -> dict[str, torch.Tensor]:
        x = {
            t: self.emb[t].weight if t in self.emb else torch.zeros(n, self.hidden)
            for t, n in self.num_nodes.items()
        }
        for t, feats in self.text.items():
            x[t] = x[t] + self.proj[t](feats)
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index_dict)
            x = {t: out.get(t, x[t]) + x[t] for t in x}  # residual keeps isolated types
            if i < len(self.convs) - 1:
                x = {t: self.dropout(torch.relu(v)) for t, v in x.items()}
        return x

    def decode(self, h_drug: torch.Tensor, h_disease: torch.Tensor) -> torch.Tensor:
        return (h_drug * self.rel * h_disease).sum(-1)


def score_fn(model: HeteroGNN, graph: Graph, drugs: np.ndarray) -> ScoreFn:
    model.eval()
    with torch.no_grad():
        h = model.encode(graph.data.edge_index_dict)
        weighted = h["drug"][graph.local("drug", drugs)] * model.rel

    def score(chunk: np.ndarray) -> np.ndarray:
        return (h["disease"][graph.local("disease", chunk)] @ weighted.T).numpy()

    return score


def train(
    split: Split,
    nodes: pl.DataFrame,
    drugs: np.ndarray,
    cfg: dict,
    text_path: Path | None = TEXT,
    log=print,
) -> tuple[HeteroGNN, Graph, dict]:
    torch.manual_seed(split.seed)
    rng = np.random.default_rng(split.seed)
    graph = build_graph(nodes, split)
    text = text_features(graph, text_path) if cfg["text"] and text_path else None
    model = HeteroGNN(graph, cfg, text)
    with torch.no_grad():  # materialise lazy SAGEConv shapes before building the optimiser
        model.encode(graph.data.edge_index_dict)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    all_drugs = torch.arange(len(graph.index["drug"]))
    best, best_mrr, bad, history = None, -1.0, 0, []
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        mask = rng.random(split.train.height) < cfg["supervision_ratio"]
        graph.set_indications(split.train.filter(~pl.Series(mask)))
        sup = split.train.filter(pl.Series(mask))
        drug = graph.local("drug", sup["drug"].to_numpy())
        disease = graph.local("disease", sup["disease"].to_numpy())
        k = cfg["negatives"]
        neg_drug = all_drugs[torch.randint(len(all_drugs), (len(drug) * k,))]
        h = model.encode(graph.data.edge_index_dict)
        pos = model.decode(h["drug"][drug], h["disease"][disease])
        neg = model.decode(h["drug"][neg_drug], h["disease"][disease.repeat(k)])
        logits = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            graph.set_indications(split.train)
            mrr = evaluate(score_fn(model, graph, drugs), drugs, split.valid, split.train)["mrr"]
            history.append({"epoch": epoch, "loss": float(loss), "valid_mrr": mrr})
            log(f"  epoch {epoch:4d} loss={float(loss):.4f} valid_mrr={mrr:.4f}")
            if mrr > best_mrr:
                best, best_mrr, bad = copy.deepcopy(model.state_dict()), mrr, 0
            elif (bad := bad + 1) >= cfg["patience"]:
                break
    graph.set_indications(split.train)
    if best is not None:
        model.load_state_dict(best)
    info = {
        "best_valid_mrr": best_mrr,
        "epochs_run": epoch,
        "train_seconds": time.perf_counter() - start,
        "history": history,
    }
    return model, graph, info


def save_checkpoint(model: HeteroGNN, cfg: dict, info: dict, name: str) -> Path:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg, "info": info}, path)
    return path


def load_checkpoint(
    path: Path, split: Split, nodes: pl.DataFrame, text_path: Path | None = TEXT
) -> tuple[HeteroGNN, Graph]:
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    graph = build_graph(nodes, split)
    text = text_features(graph, text_path) if cfg["text"] and text_path else None
    model = HeteroGNN(graph, cfg, text)
    with torch.no_grad():
        model.encode(graph.data.edge_index_dict)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), graph


NODES = Path("data/processed/nodes.parquet")


def _model(text: bool):
    def fit(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
        cfg = load_config(text=text)
        model, graph, info = train(split, pl.read_parquet(NODES), drugs, cfg)
        name = f"{split.name}_seed{split.seed}_{'text' if text else 'notext'}"
        save_checkpoint(model, cfg, info, name)
        return score_fn(model, graph, drugs)

    return fit


gnn = _model(text=True)
gnn_notext = _model(text=False)
