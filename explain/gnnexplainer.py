"""GNNExplainer on the served GNN, compared with the DWPC metapath explanations.

For each sampled served prediction (rank-1 PREDICTED_INDICATION of a random disease), PyG's
GNNExplainer learns a soft mask over the edges that feed the pair's DistMult logit. Its top
edges are compared with the edges on the top metapath example paths from `explain.paths`, and
both edge sets are deleted from the message-passing graph to see how much the logit drops.

    uv run --extra ml --extra graph python -m explain.gnnexplainer
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from ml.gnn import Graph, HeteroGNN

EdgeType = tuple[str, str, str]
Key = tuple[str, int, int]  # (relation without "rev_", smaller node id, larger node id)

CHECKPOINT = Path("artifacts/gnn/random_seed0_notext.pt")
RESULTS = Path("ml/results/gnnexplainer.json")
MODEL_CARD = Path("ml/model_card.md")
KS = (10, 20)
SERVED = """
MATCH (d:Drug)-[r:PREDICTED_INDICATION {rank: 1}]->(s:Disease)
RETURN d.id AS drug, d.name AS drug_name, s.id AS disease, s.name AS disease_name
ORDER BY s.id
"""


class PairScore(nn.Module):
    """The served model's message passing + DistMult as f(x, edges) -> logit, for the explainer.

    `x` is the input embedding (`initial_features`), so edge masks are the only thing learned.
    """

    def __init__(self, model: HeteroGNN):
        super().__init__()
        self.model = model
        self.fixed: dict[EdgeType, torch.Tensor] = {}  # unmasked edges, e.g. empty edge types

    def forward(self, x, edge_index, pair_drug, pair_disease):
        edge_index = self.fixed | edge_index
        for i, conv in enumerate(self.model.convs):
            out = conv(x, edge_index)
            x = {t: out.get(t, x[t]) + x[t] for t in x}
            if i < len(self.model.convs) - 1:
                x = {t: torch.relu(v) for t, v in x.items()}
        return self.model.decode(x["drug"][pair_drug], x["disease"][pair_disease]).view(1)


def initial_features(model: HeteroGNN) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        x = {
            t: model.emb[t].weight.clone() if t in model.emb else torch.zeros(n, model.hidden)
            for t, n in model.num_nodes.items()
        }
        for t, feats in model.text.items():
            x[t] = x[t] + model.proj[t](feats)
    return x


def computation_edges(
    edge_index: dict[EdgeType, torch.Tensor], targets: dict[str, list[int]], layers: int = 2
) -> dict[EdgeType, torch.Tensor]:
    """Every edge that can reach `targets` within `layers` hops of message passing.

    Keeping all in-edges of each node in the receptive field (and every edge type into a
    receptive-field node type) makes the pair's logit identical to the full graph's.
    """
    field = {t: torch.tensor(sorted(ids)) for t, ids in targets.items()}
    for _ in range(layers - 1):
        grown = {t: [v] for t, v in field.items()}
        for (src, _, dst), e in edge_index.items():
            if dst in field:
                grown.setdefault(src, []).append(e[0][torch.isin(e[1], field[dst])])
        field = {t: torch.unique(torch.cat(v)) for t, v in grown.items()}
    # Edge types stay even when empty: SAGEConv adds a root term per edge type into its dst type.
    return {
        et: e[:, torch.isin(e[1], field[et[2]])] for et, e in edge_index.items() if et[2] in field
    }


def edge_key(rel: str, a: int, b: int) -> Key:
    return (rel.removeprefix("rev_"), min(a, b), max(a, b))


def global_keys(graph: Graph, et: EdgeType, e: torch.Tensor) -> list[Key]:
    src = graph.index[et[0]][e[0].numpy()]
    dst = graph.index[et[2]][e[1].numpy()]
    return [edge_key(et[1], int(a), int(b)) for a, b in zip(src, dst, strict=True)]


def gnn_edge_weights(
    masks: dict[EdgeType, torch.Tensor], edges: dict[EdgeType, torch.Tensor], graph: Graph
) -> dict[Key, float]:
    """Mask weight per undirected edge (max over its two directions); unreached edges dropped."""
    weights: dict[Key, float] = {}
    for et, mask in masks.items():
        for key, w in zip(global_keys(graph, et, edges[et]), mask.tolist(), strict=True):
            if w > 0:
                weights[key] = max(w, weights.get(key, 0.0))
    return weights


def path_edges(result: dict) -> set[Key]:
    """Edges on the example paths of the top metapaths returned by `explain.paths.explain`."""
    keys = set()
    for m in result["metapaths"]:
        for p in m["paths"]:
            for rel, a, b in zip(p["rels"], p["nodes"], p["nodes"][1:], strict=False):
                keys.add(edge_key(rel, a["id"], b["id"]))
    return keys


def overlap(ranked: list[Key], paths: set[Key], candidates: int, k: int) -> dict:
    """Top-k GNNExplainer edges vs path edges, with the overlap a random top-k would get."""
    hits = len(set(ranked[:k]) & paths)
    base = len(paths) / candidates if candidates else 0.0
    precision = hits / k
    return {
        "hits": hits,
        "precision": precision,
        "recall": hits / len(paths) if paths else 0.0,
        "random_precision": base,
        "lift": precision / base if base else None,
    }


def without(
    edges: dict[EdgeType, torch.Tensor], graph: Graph, drop: set[Key]
) -> dict[EdgeType, torch.Tensor]:
    out = {}
    for et, e in edges.items():
        keep = torch.tensor([k not in drop for k in global_keys(graph, et, e)], dtype=torch.bool)
        out[et] = e[:, keep]
    return out


def explain_pair(
    scorer: PairScore,
    x: dict[str, torch.Tensor],
    graph: Graph,
    drug: int,
    disease: int,
    paths: set[Key],
    epochs: int = 100,
    seed: int = 0,
) -> dict:
    from torch_geometric.explain import Explainer, GNNExplainer

    d, s = graph.local("drug", np.array([drug])), graph.local("disease", np.array([disease]))
    field = computation_edges(graph.data.edge_index_dict, {"drug": [int(d)], "disease": [int(s)]})
    # GNNExplainer can't mask empty edge types, but the model still needs them for exactness.
    scorer.fixed = {et: e for et, e in field.items() if e.shape[1] == 0}
    edges = {et: e for et, e in field.items() if e.shape[1] > 0}
    torch.manual_seed(seed)
    explainer = Explainer(
        scorer,
        GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type="model",
        edge_mask_type="object",
        node_mask_type=None,
        model_config={"mode": "binary_classification", "task_level": "graph", "return_type": "raw"},
    )
    explanation = explainer(x, edges, pair_drug=d, pair_disease=s)
    masks = {et: explanation[et].edge_mask for et in edges if "edge_mask" in explanation[et]}
    weights = gnn_edge_weights(masks, edges, graph)
    ranked = sorted(weights, key=lambda k: (-weights[k], k))
    covered = paths & weights.keys()
    rng = np.random.default_rng(seed)
    pool = list(weights)
    random_drop = {pool[i] for i in rng.choice(len(pool), len(covered), replace=False)}
    with torch.no_grad():
        logit = float(scorer(x, edges, d, s))
        drop = {
            "paths": float(scorer(x, without(edges, graph, covered), d, s)),
            "gnnexplainer": float(
                scorer(x, without(edges, graph, set(ranked[: len(covered)])), d, s)
            ),
            "random": float(scorer(x, without(edges, graph, random_drop), d, s)),
        }
    return {
        "candidate_edges": len(weights),
        "path_edges": len(paths),
        "path_edges_in_field": len(covered),
        "overlap": {f"top{k}": overlap(ranked, covered, len(weights), k) for k in KS},
        "logit": logit,
        "logit_after_removing": drop,
        "top_edges": [
            {"rel": k[0], "a": k[1], "b": k[2], "weight": weights[k]} for k in ranked[:20]
        ],
    }


def summarise(pairs: list[dict]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    out = {"pairs": len(pairs)}
    for k in KS:
        rows = [p["overlap"][f"top{k}"] for p in pairs]
        out[f"top{k}"] = {
            m: mean(r[m] for r in rows) for m in ("precision", "recall", "random_precision", "lift")
        } | {"pairs_with_hit": sum(r["hits"] > 0 for r in rows)}
    out["mean_logit_drop"] = {
        name: mean(p["logit"] - p["logit_after_removing"][name] for p in pairs)
        for name in ("paths", "gnnexplainer", "random")
    }
    top_rels = Counter(e["rel"] for p in pairs for e in p["top_edges"])
    path_rels = Counter(rel for p in pairs for rel in p["path_rels"])
    out["top20_relations"] = dict(top_rels.most_common())
    out["path_relations"] = dict(path_rels.most_common())
    return out


def card_section(results: dict) -> str:
    s = results["summary"]
    lines = [
        f"{s['pairs']} served rank-1 predictions, {results['epochs']} GNNExplainer epochs each.",
        "",
        "| Top-k GNNExplainer edges | Share on metapath paths | Random top-k | Lift "
        "| Path edges recovered | Pairs with ≥1 shared edge |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in KS:
        r = s[f"top{k}"]
        lift = f"{r['lift']:.1f}×" if r["lift"] is not None else "–"
        lines.append(
            f"| {k} | {r['precision']:.3f} | {r['random_precision']:.4f} | {lift} "
            f"| {r['recall']:.3f} | {r['pairs_with_hit']}/{s['pairs']} |"
        )
    d = s["mean_logit_drop"]
    lines += [
        "",
        "Mean drop of the pair's logit when the edges are deleted from message passing "
        "(same number of edges each):",
        "",
        "| Deleted edges | Mean logit drop |",
        "| --- | ---: |",
        f"| metapath path edges | {d['paths']:.3f} |",
        f"| top GNNExplainer edges | {d['gnnexplainer']:.3f} |",
        f"| random receptive-field edges | {d['random']:.3f} |",
        "",
        "Relations of the top-20 GNNExplainer edges (all pairs): "
        + ", ".join(f"{r} {n}" for r, n in list(s["top20_relations"].items())[:8]),
        "",
        "Relations of the metapath path edges: "
        + ", ".join(f"{r} {n}" for r, n in list(s["path_relations"].items())[:8]),
    ]
    return "\n".join(lines)


def update_card(section: str, card: Path = MODEL_CARD) -> None:
    start, end = "<!-- gnnexplainer:start -->", "<!-- gnnexplainer:end -->"
    text = card.read_text()
    if start not in text:
        text += f"\n## GNNExplainer vs metapath explanations\n\n{start}\n{end}\n"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    card.write_text(f"{head}{start}\n{section}\n{end}{tail}")


def main() -> None:
    import polars as pl

    from explain.paths import explain
    from kg.client import KGClient
    from ml.gnn import load_checkpoint
    from ml.run import get_split

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    split = get_split("random", 0, args.processed, args.processed / "splits")
    model, graph = load_checkpoint(CHECKPOINT, split, nodes, None)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = PairScore(model).eval()
    x = initial_features(model)
    names = dict(zip(nodes["node_index"].to_list(), nodes["node_name"].to_list(), strict=True))
    rng = np.random.default_rng(args.seed)
    pairs = []
    with KGClient() as kg:
        served = kg.read(SERVED)
        for i in rng.permutation(len(served)):
            if len(pairs) == args.pairs:
                break
            row = served[int(i)]
            result = explain(kg, row["drug"], row["disease"], top=3, examples=20)
            if not result["metapaths"]:
                continue
            paths = path_edges(result)
            pair = explain_pair(
                scorer, x, graph, row["drug"], row["disease"], paths, args.epochs, args.seed
            )
            pair |= {
                **row,
                "path_rels": [k[0] for k in paths],
                "top_metapath": result["metapaths"][0]["metapath"],
                "top_path": result["metapaths"][0]["paths"][0]["text"],
            }
            for e in pair["top_edges"]:
                e["text"] = f"{names[e['a']]} -{e['rel']}- {names[e['b']]}"
            r20 = pair["overlap"]["top20"]
            print(
                f"{row['drug_name']} → {row['disease_name']}: {pair['candidate_edges']} edges, "
                f"top20 on paths {r20['hits']}, logit {pair['logit']:.2f}",
                flush=True,
            )
            pairs.append(pair)
    results = {"model": CHECKPOINT.stem, "epochs": args.epochs, "summary": summarise(pairs)}
    results["pairs"] = pairs
    RESULTS.write_text(json.dumps(results, indent=2))
    update_card(card_section(results))
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
