"""Which model should rank diseases with no known drugs? Subgroup evaluation on the
held-out-disease splits, by whether the disease has gene/phenotype links (the only signal
`similar_disease` uses). Picks per subgroup on mean validation MRR over `disease_strict` seeds.

    uv run --extra ml python -m ml.routing      # writes ml/results/routing.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ml.baselines import DISEASE_NEIGHBOURS, popularity, similar_disease
from ml.evaluate import ScoreFn, evaluate
from ml.splits import Split

SPLITS = ("disease", "disease_strict")
CANDIDATES = ("popularity", "similar_disease", "gnn", "gnn_notext", "gnn_dmask")
SUBGROUPS = ("with_genes", "without_genes")
OUT = Path(__file__).with_name("results") / "routing.json"


def has_genes(graph: pl.DataFrame) -> set[int]:
    """Diseases with at least one DISEASE_PROTEIN / DISEASE_PHENOTYPE_POSITIVE edge."""
    rows = graph.filter(pl.col("rel_type").is_in(DISEASE_NEIGHBOURS))
    return set(rows["src_index"].unique().to_list())


def subgroup(pairs: pl.DataFrame, genes: set[int], name: str) -> pl.DataFrame:
    inside = pl.col("disease").is_in(list(genes))
    return pairs.filter(inside if name == "with_genes" else ~inside)


def pick(results: dict, split: str = "disease_strict") -> dict[str, str]:
    """Best candidate per subgroup by mean validation MRR (models missing on any seed skipped)."""
    best = {}
    for group in SUBGROUPS:
        means = {
            m: float(np.mean([r[group]["valid"]["mrr"] for r in runs]))
            for m, runs in results[split].items()
            if all(group in r and r[group]["valid"]["pairs"] for r in runs)
        }
        best[group] = max(means, key=means.get) if means else "similar_disease"
    return best


def _scorers(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> dict[str, ScoreFn]:
    from ml.gnn import CHECKPOINTS, NODES, load_checkpoint, score_fn

    out = {
        "popularity": popularity(split, drugs, diseases),
        "similar_disease": similar_disease(split, drugs, diseases),
    }
    nodes = pl.read_parquet(NODES)
    for name, tag in (("gnn", "text"), ("gnn_notext", "notext"), ("gnn_dmask", "dmask")):
        path = CHECKPOINTS / f"{split.name}_seed{split.seed}_{tag}.pt"
        if path.exists():
            model, graph = load_checkpoint(path, split, nodes)
            out[name] = score_fn(model, graph, drugs)
    return out


def main() -> None:
    from ml.run import get_split

    processed = Path("data/processed")
    nodes = pl.read_parquet(processed / "nodes.parquet")
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    results: dict = {}
    for name in SPLITS:
        results[name] = {}
        for seed in (0, 1, 2):
            split = get_split(name, seed, processed, processed / "splits")
            genes = has_genes(split.graph)
            known_valid, known_test = split.train, pl.concat([split.train, split.valid])
            for model, score in _scorers(split, drugs, diseases).items():
                row = {"seed": seed}
                for group in SUBGROUPS:
                    row[group] = {}
                    for part, pairs, known in (
                        ("valid", split.valid, known_valid),
                        ("test", split.test, known_test),
                    ):
                        sub = subgroup(pairs, genes, group)
                        m = evaluate(score, drugs, sub, known, seed=seed) if sub.height else {}
                        row[group][part] = {
                            "pairs": sub.height,
                            "diseases": sub["disease"].n_unique(),
                            **{k: m[k] for k in ("mrr", "hits@50") if k in m},
                        }
                results[name].setdefault(model, []).append(row)
                print(name, seed, model, json.dumps({g: row[g]["test"] for g in SUBGROUPS}))
    results["picked"] = pick(results)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print("picked", results["picked"])


if __name__ == "__main__":
    main()
