"""Build splits, score models, write metrics JSON and refresh the model card table.

uv run --extra ml python -m ml.run --models popularity,shared_neighbour --split disease
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from ml.baselines import popularity, shared_neighbour, similar_disease
from ml.evaluate import ScoreFn, evaluate
from ml.splits import SPLITS, Split, disease_groups, load_split, make_split, save_split, split_dir

Model = Callable[[Split, np.ndarray, np.ndarray], ScoreFn]
RESULTS = Path(__file__).with_name("results")
MODEL_CARD = Path(__file__).with_name("model_card.md")
METRICS = ("mrr", "hits@1", "hits@10", "hits@50", "auroc", "auprc")


def models() -> dict[str, Model]:
    registry: dict[str, Model] = {
        "popularity": popularity,
        "shared_neighbour": shared_neighbour,
        "similar_disease": similar_disease,
    }
    try:
        from ml.blend import gnn_blend
        from ml.gnn import gnn, gnn_dmask, gnn_notext, gnn_pool
        from ml.kge import complex_, rotate
    except ImportError:  # torch/pykeen/pyg not installed
        return registry
    return (
        registry
        | {"rotate": rotate, "complex": complex_, "gnn": gnn, "gnn_notext": gnn_notext}
        | {"gnn_blend": gnn_blend, "gnn_dmask": gnn_dmask, "gnn_pool": gnn_pool}
    )


def get_split(name: str, seed: int, processed: Path, root: Path) -> Split:
    if (split_dir(root, name, seed) / "test.parquet").exists():
        return load_split(root, name, seed)
    edges = pl.read_parquet(processed / "edges.parquet")
    nodes = pl.read_parquet(processed / "nodes.parquet")
    features = pl.read_parquet(processed / "disease_features.parquet")
    split = make_split(name, edges, disease_groups(nodes, features), seed)
    save_split(split, root)
    return split


def summarise(runs: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        vals = np.array([r[m] for r in runs])
        out[m] = {"mean": float(vals.mean()), "std": float(vals.std())}
    out["train_seconds"] = float(np.mean([r["seconds"] for r in runs]))
    out["test_pairs"] = int(np.mean([r["test_pairs"] for r in runs]))
    out["test_diseases"] = int(np.mean([r["test_diseases"] for r in runs]))
    out["seeds"] = [r["seed"] for r in runs]
    if "alpha" in runs[0]:
        out["alpha"] = [r["alpha"] for r in runs]
    return out


def card_table(results_dir: Path = RESULTS) -> str:
    header = (
        "| Split | Model | MRR | Hits@1 | Hits@10 | Hits@50 | AUROC | AUPRC | Fit+score s | Seeds |"
    )
    lines = [header, "|" + " --- |" * 10]
    for path in sorted(p for p in results_dir.glob("*.json") if p.stem in SPLITS):
        split = path.stem
        for model, r in json.loads(path.read_text()).items():
            cells = [f"{r[m]['mean']:.3f} ± {r[m]['std']:.3f}" for m in METRICS]
            seeds = len(r["seeds"])
            lines.append(
                f"| {split} | {model} | {' | '.join(cells)} | {r['train_seconds']:.0f} | {seeds} |"
            )
    return "\n".join(lines)


def update_card(table: str, card: Path = MODEL_CARD) -> None:
    start, end = "<!-- metrics:start -->", "<!-- metrics:end -->"
    text = card.read_text()
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    card.write_text(f"{head}{start}\n{table}\n{end}{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="popularity,shared_neighbour")
    parser.add_argument("--split", default="disease", choices=[*SPLITS, "all"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    registry = models()
    names = args.models.split(",")
    if missing := [n for n in names if n not in registry]:
        raise SystemExit(f"unknown/unavailable models: {missing}; have {sorted(registry)}")
    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    RESULTS.mkdir(exist_ok=True)

    for split_name in SPLITS if args.split == "all" else (args.split,):
        path = RESULTS / f"{split_name}.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        for name in names:
            runs = []
            for seed in map(int, args.seeds.split(",")):
                split = get_split(split_name, seed, args.processed, args.processed / "splits")
                start = time.perf_counter()
                score = registry[name](split, drugs, diseases)
                known = pl.concat([split.train, split.valid])
                metrics = evaluate(score, drugs, split.test, known, seed=seed)
                metrics |= {"seconds": time.perf_counter() - start, "seed": seed}
                metrics |= getattr(score, "info", {})
                print(
                    f"{split_name:8s} {name:18s} seed={seed} "
                    + " ".join(f"{m}={metrics[m]:.3f}" for m in METRICS),
                    flush=True,
                )
                runs.append(metrics)
            results[name] = summarise(runs)
            path.write_text(json.dumps(results, indent=2) + "\n")
    update_card(card_table())


if __name__ == "__main__":
    main()
