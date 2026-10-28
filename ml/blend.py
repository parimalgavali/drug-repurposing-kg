"""GNN + similar-disease blend. Per disease, both score rows are z-scored and mixed as
α·gnn + (1 − α)·similar_disease; α is picked on validation MRR, never on test."""

from __future__ import annotations

import numpy as np
import polars as pl

from ml.baselines import similar_disease
from ml.evaluate import ScoreFn, evaluate
from ml.splits import Split

ALPHAS = tuple(round(a, 1) for a in np.linspace(0, 1, 11))


def zscore(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    std = rows.std(axis=1, keepdims=True)
    return (rows - rows.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)


def mix(a: ScoreFn, b: ScoreFn, alpha: float) -> ScoreFn:
    return lambda chunk: alpha * zscore(a(chunk)) + (1 - alpha) * zscore(b(chunk))


def pick_alpha(
    a: ScoreFn,
    b: ScoreFn,
    drugs: np.ndarray,
    valid: pl.DataFrame,
    known: pl.DataFrame,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[float, dict[float, float]]:
    mrr = {al: evaluate(mix(a, b, al), drugs, valid, known)["mrr"] for al in alphas}
    return max(mrr, key=lambda al: (mrr[al], -abs(al - 0.5))), mrr


def _gnn(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
    from ml.gnn import CHECKPOINTS, NODES, gnn, load_checkpoint, score_fn

    path = CHECKPOINTS / f"{split.name}_seed{split.seed}_text.pt"
    if not path.exists():
        return gnn(split, drugs, diseases)
    model, graph = load_checkpoint(path, split, pl.read_parquet(NODES))
    return score_fn(model, graph, drugs)


def gnn_blend(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
    g, s = _gnn(split, drugs, diseases), similar_disease(split, drugs, diseases)
    alpha, curve = pick_alpha(g, s, drugs, split.valid, split.train)
    print(f"  alpha={alpha} valid_mrr " + " ".join(f"{k}:{v:.3f}" for k, v in curve.items()))
    score = mix(g, s, alpha)
    score.info = {"alpha": alpha}  # type: ignore[attr-defined]
    return score
