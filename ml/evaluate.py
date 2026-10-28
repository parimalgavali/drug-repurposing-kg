"""Filtered ranking of all drugs per test disease (MRR, Hits@k) plus AUROC/AUPRC on 1:1
sampled negatives. Ties count half (mean rank), so constant scores don't look good."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

# disease node indexes (n,) -> scores (n, len(drugs)), columns aligned with `drugs`
ScoreFn = Callable[[np.ndarray], np.ndarray]


def _by_disease(pairs: pl.DataFrame, drug_pos: dict[int, int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for (disease,), g in pairs.group_by("disease"):
        out[disease] = np.array([drug_pos[d] for d in g["drug"]], dtype=np.int64)
    return out


def evaluate(
    score: ScoreFn,
    drugs: np.ndarray,
    test: pl.DataFrame,
    known: pl.DataFrame,
    ks: tuple[int, ...] = (1, 10, 50),
    seed: int = 0,
    batch: int = 256,
) -> dict[str, float]:
    drug_pos = {int(d): i for i, d in enumerate(drugs)}
    test_by = _by_disease(test, drug_pos)
    known_by = _by_disease(known, drug_pos)
    rng = np.random.default_rng(seed)
    diseases = np.array(sorted(test_by), dtype=np.int64)
    ranks: list[float] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for start in range(0, len(diseases), batch):
        chunk = diseases[start : start + batch]
        scores = np.asarray(score(chunk), dtype=np.float64)
        for row, disease in zip(scores, chunk, strict=True):
            pos = test_by[int(disease)]
            negative = np.ones(len(drugs), dtype=bool)
            negative[pos] = False
            negative[known_by.get(int(disease), [])] = False
            neg = row[negative]
            for s in row[pos]:
                ranks.append(1 + (neg > s).sum() + 0.5 * (neg == s).sum())
            pos_scores += row[pos].tolist()
            neg_scores += row[rng.choice(np.flatnonzero(negative), size=len(pos))].tolist()
    r = np.array(ranks)
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    s = np.r_[pos_scores, neg_scores]
    metrics = {"mrr": float(np.mean(1 / r))}
    metrics |= {f"hits@{k}": float(np.mean(r <= k)) for k in ks}
    metrics |= {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
    metrics |= {"test_pairs": len(r), "test_diseases": len(diseases)}
    return metrics
