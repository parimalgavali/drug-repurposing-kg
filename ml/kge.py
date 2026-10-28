"""PyKEEN knowledge-graph embeddings (RotatE, ComplEx) trained on the split's training graph
(CPU), with early stopping on validation MRR from ml.evaluate."""

from __future__ import annotations

import copy
import os

import numpy as np
import polars as pl
import torch
from pykeen.models import ComplEx, RotatE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

from ml.evaluate import ScoreFn, evaluate
from ml.splits import Split

EPOCHS = int(os.environ.get("KGE_EPOCHS", "30"))
EVAL_EVERY = 5
PATIENCE = 2


def _factory(split: Split) -> TriplesFactory:
    g = split.graph
    triples = np.stack(
        [
            g["src_index"].cast(pl.String).to_numpy(),
            g["rel_type"].to_numpy(),
            g["dst_index"].cast(pl.String).to_numpy(),
        ],
        axis=1,
    ).astype(str)
    return TriplesFactory.from_labeled_triples(triples)


def _score_fn(model, tf: TriplesFactory, drugs: np.ndarray) -> ScoreFn:
    rel = tf.relation_to_id["INDICATION"]
    ent = tf.entity_to_id
    cols = np.array([ent.get(str(d), -1) for d in drugs])
    present = cols >= 0

    @torch.no_grad()
    def score(chunk: np.ndarray) -> np.ndarray:
        out = np.full((len(chunk), len(drugs)), -1e9)
        tails = np.array([ent.get(str(d), -1) for d in chunk])
        ok = tails >= 0
        if ok.any():
            rt = torch.tensor(np.stack([np.full(ok.sum(), rel), tails[ok]], axis=1))
            s = model.score_h(rt).cpu().numpy()
            out[np.ix_(ok, present)] = s[:, cols[present]]
        return out

    return score


def _fit(cls, split: Split, drugs: np.ndarray, **kwargs) -> ScoreFn:
    torch.manual_seed(split.seed)
    tf = _factory(split)
    model = cls(triples_factory=tf, random_seed=split.seed, **kwargs)
    loop = SLCWATrainingLoop(
        model=model,
        triples_factory=tf,
        optimizer="Adam",
        optimizer_kwargs={"lr": 1e-3},
        negative_sampler_kwargs={"num_negs_per_pos": 4},
    )
    best, best_mrr, bad = None, -1.0, 0
    for step in range(0, EPOCHS, EVAL_EVERY):
        loop.train(
            triples_factory=tf,
            num_epochs=EVAL_EVERY,
            batch_size=8192,
            use_tqdm=False,
            continue_training=step > 0,
        )
        model.eval()
        mrr = evaluate(_score_fn(model, tf, drugs), drugs, split.valid, split.train)["mrr"]
        model.train()
        if mrr > best_mrr:
            best, best_mrr, bad = copy.deepcopy(model.state_dict()), mrr, 0
        elif (bad := bad + 1) >= PATIENCE:
            break
    model.load_state_dict(best)
    model.eval()
    return _score_fn(model, tf, drugs)


def rotate(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
    return _fit(RotatE, split, drugs, embedding_dim=64)


def complex_(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
    return _fit(ComplEx, split, drugs, embedding_dim=64)
