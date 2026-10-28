"""Non-learned baselines: drug popularity and Adamic–Adar over shared genes/phenotypes."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from ml.evaluate import ScoreFn
from ml.splits import Split

DRUG_NEIGHBOURS = ("TARGET", "ENZYME", "TRANSPORTER", "CARRIER", "DRUG_EFFECT")
DISEASE_NEIGHBOURS = ("DISEASE_PROTEIN", "DISEASE_PHENOTYPE_POSITIVE")


def popularity(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
    counts = split.train.group_by("drug").len()
    lookup = dict(zip(counts["drug"], counts["len"], strict=True))
    vec = np.array([lookup.get(int(d), 0) for d in drugs], dtype=np.float64)
    return lambda chunk: np.broadcast_to(vec, (len(chunk), len(drugs)))


def shared_neighbour(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
    """score(drug, disease) = Σ_z 1/log(deg z) over genes/phenotypes z linked to both."""
    g = split.graph
    drug_z = g.filter(pl.col("rel_type").is_in(DRUG_NEIGHBOURS)).select(
        pl.col("src_index").alias("a"), pl.col("dst_index").alias("z")
    )
    dis_z = g.filter(pl.col("rel_type").is_in(DISEASE_NEIGHBOURS)).select(
        pl.col("src_index").alias("a"), pl.col("dst_index").alias("z")
    )
    zs = np.unique(np.r_[drug_z["z"].to_numpy(), dis_z["z"].to_numpy()])
    z_pos = {int(z): i for i, z in enumerate(zs)}
    deg = np.zeros(len(zs))
    for df in (drug_z, dis_z):
        np.add.at(deg, [z_pos[z] for z in df["z"]], 1)
    weight = 1 / np.log(np.maximum(deg, 2))

    def matrix(df: pl.DataFrame, rows: np.ndarray) -> sp.csr_matrix:
        row_pos = {int(r): i for i, r in enumerate(rows)}
        df = df.unique().filter(pl.col("a").is_in(rows.tolist()))
        r = [row_pos[a] for a in df["a"]]
        c = [z_pos[z] for z in df["z"]]
        return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), len(zs)))

    drug_m = (matrix(drug_z, drugs) @ sp.diags(weight)).T.tocsc()
    dis_m = matrix(dis_z, diseases)
    dis_pos = {int(d): i for i, d in enumerate(diseases)}
    return lambda chunk: (dis_m[[dis_pos[int(d)] for d in chunk]] @ drug_m).toarray()


def similar_disease(split: Split, drugs: np.ndarray, diseases: np.ndarray) -> ScoreFn:
    """Collaborative filter: Adamic–Adar disease–disease similarity over shared genes/phenotypes,
    times the training indications of the similar diseases. Popularity breaks ties."""
    g = split.graph.filter(pl.col("rel_type").is_in(DISEASE_NEIGHBOURS))
    zs = np.unique(g["dst_index"].to_numpy())
    z_pos = {int(z): i for i, z in enumerate(zs)}
    dis_pos = {int(d): i for i, d in enumerate(diseases)}
    drug_pos = {int(d): i for i, d in enumerate(drugs)}
    rows = [dis_pos[a] for a in g["src_index"]]
    cols = [z_pos[z] for z in g["dst_index"]]
    x = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(diseases), len(zs)))
    x.data[:] = 1
    weight = sp.diags(1 / np.log(np.maximum(np.asarray(x.sum(0)).ravel(), 2)))
    sim = (x @ weight @ x.T).tocsr()
    sim.setdiag(0)
    t = split.train
    y = sp.csr_matrix(
        (
            np.ones(t.height),
            ([dis_pos[d] for d in t["disease"]], [drug_pos[d] for d in t["drug"]]),
        ),
        shape=(len(diseases), len(drugs)),
    )
    pop = np.asarray(y.sum(0)).ravel()
    return lambda chunk: (sim[[dis_pos[int(d)] for d in chunk]] @ y).toarray() + 1e-6 * pop
