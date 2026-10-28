"""Train/valid/test splits of INDICATION edges plus the matching training graph.

- disease: hold out disease groups (PrimeKG `group_id_bert`, else the disease itself); every
  INDICATION / CONTRAINDICATION / OFF_LABEL_USE edge of a held-out disease leaves the graph.
- disease_strict: like disease, but indicated diseases joined by a DISEASE_DISEASE (MONDO) edge or
  a shared `group_id_bert` form one group, so no test disease has a direct ontology neighbour with
  training indications. Groups are held out until 20% / 10% of indicated diseases are reached.
- random: 80/10/10 over INDICATION pairs; the clinical edges of held-out pairs leave the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

CLINICAL = ("INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE")
PRUNED = ("DRUG_DRUG", "ANATOMY_PROTEIN_PRESENT", "ANATOMY_PROTEIN_ABSENT")
SPLITS = ("disease", "disease_strict", "random")
PAIR = ["drug", "disease"]


@dataclass(frozen=True)
class Split:
    name: str
    seed: int
    graph: pl.DataFrame  # training edges (rel_type, src_index, src_type, dst_index, dst_type)
    train: pl.DataFrame  # INDICATION pairs (drug, disease)
    valid: pl.DataFrame
    test: pl.DataFrame


def indications(edges: pl.DataFrame) -> pl.DataFrame:
    return (
        edges.filter(pl.col("rel_type") == "INDICATION")
        .select(pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease"))
        .sort(PAIR)
    )


def disease_groups(nodes: pl.DataFrame, features: pl.DataFrame | None = None) -> pl.DataFrame:
    diseases = nodes.filter(pl.col("node_type") == "disease").select("node_index")
    if features is None:
        return diseases.with_columns(pl.col("node_index").cast(pl.String).alias("group"))
    groups = diseases.join(
        features.select("node_index", "group_id_bert"), on="node_index", how="left"
    )
    fallback = pl.col("node_index").cast(pl.String)
    group = pl.when(pl.col("group_id_bert").fill_null("") == "").then(fallback)
    return groups.select("node_index", group.otherwise(pl.col("group_id_bert")).alias("group"))


def ontology_groups(edges: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """Connected components of indicated diseases linked by DISEASE_DISEASE or a shared group."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ind = np.unique(indications(edges)["disease"].to_numpy())
    pos = {int(d): i for i, d in enumerate(ind)}
    dd = edges.filter(
        (pl.col("rel_type") == "DISEASE_DISEASE")
        & pl.col("src_index").is_in(ind.tolist())
        & pl.col("dst_index").is_in(ind.tolist())
    )
    src = [pos[s] for s in dd["src_index"]]
    dst = [pos[d] for d in dd["dst_index"]]
    labelled = pl.DataFrame({"node_index": ind}).join(groups, on="node_index", how="left")
    labelled = labelled.with_columns(
        pl.col("group").fill_null(pl.col("node_index").cast(pl.String))
    )
    for (_,), g in labelled.group_by("group"):  # chain members of each group
        members = [pos[int(d)] for d in g["node_index"]]
        src += members[:-1]
        dst += members[1:]
    adj = coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(ind), len(ind)))
    _, label = connected_components(adj, directed=False)
    return pl.DataFrame({"node_index": ind, "group": [f"c{c}" for c in label]})


def ontology_leakage(edges: pl.DataFrame, split: Split) -> dict[str, float]:
    """Share of test pairs whose drug is a training indication of a 1-hop MONDO neighbour."""
    dd = edges.filter(pl.col("rel_type") == "DISEASE_DISEASE").select(
        pl.col("src_index").alias("disease"), pl.col("dst_index").alias("neighbour")
    )
    dd = pl.concat(
        [dd, dd.select(pl.col("neighbour").alias("disease"), pl.col("disease").alias("neighbour"))]
    )
    train = split.train.rename({"disease": "neighbour"})
    test_dis = split.test["disease"].unique()
    linked = dd.filter(pl.col("disease").is_in(test_dis.implode())).join(train, on="neighbour")
    leaked = split.test.join(linked.select(PAIR).unique(), on=PAIR, how="semi")
    has_nb = linked["disease"].unique()
    return {
        "test_pairs_leaked": leaked.height / split.test.height,
        "test_diseases_with_neighbour": len(has_nb) / len(test_dis),
    }


def _training_graph(edges: pl.DataFrame, held: pl.Expr, prune: bool) -> pl.DataFrame:
    graph = edges.filter(~(pl.col("rel_type").is_in(CLINICAL) & held))
    if prune:
        graph = graph.filter(~pl.col("rel_type").is_in(PRUNED))
    return graph


def disease_split(
    edges: pl.DataFrame,
    groups: pl.DataFrame,
    seed: int,
    test_frac: float = 0.2,
    valid_frac: float = 0.1,
    prune: bool = True,
    name: str = "disease",
    by_size: bool = False,
) -> Split:
    ind = indications(edges)
    labelled = ind.join(groups, left_on="disease", right_on="node_index", how="left")
    labelled = labelled.with_columns(pl.col("group").fill_null(pl.col("disease").cast(pl.String)))
    names = np.array(sorted(labelled["group"].unique()))
    perm = np.random.default_rng(seed).permutation(names)
    if by_size:  # uneven components: cut by cumulative number of diseases, not of groups
        size = dict(labelled.group_by("group").agg(pl.col("disease").n_unique()).iter_rows())
        cum = np.cumsum([size[g] for g in perm]) / sum(size.values())
        n_test = int(np.searchsorted(cum, test_frac)) + 1
        n_valid = int(np.searchsorted(cum, test_frac + valid_frac)) + 1 - n_test
    else:
        n_test, n_valid = round(test_frac * len(names)), round(valid_frac * len(names))
    test_g, valid_g = perm[:n_test].tolist(), perm[n_test : n_test + n_valid].tolist()

    def part(gs: list[str]) -> pl.DataFrame:
        return labelled.filter(pl.col("group").is_in(gs)).select(PAIR)

    test, valid = part(test_g), part(valid_g)
    train = labelled.filter(~pl.col("group").is_in(test_g + valid_g)).select(PAIR)
    held = pl.concat([test, valid])["disease"].unique().implode()
    graph = _training_graph(edges, pl.col("dst_index").is_in(held), prune)
    return Split(name, seed, graph, train, valid, test)


def random_split(
    edges: pl.DataFrame,
    seed: int,
    test_frac: float = 0.1,
    valid_frac: float = 0.1,
    prune: bool = True,
) -> Split:
    ind = indications(edges)
    order = np.random.default_rng(seed).permutation(ind.height)
    n_test, n_valid = round(test_frac * ind.height), round(valid_frac * ind.height)
    test = ind[order[:n_test]].sort(PAIR)
    valid = ind[order[n_test : n_test + n_valid]].sort(PAIR)
    train = ind[order[n_test + n_valid :]].sort(PAIR)
    held_pairs = pl.concat([test, valid]).select(
        pl.col("drug").alias("src_index"), pl.col("disease").alias("dst_index"), held=pl.lit(True)
    )
    tagged = edges.join(held_pairs, on=["src_index", "dst_index"], how="left")
    graph = _training_graph(tagged, pl.col("held").fill_null(False), prune).drop("held")
    return Split("random", seed, graph, train, valid, test)


def make_split(
    name: str, edges: pl.DataFrame, groups: pl.DataFrame, seed: int, prune: bool = True
) -> Split:
    if name == "disease":
        return disease_split(edges, groups, seed, prune=prune)
    if name == "disease_strict":
        strict = ontology_groups(edges, groups)
        return disease_split(edges, strict, seed, prune=prune, name=name, by_size=True)
    if name == "random":
        return random_split(edges, seed, prune=prune)
    raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")


def split_dir(root: Path, name: str, seed: int) -> Path:
    return root / name / f"seed_{seed}"


def save_split(split: Split, root: Path) -> Path:
    out = split_dir(root, split.name, split.seed)
    out.mkdir(parents=True, exist_ok=True)
    for part in ("graph", "train", "valid", "test"):
        getattr(split, part).write_parquet(out / f"{part}.parquet")
    return out


def load_split(root: Path, name: str, seed: int) -> Split:
    d = split_dir(root, name, seed)
    parts = {p: pl.read_parquet(d / f"{p}.parquet") for p in ("graph", "train", "valid", "test")}
    return Split(name, seed, **parts)
