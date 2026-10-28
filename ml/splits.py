"""Train/valid/test splits of INDICATION edges plus the matching training graph.

- disease: hold out disease groups (PrimeKG `group_id_bert`, else the disease itself); every
  INDICATION / CONTRAINDICATION / OFF_LABEL_USE edge of a held-out disease leaves the graph.
- random: 80/10/10 over INDICATION pairs; the clinical edges of held-out pairs leave the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

CLINICAL = ("INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE")
PRUNED = ("DRUG_DRUG", "ANATOMY_PROTEIN_PRESENT", "ANATOMY_PROTEIN_ABSENT")
SPLITS = ("disease", "random")
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
) -> Split:
    ind = indications(edges)
    labelled = ind.join(groups, left_on="disease", right_on="node_index", how="left")
    labelled = labelled.with_columns(pl.col("group").fill_null(pl.col("disease").cast(pl.String)))
    names = np.array(sorted(labelled["group"].unique()))
    perm = np.random.default_rng(seed).permutation(names)
    n_test, n_valid = round(test_frac * len(names)), round(valid_frac * len(names))
    test_g, valid_g = perm[:n_test].tolist(), perm[n_test : n_test + n_valid].tolist()

    def part(gs: list[str]) -> pl.DataFrame:
        return labelled.filter(pl.col("group").is_in(gs)).select(PAIR)

    test, valid = part(test_g), part(valid_g)
    train = labelled.filter(~pl.col("group").is_in(test_g + valid_g)).select(PAIR)
    held = pl.concat([test, valid])["disease"].unique().implode()
    graph = _training_graph(edges, pl.col("dst_index").is_in(held), prune)
    return Split("disease", seed, graph, train, valid, test)


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
