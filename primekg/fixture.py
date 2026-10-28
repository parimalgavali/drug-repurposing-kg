"""Deterministic small subgraph around a few well-known diseases, for tests and CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "primekg_mini"
NODE_SCHEMA = {
    "node_index": pl.Int64,
    "node_id": pl.String,
    "node_type": pl.String,
    "label": pl.String,
    "node_name": pl.String,
    "node_source": pl.String,
}
EDGE_SCHEMA = {
    "rel_type": pl.String,
    "src_index": pl.Int64,
    "src_type": pl.String,
    "dst_index": pl.Int64,
    "dst_type": pl.String,
}

SEED_DISEASES = (
    "asthma",
    "psoriasis",
    "Parkinson disease",
    "type 2 diabetes mellitus",
    "rheumatoid arthritis",
)
LABEL_RELS = ("INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE")
EXCLUDED_RELS = ("DRUG_DRUG", "ANATOMY_PROTEIN_PRESENT", "ANATOMY_PROTEIN_ABSENT")


def read_fixture(path: Path = FIXTURE_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    nodes = pl.read_csv(path / "nodes.csv", schema=NODE_SCHEMA)
    edges = pl.read_csv(path / "edges.csv", schema=EDGE_SCHEMA)
    return nodes, edges


def _sample_per_group(df: pl.DataFrame, group: str, cap: int, rng: np.random.Generator):
    df = df.sort("rel_type", "src_index", "dst_index")
    return (
        df.with_columns(pl.Series("_r", rng.random(df.height)))
        .with_columns(pl.col("_r").rank("ordinal").over(group).alias("_rank"))
        .filter(pl.col("_rank") <= cap)
        .drop("_r", "_rank")
    )


def seed_disease_ids(nodes: pl.DataFrame, names=SEED_DISEASES) -> list[int]:
    found = nodes.filter(
        (pl.col("node_type") == "disease")
        & pl.col("node_name").str.to_lowercase().is_in([n.lower() for n in names])
    )
    missing = set(n.lower() for n in names) - set(found["node_name"].str.to_lowercase())
    if missing:
        raise ValueError(f"seed diseases not found: {sorted(missing)}")
    return sorted(found["node_index"].to_list())


def build_fixture(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    seeds=SEED_DISEASES,
    seed: int = 7,
    max_edges: int = 2500,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    diseases = seed_disease_ids(nodes, seeds)

    def rel(name: str) -> pl.DataFrame:
        return edges.filter(pl.col("rel_type") == name)

    ind = rel("INDICATION").filter(pl.col("dst_index").is_in(diseases))
    drugs = _sample_per_group(ind, "dst_index", 12, rng)["src_index"].unique()
    contra = rel("CONTRAINDICATION").filter(pl.col("dst_index").is_in(diseases))
    drugs = pl.concat([drugs, _sample_per_group(contra, "dst_index", 3, rng)["src_index"]])

    genes = _sample_per_group(
        rel("DISEASE_PROTEIN").filter(pl.col("src_index").is_in(diseases)), "src_index", 25, rng
    )["dst_index"]
    targets = _sample_per_group(
        rel("TARGET").filter(pl.col("src_index").is_in(drugs.implode())), "src_index", 5, rng
    )["dst_index"]
    phenos = _sample_per_group(
        rel("DISEASE_PHENOTYPE_POSITIVE").filter(pl.col("src_index").is_in(diseases)),
        "src_index",
        8,
        rng,
    )["dst_index"]
    other_ind = rel("INDICATION").filter(
        pl.col("src_index").is_in(drugs.implode()) & ~pl.col("dst_index").is_in(diseases)
    )
    extra_diseases = _sample_per_group(other_ind, "src_index", 2, rng)["dst_index"]
    all_genes = pl.concat([genes, targets]).unique()
    pathways = _sample_per_group(
        rel("PATHWAY_PROTEIN").filter(pl.col("src_index").is_in(all_genes.implode())),
        "src_index",
        1,
        rng,
    )["dst_index"]

    keep = (
        pl.concat(
            [
                pl.Series(diseases, dtype=pl.Int64),
                drugs,
                all_genes,
                phenos,
                extra_diseases,
                pathways,
            ]
        )
        .unique()
        .sort()
    )

    sub = edges.filter(
        pl.col("src_index").is_in(keep.implode())
        & pl.col("dst_index").is_in(keep.implode())
        & ~pl.col("rel_type").is_in(EXCLUDED_RELS)
    ).sort("rel_type", "src_index", "dst_index")
    labels = sub.filter(pl.col("rel_type").is_in(LABEL_RELS))
    rest = sub.filter(~pl.col("rel_type").is_in(LABEL_RELS))
    budget = max(max_edges - labels.height, 0)
    if rest.height > budget:
        idx = np.sort(rng.choice(rest.height, size=budget, replace=False))
        rest = rest[idx]
    sub = pl.concat([labels, rest]).sort("rel_type", "src_index", "dst_index")

    used = pl.concat([sub["src_index"], sub["dst_index"]]).unique()
    sub_nodes = nodes.filter(pl.col("node_index").is_in(used.implode())).sort("node_index")
    return sub_nodes, sub
