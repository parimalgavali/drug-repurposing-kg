"""Normalise raw PrimeKG files into canonical parquet tables.

PrimeKG stores every undirected edge twice, as (x, y) and (y, x). `canonical_edges` keeps one
row per (rel_type, pair). The source node is the one whose type comes first in `TYPE_ORDER`,
and for same-type relations it is the smaller `node_index`.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import polars as pl

TYPE_ORDER = (
    "drug",
    "disease",
    "gene/protein",
    "effect/phenotype",
    "exposure",
    "pathway",
    "biological_process",
    "molecular_function",
    "cellular_component",
    "anatomy",
)
LABELS = {
    "drug": "Drug",
    "disease": "Disease",
    "gene/protein": "Gene",
    "effect/phenotype": "Phenotype",
    "exposure": "Exposure",
    "pathway": "Pathway",
    "biological_process": "BiologicalProcess",
    "molecular_function": "MolecularFunction",
    "cellular_component": "CellularComponent",
    "anatomy": "Anatomy",
}
EXPECTED_NODES = 129_375
EXPECTED_EDGES = 4_050_249

EDGE_COLUMNS = ("rel_type", "src_index", "src_type", "dst_index", "dst_type")
NODE_COLUMNS = ("node_index", "node_id", "node_type", "label", "node_name", "node_source")


def normalise_rel(relation: str, display_relation: str) -> str:
    """`drug_protein` is split by its display relation (target/enzyme/transporter/carrier)."""
    name = display_relation if relation == "drug_protein" else relation
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def _rel_type_expr() -> pl.Expr:
    name = (
        pl.when(pl.col("relation") == "drug_protein")
        .then(pl.col("display_relation"))
        .otherwise(pl.col("relation"))
    )
    return (
        name.str.replace_all(r"[^A-Za-z0-9]+", "_")
        .str.strip_chars("_")
        .str.to_uppercase()
        .alias("rel_type")
    )


def canonical_edges(kg: pl.LazyFrame) -> pl.LazyFrame:
    rank = {t: i for i, t in enumerate(TYPE_ORDER)}
    x_rank = pl.col("x_type").replace_strict(rank, return_dtype=pl.Int8)
    y_rank = pl.col("y_type").replace_strict(rank, return_dtype=pl.Int8)
    x_idx = pl.col("x_index").cast(pl.Int64)
    y_idx = pl.col("y_index").cast(pl.Int64)
    swap = (x_rank > y_rank) | ((x_rank == y_rank) & (x_idx > y_idx))
    return (
        kg.select(
            _rel_type_expr(),
            pl.when(swap).then(y_idx).otherwise(x_idx).alias("src_index"),
            pl.when(swap).then(pl.col("y_type")).otherwise(pl.col("x_type")).alias("src_type"),
            pl.when(swap).then(x_idx).otherwise(y_idx).alias("dst_index"),
            pl.when(swap).then(pl.col("x_type")).otherwise(pl.col("y_type")).alias("dst_type"),
        )
        .unique(subset=["rel_type", "src_index", "dst_index"])
        .sort("rel_type", "src_index", "dst_index")
    )


def read_tab(path: Path) -> pl.DataFrame:
    """Read a PrimeKG `.tab` file as strings.

    The csv module is used because the feature files have quoted fields with embedded
    newlines that the polars reader rejects.
    """
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        header = next(reader)
        columns = list(zip(*reader, strict=True)) or [()] * len(header)
    return pl.DataFrame(
        {name: list(values) for name, values in zip(header, columns, strict=True)},
        schema={name: pl.String for name in header},
    )


def prepare_nodes(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw.with_columns(
            pl.col("node_index").cast(pl.Int64),
            pl.col("node_type").replace_strict(LABELS).alias("label"),
        )
        .select(NODE_COLUMNS)
        .sort("node_index")
    )


def prepare_features(raw: pl.DataFrame) -> pl.DataFrame:
    """Feature files contain a few repeated `node_index` rows; keep the first."""
    return (
        raw.with_columns(pl.col("node_index").cast(pl.Int64))
        .unique(subset=["node_index"], keep="first", maintain_order=True)
        .sort("node_index")
    )


def prepare_all(raw_dir: Path, out_dir: Path, check_counts: bool = True) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = prepare_nodes(read_tab(raw_dir / "nodes.tab"))
    kg = pl.scan_csv(raw_dir / "kg.csv", infer_schema_length=0)
    edges = canonical_edges(kg).collect()

    if check_counts:
        if nodes.height != EXPECTED_NODES:
            raise ValueError(f"nodes: {nodes.height:,} != expected {EXPECTED_NODES:,}")
        if edges.height != EXPECTED_EDGES:
            raise ValueError(f"edges: {edges.height:,} != expected {EXPECTED_EDGES:,}")

    nodes.write_parquet(out_dir / "nodes.parquet")
    edges.write_parquet(out_dir / "edges.parquet")
    counts = {"nodes": nodes.height, "edges": edges.height}
    for name in ("drug_features", "disease_features"):
        src = raw_dir / f"{name}.tab"
        if src.exists():
            feats = prepare_features(read_tab(src))
            feats.write_parquet(out_dir / f"{name}.parquet")
            counts[name] = feats.height
    return counts
