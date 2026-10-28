"""Demo graph without the PrimeKG download: the mini fixture plus popularity predictions.

Loads data/fixtures/primekg_mini into an *empty* Neo4j, applies the schema and writes
PREDICTED_INDICATION from a popularity baseline (model_version "fixture-popularity"), so the API
and the UI have candidates to show. Used by the e2e test and `docker compose` demos.

    PYTHONPATH=. uv run --extra data --extra graph python scripts/demo_fixture.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from kg.client import connect
from kg.load import load_frames, node_count
from kg.schema import apply_schema
from ml.write_predictions import top_predictions, write_predictions
from primekg.fixture import read_fixture

VERSION = "fixture-popularity"


def popularity_predictions(nodes: pl.DataFrame, edges: pl.DataFrame, k: int = 20) -> pl.DataFrame:
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    known = edges.filter(pl.col("rel_type") == "INDICATION").select(
        pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease")
    )
    counts = dict(known.group_by("drug").len().iter_rows())
    # Indication count, ties broken towards lower node index so the output is deterministic.
    row = np.array([counts.get(int(d), 0) for d in drugs], dtype=float)
    row -= np.arange(len(drugs)) * 1e-6
    return top_predictions(lambda c: np.tile(row, (len(c), 1)), drugs, diseases, known, k=k)


def main() -> None:
    nodes, edges = read_fixture()
    driver = connect()
    try:
        n = node_count(driver)
        if n == 0:
            load_frames(driver, nodes, edges)
        elif n != nodes.height:
            raise SystemExit(f"database has {n} nodes, not the fixture; refusing to overwrite it")
        apply_schema(driver)
        written = write_predictions(driver, popularity_predictions(nodes, edges), VERSION)
        print(f"fixture graph ready: {nodes.height} nodes, {written} PREDICTED_INDICATION")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
