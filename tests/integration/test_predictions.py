"""PREDICTED_INDICATION writer against a live Neo4j (skipped unless NEO4J_URI is set)."""

import os

import numpy as np
import polars as pl
import pytest

pytest.importorskip("neo4j")
if not os.environ.get("NEO4J_URI"):
    pytest.skip("NEO4J_URI not set", allow_module_level=True)

from kg.client import connect  # noqa: E402
from kg.load import load_frames, node_count  # noqa: E402
from ml.write_predictions import top_predictions, write_predictions  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402


@pytest.fixture(scope="module")
def driver():
    driver = connect()
    nodes, edges = read_fixture()
    n = node_count(driver)
    if n == 0:
        load_frames(driver, nodes, edges)
    elif n != nodes.height:
        driver.close()
        pytest.skip("writer replaces every PREDICTED_INDICATION; only run on the fixture graph")
    yield driver
    with driver.session() as s:
        s.run("MATCH ()-[r:PREDICTED_INDICATION]->() DELETE r").consume()
    driver.close()


def test_writer_is_idempotent(driver):
    nodes, edges = read_fixture()
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())[:5]
    known = edges.filter(pl.col("rel_type") == "INDICATION").select(
        pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease")
    )
    rng = np.random.default_rng(0)
    preds = top_predictions(lambda c: rng.random((len(c), len(drugs))), drugs, diseases, known, k=3)
    assert preds.join(known, on=["drug", "disease"]).is_empty()
    first = write_predictions(driver, preds, "test-v1")
    second = write_predictions(driver, preds, "test-v2")
    assert first == second == preds.height
    with driver.session() as s:
        versions = s.run(
            "MATCH ()-[r:PREDICTED_INDICATION]->() RETURN collect(DISTINCT r.model_version) AS v"
        ).single()["v"]
    assert versions == ["test-v2"]
