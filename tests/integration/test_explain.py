"""Evidence paths against a live Neo4j (skipped unless NEO4J_URI is set)."""

import os

import pytest

pytest.importorskip("neo4j")
if not os.environ.get("NEO4J_URI"):
    pytest.skip("NEO4J_URI not set", allow_module_level=True)

from explain.paths import explain  # noqa: E402
from kg.client import KGClient, connect  # noqa: E402
from kg.load import load_frames, node_count  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402


@pytest.fixture(scope="module")
def kg():
    driver = connect()
    if node_count(driver) == 0:
        load_frames(driver, *read_fixture())
    client = KGClient(driver)
    yield client
    client.close()


def test_known_indications_get_ranked_metapaths(kg):
    pairs = kg.read(
        "MATCH (d:Drug)-[:INDICATION]->(s:Disease) RETURN d.id AS drug, s.id AS disease LIMIT 20"
    )
    results = [explain(kg, p["drug"], p["disease"]) for p in pairs]
    explained = [r for r in results if r["metapaths"]]
    assert explained, "no indication pair had a 2–3 hop path"
    for r in explained:
        dwpc = [m["dwpc"] for m in r["metapaths"]]
        assert dwpc == sorted(dwpc, reverse=True) and len(dwpc) <= 3
        for m in r["metapaths"]:
            assert m["metapath"].startswith("Drug-") and m["metapath"].endswith("-Disease")
            assert 1 <= len(m["paths"]) <= 5
            assert all(len(p["rels"]) in (2, 3) for p in m["paths"])
        ids = {n["data"]["id"] for n in r["cytoscape"]["nodes"]}
        assert all(
            {e["data"]["source"], e["data"]["target"]} <= ids for e in r["cytoscape"]["edges"]
        )
