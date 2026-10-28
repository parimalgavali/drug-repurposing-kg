"""Runs against a live Neo4j (skipped unless NEO4J_URI is set). An empty database gets the
mini fixture loaded; a non-empty one (e.g. the full PrimeKG import) is only read."""

import os
import time

import pytest

pytest.importorskip("neo4j")
if not os.environ.get("NEO4J_URI"):
    pytest.skip("NEO4J_URI not set", allow_module_level=True)

from neo4j.exceptions import ClientError  # noqa: E402

from kg.client import KGClient, connect  # noqa: E402
from kg.load import load_frames, node_count  # noqa: E402
from kg.queries import load_queries  # noqa: E402
from kg.schema import apply_schema  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402


@pytest.fixture(scope="module")
def kg():
    driver = connect()
    if node_count(driver) == 0:
        load_frames(driver, *read_fixture())
    apply_schema(driver)
    client = KGClient(driver)
    yield client
    client.close()


@pytest.mark.parametrize("query", load_queries(), ids=lambda q: q.name)
def test_example_query_returns_rows_quickly(kg, query):
    start = time.perf_counter()
    rows = kg.read(query.cypher, query.params, timeout=10)
    assert rows, query.name
    assert time.perf_counter() - start < 2.0


def test_relationship_types_have_one_direction(kg):
    rows = kg.read(
        "MATCH (a)-[:INDICATION]->(b) RETURN labels(a)[0] AS s, labels(b)[0] AS d, count(*) AS n"
    )
    assert [(r["s"], r["d"]) for r in rows] == [("Drug", "Disease")]


def test_read_rejects_writes(kg):
    with pytest.raises(ClientError, match="read access mode"):
        kg.read("CREATE (:Drug {id: -1})")
