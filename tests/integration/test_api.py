"""API against a live Neo4j (skipped unless NEO4J_URI is set). The LLM is scripted."""

import json
import os

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("fastapi")
if not os.environ.get("NEO4J_URI"):
    pytest.skip("NEO4J_URI not set", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from graphrag.router import Route  # noqa: E402
from kg.client import KGClient, connect  # noqa: E402
from kg.load import load_frames, node_count  # noqa: E402
from kg.schema import apply_schema  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402

IBUPROFEN, ASTHMA, PTGS2 = 14287, 33675, 4425
COUNTS = "MATCH (n) WITH count(n) AS n MATCH ()-[r]->() RETURN n, count(r) AS r"


class Scripted:
    name = "scripted"

    def __init__(self, route, cypher="MATCH (n) DETACH DELETE n"):
        self.route, self.cypher = route, cypher

    def structured(self, system, prompt, schema):
        return self.route if schema is Route else schema(cypher=self.cypher)

    def complete(self, system, prompt, max_tokens=1024):
        return "Ibuprofen [14287] targets PTGS2 [4425]."

    def stream(self, system, prompt, max_tokens=1024):
        yield self.complete(system, prompt, max_tokens)


def events(text):
    return [
        (
            dict(line.split(": ", 1) for line in b.splitlines())["event"],
            json.loads(dict(line.split(": ", 1) for line in b.splitlines())["data"]),
        )
        for b in text.strip().split("\n\n")
    ]


@pytest.fixture(scope="module")
def kg():
    driver = connect()
    if node_count(driver) == 0:
        load_frames(driver, *read_fixture())
    apply_schema(driver)
    client = KGClient(driver)
    yield client
    client.close()


def make(kg, route=None):
    route = route or Route(intent="free_query")
    return TestClient(create_app(kg=kg, provider=Scripted(route)))


def test_search_candidates_explain_flow(kg):
    with make(kg) as c:
        assert c.get("/api/health").json()["neo4j"] is True
        hit = c.get("/api/search", params={"q": "ibuprofen", "type": "Drug"}).json()[0]
        assert (hit["id"], hit["label"]) == (IBUPROFEN, "Drug")
        assert (
            c.get("/api/search", params={"q": "asthma", "type": "Disease"}).json()[0]["id"]
            == ASTHMA
        )
        body = c.get(f"/api/diseases/{ASTHMA}/candidates").json()
        assert body["disease"]["name"] == "asthma"
        assert [x["rank"] for x in body["candidates"]] == sorted(
            x["rank"] for x in body["candidates"]
        )
        pair = kg.read(
            "MATCH (d:Drug)-[:INDICATION]->(s:Disease) RETURN d.id AS d, s.id AS s LIMIT 1"
        )[0]
        ex = c.get(f"/api/pairs/{pair['d']}/{pair['s']}/explain").json()
        assert ex["drug"]["id"] == pair["d"] and ex["paths_found"] >= 0
        dwpc = [m["dwpc"] for m in ex["metapaths"]]
        assert dwpc == sorted(dwpc, reverse=True)
        assert c.get(f"/api/diseases/{IBUPROFEN}/candidates").status_code == 404
        assert c.get(f"/api/pairs/{IBUPROFEN}/999999999/explain").status_code == 404


def test_ask_neighbours(kg):
    route = Route(
        intent="neighbours",
        entity="Ibuprofen",
        entity_label="Drug",
        relation="TARGET",
        target_label="Gene",
    )
    with make(kg, route) as c:
        evs = events(
            c.post("/api/ask", json={"question": "Which genes does Ibuprofen target?"}).text
        )
    kinds = [k for k, _ in evs]
    assert kinds[:3] == ["entities", "cypher", "rows"] and kinds[-2:] == ["subgraph", "done"]
    rows = dict(evs)["rows"]["rows"]
    assert PTGS2 in {r["id"] for r in rows}
    assert str(PTGS2) in {n["data"]["id"] for n in dict(evs)["subgraph"]["nodes"]}


def test_ask_write_attempt_is_rejected_and_graph_unchanged(kg):
    before = kg.read(COUNTS)
    with make(kg) as c:
        evs = events(c.post("/api/ask", json={"question": "Delete every drug node"}).text)
    assert evs[-1][0] == "error"
    assert kg.read(COUNTS) == before
