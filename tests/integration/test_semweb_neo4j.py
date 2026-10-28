"""RDF export of a live Neo4j subgraph (skipped unless NEO4J_URI is set)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("pyshacl")
if not os.environ.get("NEO4J_URI"):
    pytest.skip("NEO4J_URI not set", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402
from pyshacl import validate  # noqa: E402
from rdflib import Graph  # noqa: E402

from api.main import create_app  # noqa: E402
from kg.client import KGClient, connect  # noqa: E402
from kg.load import load_frames, node_count  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402
from semweb.export import export  # noqa: E402
from semweb.store import RdfStore  # noqa: E402

TREATED = "MATCH (:Drug)-[:INDICATION]-(s:Disease) RETURN s.id AS id ORDER BY s.id LIMIT 1"


@pytest.fixture(scope="module")
def kg():
    driver = connect()
    if node_count(driver) == 0:
        load_frames(driver, *read_fixture())
    driver.close()
    client = KGClient(connect())
    yield client
    client.close()


def test_export_conforms_and_is_queryable(kg):
    disease = kg.read(TREATED, {})[0]["id"]
    g = export(kg, disease, k=5)
    conforms, _, report = validate(g, shacl_graph=Graph().parse(Path("semweb/shapes.ttl")))
    assert conforms, report
    store = RdfStore(g.serialize(format="turtle"))
    out = store.query(
        "PREFIX rkg: <https://w3id.org/repurposekg/vocab#>\n"
        "SELECT ?d WHERE { ?d rkg:indication ?s . ?s rkg:servedId ?id } LIMIT 1"
    )
    assert out["rows"]


def test_disease_rdf_endpoint(kg):
    disease = kg.read(TREATED, {})[0]["id"]
    with TestClient(create_app(kg=kg, provider=None)) as client:
        r = client.get(f"/api/diseases/{disease}/rdf", params={"k": 5})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/turtle")
        assert "rkg:Disease" in r.text
        assert client.get("/api/diseases/-1/rdf").status_code == 404
