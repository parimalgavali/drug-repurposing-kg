from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pyshacl import validate
from rdflib import Graph
from rdflib.namespace import RDFS

from api.main import create_app
from semweb.export import to_graph
from semweb.store import RdfStore, SparqlError

SHAPES = Graph().parse(Path("semweb/shapes.ttl"))
PREFIXES = """PREFIX rkg: <https://w3id.org/repurposekg/vocab#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
NODES = [
    {"id": 33675, "label": "Disease", "name": "asthma", "source": "MONDO", "source_id": "4979"},
    {
        "id": 14263,
        "label": "Drug",
        "name": "Salmeterol",
        "source": "DrugBank",
        "source_id": "DB00938",
    },
    {
        "id": 15479,
        "label": "Drug",
        "name": "Salbutamol",
        "source": "DrugBank",
        "source_id": "DB01001",
    },
    {"id": 2135, "label": "Gene", "name": "ADRB2", "source": "NCBI", "source_id": "154"},
    {"id": 90001, "label": "Phenotype", "name": "Wheezing", "source": "HPO", "source_id": "30828"},
    {
        "id": 27160,
        "label": "Disease",
        "name": "grouped",
        "source": "MONDO_grouped",
        "source_id": "8099_12497",
    },
]
EDGES = [
    {"src": 14263, "rel": "TARGET", "dst": 2135},
    {"src": 15479, "rel": "TARGET", "dst": 2135},
    {"src": 15479, "rel": "INDICATION", "dst": 33675},
    {"src": 33675, "rel": "DISEASE_PROTEIN", "dst": 2135},
    {"src": 33675, "rel": "DISEASE_PHENOTYPE_POSITIVE", "dst": 90001},
]
PREDICTIONS = [
    {"drug": 14263, "score": 4.99, "rank": 1, "model_version": "gnn", "ct_studies": 3},
]


@pytest.fixture
def graph() -> Graph:
    return to_graph(33675, NODES, EDGES, PREDICTIONS)


def test_iris_use_source_ontologies(graph):
    subjects = {str(s) for s in graph.subjects()}
    assert "http://purl.obolibrary.org/obo/MONDO_0004979" in subjects
    assert "https://go.drugbank.com/drugs/DB00938" in subjects
    assert "https://identifiers.org/ncbigene:154" in subjects
    assert "http://purl.obolibrary.org/obo/HP_0030828" in subjects
    assert "https://w3id.org/repurposekg/node/27160" in subjects


def test_export_conforms_to_shapes(graph):
    conforms, _, report = validate(graph, shacl_graph=SHAPES)
    assert conforms, report


def test_shapes_catch_missing_label_and_bad_rank(graph):
    graph.remove((None, RDFS.label, None))
    conforms, _, report = validate(graph, shacl_graph=SHAPES)
    assert not conforms and "label" in report


@pytest.fixture
def store(graph) -> RdfStore:
    return RdfStore(graph.serialize(format="turtle"))


def test_select_and_ask(store):
    q = (
        PREFIXES
        + """SELECT ?drug ?gene WHERE {
      ?p a rkg:Prediction ; rkg:drug ?d ; rkg:disease ?s .
      ?d rdfs:label ?drug ; rkg:target ?g . ?s rkg:diseaseProtein ?g . ?g rdfs:label ?gene }"""
    )
    out = store.query(q)
    assert out["vars"] == ["drug", "gene"]
    assert [(r["drug"]["value"], r["gene"]["value"]) for r in out["rows"]] == [
        ("Salmeterol", "ADRB2")
    ]
    assert store.query(PREFIXES + "ASK { ?d rkg:indication ?s }") == {"boolean": True}


@pytest.mark.parametrize(
    "query",
    [
        "DELETE WHERE { ?s ?p ?o }",
        "INSERT DATA { <http://a> <http://b> <http://c> }",
        "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
        "SELECT * WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }",
        "# comment\nDROP ALL",
        "SELECT WHERE {",
    ],
)
def test_rejects_non_read_queries(store, query):
    n = len(store)
    with pytest.raises(SparqlError):
        store.query(query)
    assert len(store) == n


def test_sparql_endpoint(store):
    with TestClient(create_app(kg=object(), provider=None, rdf=store)) as client:
        q = PREFIXES + "SELECT ?n WHERE { ?d a rkg:Drug ; rdfs:label ?n } ORDER BY ?n"
        r = client.get("/api/sparql", params={"query": q})
        assert r.status_code == 200
        assert [row["n"]["value"] for row in r.json()["rows"]] == ["Salbutamol", "Salmeterol"]
        assert (
            client.post("/api/sparql", json={"query": "DELETE WHERE {?s ?p ?o}"}).status_code == 400
        )
