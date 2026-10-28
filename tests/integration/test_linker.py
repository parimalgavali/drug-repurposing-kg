"""Entity linking against a live Neo4j (skipped unless NEO4J_URI is set)."""

import os

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("rapidfuzz")
if not os.environ.get("NEO4J_URI"):
    pytest.skip("NEO4J_URI not set", allow_module_level=True)

from graphrag.linker import link  # noqa: E402
from graphrag.templates import neighbours  # noqa: E402
from kg.client import KGClient, connect  # noqa: E402
from kg.load import load_frames, node_count  # noqa: E402
from kg.schema import apply_schema  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402


@pytest.fixture(scope="module")
def kg():
    driver = connect()
    if node_count(driver) == 0:
        load_frames(driver, *read_fixture())
    apply_schema(driver)  # the linker needs the entityNames full-text index
    client = KGClient(driver)
    yield client
    client.close()


@pytest.mark.parametrize(
    "mention,labels,name,label",
    [
        ("Ibuprofen", ("Drug",), "Ibuprofen", "Drug"),
        ("ibuprofen", ("Drug",), "Ibuprofen", "Drug"),
        ("ibuprofn", ("Drug",), "Ibuprofen", "Drug"),
        ("PTGS2", (), "PTGS2", "Gene"),
        ("asthma", ("Disease",), "asthma", "Disease"),
    ],
)
def test_links_to_expected_node(kg, mention, labels, name, label):
    best = link(kg, mention, labels).best
    assert best is not None and (best.name, best.label) == (name, label)


def test_nonsense_asks_for_clarification(kg):
    lk = link(kg, "zzqxv flurbo", ("Drug",))
    assert lk.best is None or lk.ambiguous
    assert lk.clarification()


def test_neighbours_template_finds_ibuprofen_targets(kg):
    ibu = link(kg, "Ibuprofen", ("Drug",)).best
    res = neighbours(kg, ibu.id, "Drug", "TARGET", "Gene")
    assert {"PTGS1", "PTGS2"} <= {r["name"] for r in res.rows}
