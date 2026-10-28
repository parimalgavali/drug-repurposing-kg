"""Entity linking on scripted full-text rows: German Wikidata names count as matches."""

import pytest

pytest.importorskip("rapidfuzz")

from graphrag.linker import SEARCH, link  # noqa: E402

ROWS = [
    {"id": 29448, "name": "scurvy", "name_de": "Skorbut", "label": "Disease", "score": 2.0},
    {"id": 1, "name": "scurvy-like rash", "name_de": None, "label": "Phenotype", "score": 1.0},
]


class FakeKG:
    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def read(self, query, params=None, timeout=10.0):
        self.queries.append((query, params))
        return self.rows


def test_search_queries_english_and_german_indexes():
    assert "'entityNames'" in SEARCH and "'entityNamesDe'" in SEARCH


def test_german_name_links_confidently():
    lk = link(FakeKG(ROWS), "Skorbut")
    assert (lk.best.name, lk.best.confidence) == ("scurvy", 1.0)
    assert not lk.ambiguous


def test_english_ranking_unchanged_without_german_names():
    rows = [{**r, "name_de": None} for r in ROWS]
    assert link(FakeKG(rows), "scurvy").best.name == "scurvy"
    assert link(FakeKG([{k: v for k, v in r.items() if k != "name_de"} for r in ROWS]), "scurvy")
