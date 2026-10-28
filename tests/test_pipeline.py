"""End-to-end GraphRAG pipeline, offline: recorded LLM responses and a scripted graph.

Re-record with RECORD_LLM=1 (needs ANTHROPIC_API_KEY).
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("rapidfuzz")
pytest.importorskip("pydantic")

from graphrag.answer import DISCLAIMER, cited_ids  # noqa: E402
from graphrag.pipeline import ask  # noqa: E402
from llm import AnthropicProvider, RecordedProvider  # noqa: E402

FIXTURES = Path("tests/fixtures/llm")

NODES = [
    {"id": 14287, "name": "Ibuprofen", "label": "Drug"},
    {"id": 33675, "name": "asthma", "label": "Disease"},
    {"id": 4425, "name": "PTGS2", "label": "Gene"},
    {"id": 9031, "name": "PTGS1", "label": "Gene"},
    {"id": 90001, "name": "Parkinson disease", "label": "Disease"},
    {"id": 90002, "name": "Parkinson disease 2", "label": "Disease"},
]


class ScriptedKG:
    """Answers the handful of query shapes the pipeline sends."""

    def __init__(self):
        self.queries = []

    def read(self, query, params=None, timeout=10.0):
        params = params or {}
        self.queries.append(query)
        if "db.index.fulltext.queryNodes" in query:
            words = [w for w in params["q"].lower().replace('"', " ").split() if w.isalnum()]
            labels = params.get("labels") or []
            return [
                {**n, "score": 1.0}
                for n in NODES
                if (not labels or n["label"] in labels)
                and any(w.rstrip("s") in n["name"].lower() for w in words)
            ]
        if query.startswith("EXPLAIN"):
            return []
        if "PREDICTED_INDICATION" in query and "$k" in query:
            return [
                {
                    "drug_id": 14287,
                    "drug": "Ibuprofen",
                    "rank": 1,
                    "score": 0.8,
                    "ct_studies": 3,
                    "ct_max_phase": 2.0,
                }
            ]
        if "AS relation, labels(m)[0] AS label" in query:
            return [
                {"relation": "TARGET", "label": "Gene", "id": 4425, "name": "PTGS2"},
                {"relation": "TARGET", "label": "Gene", "id": 9031, "name": "PTGS1"},
            ]
        return [{"n": 2}]


@pytest.fixture
def provider():
    inner = AnthropicProvider() if os.environ.get("RECORD_LLM") else None
    return RecordedProvider(FIXTURES, inner=inner)


def test_neighbours_question_uses_template_and_cites_ids(provider):
    kg = ScriptedKG()
    ans = ask("Which genes does Ibuprofen target?", provider, kg)
    assert ans.intent == "neighbours"
    assert {r["name"] for r in ans.rows} == {"PTGS1", "PTGS2"}
    assert {4425, 9031} <= cited_ids(ans.text)


def test_repurpose_question_adds_disclaimer(provider):
    ans = ask("Which drugs could be repurposed for asthma?", provider, ScriptedKG())
    assert ans.intent == "repurpose"
    assert ans.rows[0]["drug_id"] == 14287
    assert DISCLAIMER in ans.text


def test_free_query_goes_through_guard_and_explain(provider):
    kg = ScriptedKG()
    ans = ask("How many genes does Ibuprofen target?", provider, kg)
    assert ans.intent == "free_query"
    assert ans.cypher and "LIMIT" in ans.cypher[0].upper()
    assert any(q.startswith("EXPLAIN ") for q in kg.queries)


def test_ambiguous_disease_asks_instead_of_guessing(provider):
    ans = ask("Which drugs could be repurposed for parkinsons?", provider, ScriptedKG())
    assert ans.clarification and "Parkinson disease" in ans.clarification
    assert ans.rows == []
