"""Parametrised Cypher for the routed intents (free_query goes through text2cypher)."""

from __future__ import annotations

from dataclasses import dataclass, field

from explain.paths import explain
from graphrag.schema import LABELS, REL_TYPES
from kg.client import KGClient

REPURPOSE = """
MATCH (d:Drug)-[p:PREDICTED_INDICATION]->(s:Disease {id: $disease})
RETURN d.id AS drug_id, d.name AS drug, p.rank AS rank, round(p.score, 3) AS score,
       p.ct_studies AS ct_studies, p.ct_max_phase AS ct_max_phase
ORDER BY p.rank
LIMIT $k
"""
KNOWN = """
MATCH (d:Drug)-[r:INDICATION|OFF_LABEL_USE]->(s:Disease {id: $disease})
RETURN d.id AS drug_id, d.name AS drug, type(r) AS relation
ORDER BY d.name
LIMIT 50
"""
PAIR = """
MATCH (d:Drug {id: $drug}), (s:Disease {id: $disease})
OPTIONAL MATCH (d)-[r:INDICATION|OFF_LABEL_USE|CONTRAINDICATION]->(s)
OPTIONAL MATCH (d)-[p:PREDICTED_INDICATION]->(s)
RETURN d.id AS drug_id, d.name AS drug, s.id AS disease_id, s.name AS disease,
       collect(DISTINCT type(r)) AS known_relations,
       p.rank AS predicted_rank, p.score AS predicted_score, p.ct_studies AS ct_studies
"""


def neighbours_query(label: str, relation: str | None, target_label: str | None) -> str:
    if label not in LABELS or (target_label and target_label not in LABELS):
        raise ValueError("unknown label")
    if relation and relation not in REL_TYPES:
        raise ValueError("unknown relationship type")
    rel = f":{relation}" if relation else ""
    target = f":{target_label}" if target_label else ""
    return (
        f"MATCH (n:{label} {{id: $id}})-[r{rel}]-(m{target})\n"
        "WHERE type(r) <> 'PREDICTED_INDICATION'\n"
        "RETURN type(r) AS relation, labels(m)[0] AS label, m.id AS id, m.name AS name\n"
        "ORDER BY relation, name\nLIMIT 50"
    )


@dataclass
class TemplateResult:
    cypher: list[str]
    params: dict
    rows: list[dict]
    extra: dict = field(default_factory=dict)


def repurpose(kg: KGClient, disease: int, k: int = 10) -> TemplateResult:
    params = {"disease": disease, "k": k}
    rows = kg.read(REPURPOSE, params)
    known = kg.read(KNOWN, params)
    return TemplateResult([REPURPOSE.strip(), KNOWN.strip()], params, rows, {"known": known})


def explain_pair(kg: KGClient, drug: int, disease: int) -> TemplateResult:
    params = {"drug": drug, "disease": disease}
    rows = kg.read(PAIR, params)
    evidence = explain(kg, drug, disease)
    from explain.paths import PATHS

    return TemplateResult([PAIR.strip(), PATHS.strip()], params, rows, {"evidence": evidence})


def neighbours(
    kg: KGClient,
    node: int,
    label: str,
    relation: str | None = None,
    target_label: str | None = None,
) -> TemplateResult:
    query = neighbours_query(label, relation, target_label)
    params = {"id": node}
    return TemplateResult([query], params, kg.read(query, params))
