"""Question → route → link → template or text2cypher → grounded answer.

uv run --extra graph --extra llm python -m graphrag.pipeline "Which genes does ibuprofen target?"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

from graphrag import templates
from graphrag.answer import MAX_ROWS, answer
from graphrag.linker import Link, link
from graphrag.router import Route, route
from graphrag.text2cypher import text2cypher
from kg.client import KGClient
from llm import Provider, get_provider

SURE = 0.97


@dataclass
class Answer:
    question: str
    intent: str
    text: str
    cypher: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    evidence: dict | None = None
    clarification: str | None = None
    attempts: list[dict] = field(default_factory=list)


def _resolve(lk: Link):
    """The node to use, or None when the user should pick."""
    if lk.best and (not lk.ambiguous or lk.best.confidence >= SURE):
        return lk.best
    return None


def _path(path: dict) -> str:
    nodes, rels = path["nodes"], path["rels"]
    out = f"{nodes[0]['name']} [{nodes[0]['id']}]"
    for rel, n in zip(rels, nodes[1:], strict=True):
        out += f" -{rel}- {n['name']} [{n['id']}]"
    return out


def _links(kg: KGClient, r: Route) -> dict[str, Link]:
    out = {}
    if r.drug:
        out["drug"] = link(kg, r.drug, ("Drug",))
    if r.disease:
        out["disease"] = link(kg, r.disease, ("Disease",))
    if r.entity:
        out["entity"] = link(kg, r.entity, (r.entity_label,) if r.entity_label else ())
    return out


@dataclass
class Plan:
    """Everything up to (not including) answer synthesis, so callers can stream the answer."""

    question: str
    intent: str
    links: list[dict]
    cypher: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    evidence: dict | None = None
    clarification: str | None = None
    attempts: list[dict] = field(default_factory=list)
    error: str | None = None


def plan(question: str, provider: Provider, kg: KGClient) -> Plan:
    r = route(provider, question)
    links = _links(kg, r)
    chosen = {slot: _resolve(lk) for slot, lk in links.items()}
    link_info = [
        {"slot": s, "mention": lk.mention, **(asdict(lk.best) if lk.best else {})}
        for s, lk in links.items()
    ]
    needed = {
        "repurpose": ["disease"],
        "explain_pair": ["drug", "disease"],
        "neighbours": ["entity"],
    }
    intent = r.intent
    if intent in needed:
        missing = [s for s in needed[intent] if s not in links]
        if missing:
            intent = "free_query"
        else:
            unclear = [links[s] for s in needed[intent] if chosen[s] is None]
            if unclear:
                text = " ".join(lk.clarification() for lk in unclear)
                return Plan(question, intent, link_info, clarification=text)

    evidence = None
    attempts: list[dict] = []
    error = None
    if intent == "repurpose":
        res = templates.repurpose(kg, chosen["disease"].id)
        cypher, rows = res.cypher, res.rows
        results = {"disease": asdict(chosen["disease"]), "predictions": rows, **res.extra}
    elif intent == "explain_pair":
        res = templates.explain_pair(kg, chosen["drug"].id, chosen["disease"].id)
        cypher, rows, evidence = res.cypher, res.rows, res.extra["evidence"]
        results = {
            "pair": rows,
            "metapaths": [
                {
                    "metapath": m["metapath"],
                    "dwpc": m["dwpc"],
                    "paths": [_path(p) for p in m["paths"]],
                }
                for m in evidence["metapaths"]
            ],
        }
    elif intent == "neighbours":
        e = chosen["entity"]
        try:
            res = templates.neighbours(kg, e.id, e.label, r.relation, r.target_label)
        except ValueError:
            res = templates.neighbours(kg, e.id, e.label)
        cypher, rows = res.cypher, res.rows
        results = {"entity": asdict(e), "neighbours": rows}
    else:
        hints = "\n".join(
            f"- {lk.mention!r} → {c.name} ({c.label}, id {c.id})"
            for lk in links.values()
            for c in lk.candidates[:1]
        )
        t2c = text2cypher(provider, kg, question, hints)
        attempts = t2c.attempts
        error = None if t2c.ok else attempts[-1]["error"]
        cypher, rows = ([t2c.cypher] if t2c.cypher else []), t2c.rows
        results = {"cypher": t2c.cypher, "rows": rows[:MAX_ROWS], "error": error}
    return Plan(question, intent, link_info, cypher, rows, results, evidence, None, attempts, error)


def ask(question: str, provider: Provider, kg: KGClient) -> Answer:
    p = plan(question, provider, kg)
    if p.clarification:
        return Answer(
            question, p.intent, p.clarification, links=p.links, clarification=p.clarification
        )
    text = answer(provider, question, p.intent, p.results)
    return Answer(question, p.intent, text, p.cypher, p.rows, p.links, p.evidence, None, p.attempts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--provider", default=None, help="anthropic | ollama | recorded")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with KGClient() as kg:
        result = ask(args.question, get_provider(args.provider), kg)
    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
        return
    print(f"[{result.intent}] {result.text}\n")
    for q in result.cypher:
        print(q, "\n")


if __name__ == "__main__":
    main()
