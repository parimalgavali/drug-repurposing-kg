"""Entity linking: a free-text mention → typed graph nodes via the `entityNames` full-text index,
re-ranked with rapidfuzz. Ambiguous mentions are returned with candidates so the caller can ask."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, utils

from kg.client import KGClient

SEARCH = """
CALL db.index.fulltext.queryNodes('entityNames', $q) YIELD node, score
WHERE size($labels) = 0 OR labels(node)[0] IN $labels
RETURN node.id AS id, node.name AS name, labels(node)[0] AS label, score
LIMIT 50
"""
CONFIDENT = 0.9
MARGIN = 0.05
_LUCENE = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


@dataclass(frozen=True)
class Candidate:
    id: int
    name: str
    label: str
    confidence: float


@dataclass
class Link:
    mention: str
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def ambiguous(self) -> bool:
        """No candidate, or a weak/near-tied top candidate: ask the user instead of guessing."""
        if not self.candidates:
            return True
        top = self.candidates[0]
        if top.confidence >= CONFIDENT and (
            len(self.candidates) == 1 or top.confidence - self.candidates[1].confidence >= MARGIN
        ):
            return False
        return top.confidence < CONFIDENT or len(self.candidates) > 1

    def clarification(self) -> str:
        if not self.candidates:
            return f"I couldn't find '{self.mention}' in the graph. Could you rephrase it?"
        options = "; ".join(f"{c.name} ({c.label})" for c in self.candidates[:5])
        return f"Which '{self.mention}' do you mean: {options}?"


def lucene_query(mention: str) -> str:
    terms = [_LUCENE.sub(r"\\\1", t) for t in mention.split() if t.strip()]
    # Exact phrase first, then each term (fuzzy for longer terms to absorb typos).
    fuzzy = [f"{t}~" if len(t) > 4 else t for t in terms]
    return f'"{" ".join(terms)}"^3 ' + " ".join(fuzzy)


def similarity(mention: str, name: str) -> float:
    a, b = utils.default_process(mention), utils.default_process(name)
    if a == b:
        return 1.0
    # "parkinsons" → "parkinson disease": singularised mention fully contained in the name.
    singular = " ".join(w[:-1] if len(w) > 4 and w.endswith("s") else w for w in a.split())
    return (
        max(
            fuzz.ratio(a, b),
            0.95 * fuzz.token_sort_ratio(a, b),
            0.9 * fuzz.token_set_ratio(singular, b),
        )
        / 100
    )


def link(kg: KGClient, mention: str, labels: tuple[str, ...] = (), top: int = 5) -> Link:
    mention = mention.strip()
    if not mention:
        return Link(mention)
    rows = kg.read(SEARCH, {"q": lucene_query(mention), "labels": list(labels)})
    scored = {
        (r["label"], r["id"]): Candidate(
            r["id"], r["name"], r["label"], similarity(mention, r["name"])
        )
        for r in rows
    }
    ranked = sorted(scored.values(), key=lambda c: (-c.confidence, len(c.name), c.name))
    return Link(mention, ranked[:top])
