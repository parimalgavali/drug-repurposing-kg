"""Text → Cypher with guards: schema-only labels/types, read-only clauses, LIMIT, EXPLAIN first,
read-only transaction with a 5 s timeout, and one retry that shows the model its error."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from graphrag.schema import LABELS, REL_TYPES, schema_prompt
from kg.client import KGClient
from llm import Provider

MAX_LIMIT = 50
TIMEOUT = 5.0
_STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|USING\s+PERIODIC)\b"
    r"|\bCALL\s*\{|\bCALL\s+(?!db\.index\.fulltext\.queryNodes\b)[\w.]+",
    re.I,
)
_NODE_LABELS = re.compile(r"\(\s*\w*\s*((?::\s*`?\w+`?\s*)+)")
_REL_TYPES = re.compile(r"\[\s*\w*\s*:\s*(`?\w+`?(?:\s*\|\s*:?\s*`?\w+`?)*)")
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*$", re.I)

SYSTEM = f"""You translate questions about a biomedical knowledge graph (PrimeKG in Neo4j) into
one read-only Cypher query.

{schema_prompt()}

Rules:
- Use only the labels and relationship types above. Match relationships undirected unless the
  direction matters.
- Match entities by the integer ids given in the "Linked entities" section, never by name.
- Return node names and ids, not whole nodes. Return at most {MAX_LIMIT} rows.
- Read-only: no CREATE, MERGE, DELETE, SET, REMOVE, CALL procedures or LOAD CSV.
- Output only the Cypher query, without explanation or code fences."""


class CypherRejected(ValueError):
    pass


class Cypher(BaseModel):
    cypher: str = Field(description="One read-only Cypher query")


def _strip(query: str) -> str:
    return _STRINGS.sub("''", _COMMENTS.sub(" ", query))


def guard(query: str) -> str:
    """Return the query (with a LIMIT) if it is allowed, else raise CypherRejected."""
    query = query.strip().strip("`").removeprefix("cypher").strip().rstrip(";").strip()
    if not query:
        raise CypherRejected("empty query")
    code = _strip(query)
    if ";" in code:
        raise CypherRejected("multiple statements")
    if m := _WRITE.search(code):
        raise CypherRejected(f"write or procedure clause not allowed: {m.group(0).strip()}")
    labels = {
        label.strip(" `")
        for group in _NODE_LABELS.findall(code)
        for label in group.split(":")
        if label.strip(" `")
    }
    if bad := labels - set(LABELS):
        raise CypherRejected(f"unknown label(s): {', '.join(sorted(bad))}")
    types = {t.strip(" `:") for group in _REL_TYPES.findall(code) for t in group.split("|")}
    if bad := types - set(REL_TYPES):
        raise CypherRejected(f"unknown relationship type(s): {', '.join(sorted(bad))}")
    if m := _LIMIT.search(query):
        if int(m.group(1)) > MAX_LIMIT:
            query = query[: m.start()] + f"LIMIT {MAX_LIMIT}"
    else:
        query += f"\nLIMIT {MAX_LIMIT}"
    return query


@dataclass
class Text2CypherResult:
    cypher: str | None
    rows: list[dict]
    attempts: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.cypher is not None


def run_guarded(kg: KGClient, query: str) -> tuple[str, list[dict]]:
    query = guard(query)
    kg.read(f"EXPLAIN {query}", timeout=TIMEOUT)
    return query, kg.read(query, timeout=TIMEOUT)


def text2cypher(
    provider: Provider, kg: KGClient, question: str, links: str = "", retries: int = 1
) -> Text2CypherResult:
    prompt = f"Linked entities:\n{links or '(none)'}\n\nQuestion: {question}"
    attempts: list[dict] = []
    for _ in range(retries + 1):
        query = provider.structured(SYSTEM, prompt, Cypher).cypher
        try:
            query, rows = run_guarded(kg, query)
            attempts.append({"cypher": query, "error": None})
            return Text2CypherResult(query, rows, attempts)
        except Exception as e:  # guard rejection or Neo4j error: show it to the model once
            error = f"{type(e).__name__}: {str(e)[:300]}"
            attempts.append({"cypher": query, "error": error})
            prompt += f"\n\nYour previous query:\n{query}\nfailed with:\n{error}\nFix it."
    return Text2CypherResult(None, [], attempts)
