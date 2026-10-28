"""Read-only SPARQL over an in-memory Oxigraph store."""

from __future__ import annotations

import re
from pathlib import Path

import pyoxigraph as ox

MAX_ROWS = 500
_COMMENT = re.compile(r"(?m)^\s*#.*$")
_READ = re.compile(r"^\s*((PREFIX\s+\S*:\s*<[^>]*>|BASE\s+<[^>]*>)\s*)*(SELECT|ASK)\b", re.I)
_SERVICE = re.compile(r"\bSERVICE\b", re.I)


class SparqlError(ValueError):
    pass


def _term(t: ox.NamedNode | ox.BlankNode | ox.Literal) -> dict:
    if isinstance(t, ox.Literal):
        return {"type": "literal", "value": t.value, "datatype": t.datatype.value}
    return {"type": "uri" if isinstance(t, ox.NamedNode) else "bnode", "value": t.value}


class RdfStore:
    def __init__(self, turtle: str | bytes) -> None:
        self._store = ox.Store()
        data = turtle.encode() if isinstance(turtle, str) else turtle
        self._store.load(data, format=ox.RdfFormat.TURTLE)

    @classmethod
    def from_path(cls, path: Path) -> RdfStore:
        return cls(path.read_bytes())

    def __len__(self) -> int:
        return len(self._store)

    def query(self, query: str) -> dict:
        """SELECT → {vars, rows, truncated}; ASK → {boolean}. Anything else is rejected."""
        text = _COMMENT.sub("", query)
        if not _READ.match(text):
            raise SparqlError("only SELECT and ASK queries are allowed")
        if _SERVICE.search(text):
            raise SparqlError("federated SERVICE queries are not allowed")
        try:
            result = self._store.query(text)
        except SyntaxError as e:
            raise SparqlError(f"invalid SPARQL: {e}") from e
        if isinstance(result, ox.QueryBoolean):
            return {"boolean": bool(result)}
        names = [v.value for v in result.variables]
        rows, truncated = [], False
        for i, solution in enumerate(result):
            if i == MAX_ROWS:
                truncated = True
                break
            rows.append({n: _term(solution[n]) for n in names if solution[n] is not None})
        return {"vars": names, "rows": rows, "truncated": truncated}
