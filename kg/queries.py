"""Example Cypher queries in kg/queries/*.cypher, each with `// name:`, `// description:` and
`// params:` (JSON example parameters) header lines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUERY_DIR = Path(__file__).with_name("queries")


@dataclass(frozen=True)
class ExampleQuery:
    name: str
    description: str
    params: dict[str, Any]
    cypher: str


def parse_query(text: str) -> ExampleQuery:
    meta: dict[str, str] = {}
    body = []
    for line in text.splitlines():
        if line.startswith("// ") and ":" in line and not body:
            key, _, value = line[3:].partition(":")
            meta[key.strip()] = value.strip()
        else:
            body.append(line)
    return ExampleQuery(
        name=meta["name"],
        description=meta["description"],
        params=json.loads(meta.get("params", "{}")),
        cypher="\n".join(body).strip(),
    )


def load_queries(directory: Path = QUERY_DIR) -> list[ExampleQuery]:
    return [parse_query(p.read_text()) for p in sorted(directory.glob("*.cypher"))]
