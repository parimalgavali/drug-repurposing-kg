"""Evidence paths for a (drug, disease) pair: paths of length 2–3, grouped by metapath and ranked
by degree-weighted path count (DWPC, Himmelstein et al. 2017, as in Hetionet).

    uv run --extra graph python -m explain.paths --drug Salbutamol --disease asthma
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kg.client import KGClient

# Mechanistic and similarity edges. DRUG_DRUG (interactions), anatomy expression and GO-term
# edges are left out: they are dense and say little about why a drug would treat a disease.
REL_TYPES = (
    "TARGET",
    "ENZYME",
    "TRANSPORTER",
    "CARRIER",
    "DISEASE_PROTEIN",
    "PROTEIN_PROTEIN",
    "DRUG_EFFECT",
    "DISEASE_PHENOTYPE_POSITIVE",
    "PHENOTYPE_PHENOTYPE",
    "PHENOTYPE_PROTEIN",
    "DISEASE_DISEASE",
    "PATHWAY_PROTEIN",
    "INDICATION",
    "OFF_LABEL_USE",
)
DAMPING = 0.4
PATH_LIMIT = 20_000

PATHS = f"""
MATCH (a:Drug {{id: $drug}}), (b:Disease {{id: $disease}})
MATCH p = (a)-[:{"|".join(REL_TYPES)}*2..3]-(b)
WHERE all(i IN range(0, length(p) - 1) WHERE NOT nodes(p)[i] IN nodes(p)[i + 1..])
WITH p LIMIT $limit
RETURN [n IN nodes(p) | {{key: elementId(n), id: n.id, name: n.name,
                          label: labels(n)[0]}}] AS nodes,
       [r IN relationships(p) | type(r)] AS rels
"""
DEGREES = """
UNWIND $items AS item
MATCH (n) WHERE elementId(n) = item.key
RETURN item.key AS key, item.type AS type, COUNT { (n)-[x]-() WHERE type(x) = item.type } AS degree
"""


@dataclass
class Path:
    nodes: list[dict]
    rels: list[str]
    pdp: float = 0.0

    @property
    def metapath(self) -> str:
        parts = [self.nodes[0]["label"]]
        for rel, node in zip(self.rels, self.nodes[1:], strict=True):
            parts += [rel, node["label"]]
        return "-".join(parts)

    def text(self) -> str:
        out = self.nodes[0]["name"]
        for rel, node in zip(self.rels, self.nodes[1:], strict=True):
            out += f" -{rel}- {node['name']}"
        return out


@dataclass
class Metapath:
    metapath: str
    dwpc: float
    count: int
    paths: list[Path] = field(default_factory=list)


def path_degree_product(
    path: Path, degree: dict[tuple[str, str], int], w: float = DAMPING
) -> float:
    """Product over every edge of (deg_type(source) · deg_type(target))^-w."""
    pdp = 1.0
    for i, rel in enumerate(path.rels):
        for node in (path.nodes[i], path.nodes[i + 1]):
            pdp *= degree[(node["key"], rel)] ** -w
    return pdp


def rank_metapaths(
    paths: list[Path],
    degree: dict[tuple[str, str], int],
    top: int = 3,
    examples: int = 5,
    w: float = DAMPING,
) -> list[Metapath]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        path.pdp = path_degree_product(path, degree, w)
        groups[path.metapath].append(path)
    ranked = sorted(
        (Metapath(mp, sum(p.pdp for p in ps), len(ps), ps) for mp, ps in groups.items()),
        key=lambda m: (-m.dwpc, m.metapath),
    )[:top]
    for m in ranked:
        m.paths = sorted(m.paths, key=lambda p: -p.pdp)[:examples]
    return ranked


def to_cytoscape(metapaths: list[Metapath]) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    for m in metapaths:
        for path in m.paths:
            for n in path.nodes:
                nodes[str(n["id"])] = {
                    "data": {"id": str(n["id"]), "label": n["name"], "type": n["label"]}
                }
            for i, rel in enumerate(path.rels):
                a, b = str(path.nodes[i]["id"]), str(path.nodes[i + 1]["id"])
                key = f"{a}-{rel}-{b}"
                edges[key] = {"data": {"id": key, "source": a, "target": b, "label": rel}}
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def explain(
    kg: KGClient, drug: int, disease: int, top: int = 3, examples: int = 5, limit: int = PATH_LIMIT
) -> dict:
    rows = kg.read(PATHS, {"drug": drug, "disease": disease, "limit": limit}, timeout=30)
    paths = [Path(r["nodes"], r["rels"]) for r in rows]
    items = {
        (n["key"], rel): {"key": n["key"], "type": rel}
        for p in paths
        for i, rel in enumerate(p.rels)
        for n in (p.nodes[i], p.nodes[i + 1])
    }
    degree = {
        (r["key"], r["type"]): r["degree"]
        for r in kg.read(DEGREES, {"items": list(items.values())}, timeout=30)
    }
    ranked = rank_metapaths(paths, degree, top, examples)
    return {
        "drug": drug,
        "disease": disease,
        "paths_found": len(paths),
        "truncated": len(paths) >= limit,
        "metapaths": [
            {
                "metapath": m.metapath,
                "dwpc": m.dwpc,
                "count": m.count,
                "paths": [
                    {
                        "nodes": [{k: n[k] for k in ("id", "name", "label")} for n in p.nodes],
                        "rels": p.rels,
                        "pdp": p.pdp,
                        "text": p.text(),
                    }
                    for p in m.paths
                ],
            }
            for m in ranked
        ],
        "cytoscape": to_cytoscape(ranked),
    }


def find_id(kg: KGClient, label: str, name: str) -> int:
    rows = kg.read(
        f"MATCH (n:{label}) WHERE toLower(n.name) = toLower($name) RETURN n.id AS id LIMIT 1",
        {"name": name},
    )
    if not rows:
        raise SystemExit(f"no {label} named {name!r}")
    return rows[0]["id"]


def main() -> None:
    from kg.client import KGClient

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drug", required=True)
    parser.add_argument("--disease", required=True)
    parser.add_argument("--json", action="store_true", help="print the full JSON (incl. Cytoscape)")
    args = parser.parse_args()
    with KGClient() as kg:
        result = explain(kg, find_id(kg, "Drug", args.drug), find_id(kg, "Disease", args.disease))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"{args.drug} → {args.disease}: {result['paths_found']} paths")
    for i, m in enumerate(result["metapaths"], 1):
        print(f"{i}. {m['metapath']}  DWPC={m['dwpc']:.4g}  ({m['count']} paths)")
        for p in m["paths"]:
            print(f"     {p['text']}")


if __name__ == "__main__":
    main()
