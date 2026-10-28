"""Export a static snapshot of the app's API responses for the GitHub Pages build.

Needs the full-graph API with live Claude running (e.g. on :8001) and Neo4j credentials:

    LLM_PROVIDER=anthropic uv run --extra graph --extra llm --extra api --extra rdf \
        uvicorn api.main:app --port 8001
    PYTHONPATH=. uv run --extra graph python scripts/export_static.py --api http://localhost:8001

Writes `site-data/` (search index, every disease's candidates, evidence graphs for the top
candidates of the most-studied diseases, saved example answers); `make pages-data` tars it for
the `static-snapshot` release that `.github/workflows/pages.yml` deploys.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import httpx

from kg.client import KGClient

# Also listed in web/src/components/AskPanel.vue (EXAMPLES).
EXAMPLES = [
    "Which genes does Ibuprofen target?",
    "Which drugs could be repurposed for asthma?",
    "Why might Salbutamol work for asthma?",
]
FEATURED = [33675, 27158]  # asthma; osteogenesis imperfecta (no known drugs)
DISEASES = "MATCH (s:Disease) RETURN s.id AS id, s.name AS name, s.name_de AS name_de"
MOST_TREATED = """
MATCH (:Drug)-[:INDICATION]->(s:Disease)
RETURN s.id AS id, count(*) AS n ORDER BY n DESC, id LIMIT $k
"""
MOST_GENES = """
MATCH (s:Disease) WHERE NOT (:Drug)-[:INDICATION]->(s)
MATCH (s)-[:DISEASE_PROTEIN|DISEASE_PHENOTYPE_POSITIVE]-()
RETURN s.id AS id, count(*) AS n ORDER BY n DESC, id LIMIT $k
"""


def sse_events(text: str) -> list[dict]:
    """Parse a text/event-stream body (single-line JSON data fields) into events."""
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        event, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if data:
            events.append({"event": event, "data": json.loads("\n".join(data))})
    return events


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8001")
    parser.add_argument("--out", type=Path, default=Path("site-data"))
    parser.add_argument("--treated", type=int, default=300, help="most-treated diseases explained")
    parser.add_argument("--untreated", type=int, default=100, help="no-drug diseases explained")
    parser.add_argument("--top", type=int, default=10, help="candidates explained per disease")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    kg = KGClient()
    diseases = kg.read(DISEASES)
    write(args.out / "search.json", [[d["id"], d["name"], d["name_de"]] for d in diseases])
    print(f"search index: {len(diseases)} diseases")

    client = httpx.Client(base_url=args.api, timeout=120)

    def get(path: str) -> dict:
        r = client.get(path)
        r.raise_for_status()
        return r.json()

    def export_candidates(disease: int) -> list[int]:
        body = get(f"/api/diseases/{disease}/candidates?limit=20")
        write(args.out / "candidates" / f"{disease}.json", body)
        return [c["drug_id"] for c in body["candidates"]]

    with ThreadPoolExecutor(args.workers) as pool:
        tops = dict(
            zip(
                [d["id"] for d in diseases],
                pool.map(export_candidates, [d["id"] for d in diseases]),
                strict=True,
            )
        )
    print(f"candidates: {len(tops)} diseases")

    chosen = list(
        dict.fromkeys(
            FEATURED
            + [r["id"] for r in kg.read(MOST_TREATED, {"k": args.treated})]
            + [r["id"] for r in kg.read(MOST_GENES, {"k": args.untreated})]
        )
    )
    pairs = [(drug, d) for d in chosen for drug in tops.get(d, [])[: args.top]]

    def export_explain(pair: tuple[int, int]) -> None:
        drug, disease = pair
        write(
            args.out / "explain" / f"{drug}_{disease}.json",
            get(f"/api/pairs/{drug}/{disease}/explain"),
        )

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(export_explain, pairs))
    explained: dict[int, list[int]] = {}
    for drug, disease in pairs:
        explained.setdefault(disease, []).append(drug)
    write(args.out / "explained.json", explained)
    print(f"evidence graphs: {len(pairs)} pairs over {len(chosen)} diseases")

    answers = {}
    for lang in ("en", "de"):
        for i, question in enumerate(EXAMPLES):
            r = client.post("/api/ask", json={"question": question, "lang": lang})
            r.raise_for_status()
            name = f"{i}-{lang}.json"
            write(args.out / "answers" / name, sse_events(r.text))
            answers.setdefault(lang, {})[question] = name
    write(args.out / "answers" / "index.json", answers)
    write(args.out / "meta.json", {"generated": date.today().isoformat(), "pairs": len(pairs)})
    print("saved answers:", sum(len(v) for v in answers.values()))


if __name__ == "__main__":
    main()
