"""German names for drugs and diseases from Wikidata (CC0), matched on DrugBank id (P715) and
MONDO id (P5270). Diseases without a MONDO id on Wikidata fall back to their Disease Ontology id
(P699) via MONDO's curated exact DOID matches. The fetched table is committed so demos and CI
work offline.

    uv run --extra data --extra graph python -m kg.labels_de fetch   # refresh the CSV
    uv run --extra data --extra graph python -m kg.labels_de apply   # set name_de in Neo4j
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

LABELS = Path("data/labels/labels_de.csv")
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "RepurposeKG/0.1 (https://github.com/parimalgavali/drug-repurposing-kg)"
PROPERTIES = {"DrugBank": "P715", "MONDO": "P5270", "DOID": "P699"}
MONDO_COMMIT = "65a4751b3e8957ed0a495cad34cd7090cb3c33e3"
MONDO_DOID = (
    f"https://raw.githubusercontent.com/monarch-initiative/mondo/{MONDO_COMMIT}"
    "/src/ontology/mappings/mondo_exactmatch_doid.sssom.tsv"
)
SPARQL = (
    "SELECT ?id ?de WHERE {{ ?item wdt:{prop} ?id . "
    "?item rdfs:label ?de FILTER(lang(?de) = 'de') }}"
)
SOURCE_LABEL = {"DrugBank": "Drug", "MONDO": "Disease"}
NODES = """
MATCH (n:Drug|Disease) WHERE n.source IN ['DrugBank', 'MONDO']
RETURN n.id AS id, n.source AS source, n.source_id AS source_id
"""
SET = "UNWIND $rows AS r MATCH (n:{label} {{id: r.id}}) SET n.name_de = r.name_de"


def normalise(source: str, raw: str) -> str | None:
    """Wikidata id → PrimeKG `source_id` ("MONDO:0004979" → "4979", "DB01050" unchanged)."""
    raw = raw.strip()
    if source == "DOID":
        return raw if re.fullmatch(r"DOID:\d+", raw) else None
    if source == "MONDO":
        m = re.fullmatch(r"MONDO[:_](\d{7})", raw)
        return str(int(m.group(1))) if m else None
    return raw if re.fullmatch(r"DB\d{5}", raw) else None


def best_labels(rows: list[tuple[str, str, str]]) -> dict[tuple[str, str], str]:
    """One label per (source, id): several Wikidata items can share an id; the shortest label
    wins, ties alphabetical, so refetches are stable."""
    best: dict[tuple[str, str], str] = {}
    for source, raw, label in rows:
        source_id = normalise(source, raw)
        if source_id is None or not label.strip():
            continue
        key, label = (source, source_id), label.strip()
        if key not in best or (len(label), label) < (len(best[key]), best[key]):
            best[key] = label
    return best


def pick(rows: list[tuple[str, str, str]], doid_mondo: dict[str, list[str]]) -> pl.DataFrame:
    """Labels keyed by PrimeKG (source, source_id), with `via` = the Wikidata property used.
    A DOID label fills a MONDO id only if no item carries that MONDO id directly."""
    best = best_labels(rows)
    out = {(s, i): (n, PROPERTIES[s]) for (s, i), n in best.items() if s != "DOID"}
    for (source, doid), name in sorted(best.items()):
        if source != "DOID":
            continue
        for mondo in doid_mondo.get(doid, []):
            out.setdefault(("MONDO", mondo), (name, PROPERTIES["DOID"]))
    rows = [(s, i, n, via) for (s, i), (n, via) in out.items()]
    return pl.DataFrame(rows, schema=["source", "source_id", "name_de", "via"], orient="row").sort(
        "source", "source_id"
    )


def read_doid_mondo(text: str) -> dict[str, list[str]]:
    """DOID → PrimeKG MONDO source_ids from MONDO's exact-match SSSOM table."""
    out: dict[str, list[str]] = {}
    lines = (line for line in text.splitlines() if not line.startswith("#"))
    for row in csv.DictReader(lines, delimiter="\t"):
        if row["predicate_id"] == "skos:exactMatch":
            mondo = normalise("MONDO", row["subject_id"])
            if mondo:
                out.setdefault(row["object_id"], []).append(mondo)
    return out


def _get(url: str, accept: str = "*/*") -> str:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()


def fetch(path: Path = LABELS) -> pl.DataFrame:
    rows = []
    for source, prop in PROPERTIES.items():
        url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': SPARQL.format(prop=prop)})}"
        reader = csv.DictReader(io.StringIO(_get(url, "text/csv")))
        rows += [(source, row["id"], row["de"]) for row in reader]
    labels = pick(rows, read_doid_mondo(_get(MONDO_DOID)))
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.write_csv(path)
    return labels


def read_labels(path: Path = LABELS) -> pl.DataFrame:
    return pl.read_csv(path, schema_overrides={"source_id": pl.String}).drop("via")


def match(nodes: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Graph nodes (id, source, source_id) joined to their German label. Grouped MONDO nodes have
    source "MONDO_grouped" and get none."""
    return nodes.join(labels, on=["source", "source_id"]).select("id", "source", "name_de")


def apply(driver, labels: pl.DataFrame) -> int:
    nodes = pl.DataFrame(
        driver.execute_query(NODES).records, schema=["id", "source", "source_id"], orient="row"
    )
    matched = match(nodes, labels)
    for source, label in SOURCE_LABEL.items():
        rows = matched.filter(pl.col("source") == source).select("id", "name_de").to_dicts()
        for start in range(0, len(rows), 5000):
            driver.execute_query(SET.format(label=label), rows=rows[start : start + 5000])
    return matched.height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "apply"])
    args = parser.parse_args()
    if args.command == "fetch":
        labels = fetch()
        print(f"wrote {labels.height} labels to {LABELS}")
        return
    from kg.client import connect

    driver = connect()
    try:
        print(f"set name_de on {apply(driver, read_labels())} nodes")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
