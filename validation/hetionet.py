"""Hetionet cross-check: do top-k predictions recover Hetionet `CtD` (treats) pairs that PrimeKG
does not record as an indication or off-label use?

    uv run --extra ml --extra graph python -m validation.hetionet fetch
    uv run --extra ml --extra graph python -m validation.hetionet

Drugs map by DrugBank id; diseases map DOID → MONDO via MONDO's curated exact-match SSSOM set,
then to every PrimeKG disease node containing that MONDO id (grouped nodes included).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from validation.run import MODEL_CARD, random_top, wilson

RAW = Path("data/raw/hetionet")
RESULTS = Path("ml/results/hetionet.json")
HETIONET_COMMIT = "8a6cc0c5604d0e2908786e4de38af67b8e46ee4a"
MONDO_COMMIT = "65a4751b3e8957ed0a495cad34cd7090cb3c33e3"  # mapping set 2026-10-06
SOURCES = {
    "edges.sif.gz": (
        f"https://github.com/hetio/hetionet/raw/{HETIONET_COMMIT}"
        "/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
        "611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947",
    ),
    "mondo_doid.sssom.tsv": (
        f"https://raw.githubusercontent.com/monarch-initiative/mondo/{MONDO_COMMIT}"
        "/src/ontology/mappings/mondo_exactmatch_doid.sssom.tsv",
        "905e51c336e48c0896e9e2bd7de3200eed3aae1cb9ff772c82fa3ef3c59e0b56",
    ),
}
USES = ["INDICATION", "OFF_LABEL_USE"]
SERVED = """
MATCH (d:Drug)-[p:PREDICTED_INDICATION]->(s:Disease)
WHERE s.id IN $diseases
RETURN d.id AS drug, s.id AS disease, p.rank AS rank
"""


def fetch(raw: Path = RAW) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in SOURCES.items():
        path = raw / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha:
            raise SystemExit(f"{name}: sha256 {digest} != {sha}")
        print(f"ok {name}")


def read_ctd(path: Path) -> list[tuple[str, str]]:
    """(DrugBank id, DOID) for every Compound–treats–Disease edge."""
    with gzip.open(path, "rt") as f:
        rows = (line.rstrip("\n").split("\t") for line in f)
        return [
            (s.removeprefix("Compound::"), t.removeprefix("Disease::"))
            for s, rel, t in rows
            if rel == "CtD"
        ]


def read_doid_mondo(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with path.open() as f:
        for row in csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"):
            if row["predicate_id"] == "skos:exactMatch":
                out.setdefault(row["object_id"], []).append(int(row["subject_id"].split(":")[1]))
    return out


def map_pairs(
    ctd: list[tuple[str, str]], doid_mondo: dict[str, list[int]], nodes: pl.DataFrame
) -> pl.DataFrame:
    drugs = dict(
        nodes.filter(pl.col("node_source") == "DrugBank")
        .select("node_id", "node_index")
        .iter_rows()
    )
    by_mondo: dict[int, set[int]] = {}
    for node_id, index in (
        nodes.filter(pl.col("node_type") == "disease").select("node_id", "node_index").iter_rows()
    ):
        for member in node_id.split("_"):
            by_mondo.setdefault(int(member), set()).add(index)
    rows = {
        (drugs[db], disease)
        for db, doid in ctd
        if db in drugs
        for mondo in doid_mondo.get(doid, [])
        for disease in by_mondo.get(mondo, ())
    }
    return pl.DataFrame(sorted(rows), schema=["drug", "disease"], orient="row")


def recorded(edges: pl.DataFrame, rel_types: list[str]) -> pl.DataFrame:
    """(drug, disease) pairs with any of `rel_types`, whichever way the edge is stored."""
    e = edges.filter(pl.col("rel_type").is_in(rel_types))
    return pl.concat(
        [
            e.filter(pl.col("src_type") == "drug").select(
                pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease")
            ),
            e.filter(pl.col("dst_type") == "drug").select(
                pl.col("dst_index").alias("drug"), pl.col("src_index").alias("disease")
            ),
        ]
    ).unique()


def evaluate(tops: dict[str, pl.DataFrame], truth: pl.DataFrame, ks: list[int]) -> dict:
    """Precision (with Wilson CI) and recall of each method's top-k against `truth`."""
    truth_set = set(truth.select("drug", "disease").iter_rows())
    out: dict = {}
    for method, top in tops.items():
        out[method] = {}
        for k in ks:
            picked = top.filter(pl.col("rank") <= k).select("drug", "disease").iter_rows()
            pairs = list(picked)
            hits = sum(p in truth_set for p in pairs)
            out[method][f"top{k}"] = {
                "pairs": len(pairs),
                "hits": hits,
                "precision": hits / len(pairs) if pairs else 0.0,
                "ci": wilson(hits, len(pairs)),
                "recall": hits / len(truth_set) if truth_set else 0.0,
            }
    return out


def card_section(results: dict) -> str:
    c = results["counts"]
    lines = [
        f"Hetionet v1.0 has {c['ctd']} `CtD` edges; {c['mapped']} map to PrimeKG "
        "drug–disease pairs. "
        f"Of those, {c['in_primekg']} are already PrimeKG indications or off-label uses and "
        f"{c['contraindicated']} are PrimeKG *contraindications*. The remaining "
        f"**{c['hetionet_only']} Hetionet-only pairs** over {c['diseases']} diseases are the "
        "truth set; every method ranks only drugs without a recorded use.",
        "",
        "| Method | Hits in top 20 | Precision@20 (95% CI) | Recall@20 "
        "| Hits in top 50 | Recall@50 |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for name, r in results["methods"].items():
        t20, t50 = r["top20"], r["top50"]
        lo, hi = t20["ci"]
        lines.append(
            f"| {name} | {t20['hits']} | {t20['precision']:.3f} ({lo:.3f}–{hi:.3f}) "
            f"| {t20['recall']:.3f} | {t50['hits']} | {t50['recall']:.3f} |"
        )
    return "\n".join(lines)


def update_card(section: str, card: Path = MODEL_CARD) -> None:
    start, end = "<!-- hetionet:start -->", "<!-- hetionet:end -->"
    text = card.read_text()
    if start not in text:
        intro = "Produced by `validation/hetionet.py`."
        text += f"\n## Hetionet cross-check\n\n{intro}\n\n{start}\n{end}\n"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    card.write_text(f"{head}{start}\n{section}\n{end}{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", choices=["fetch", "run"], default="run")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
        return

    from kg.client import KGClient
    from ml.baselines import popularity, similar_disease
    from ml.run import get_split
    from ml.write_predictions import top_predictions

    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    edges = pl.read_parquet(args.processed / "edges.parquet")
    ctd = read_ctd(RAW / "edges.sif.gz")
    pairs = map_pairs(ctd, read_doid_mondo(RAW / "mondo_doid.sssom.tsv"), nodes)
    known = recorded(edges, USES)
    truth = pairs.join(known, on=["drug", "disease"], how="anti")
    contra = pairs.join(recorded(edges, ["CONTRAINDICATION"]), on=["drug", "disease"])
    diseases = np.sort(truth["disease"].unique().to_numpy())

    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    all_diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    # The served model was selected on the random split, so baselines are fit on it too.
    split = get_split("random", args.seed, args.processed, args.processed / "splits")
    with KGClient() as kg:
        served = pl.DataFrame(kg.read(SERVED, {"diseases": [int(d) for d in diseases]}))
    # Served predictions only exclude INDICATION; drop off-label uses too and re-rank.
    served = (
        served.join(known, on=["drug", "disease"], how="anti")
        .sort("disease", "rank")
        .with_columns((pl.int_range(pl.len()).over("disease") + 1).alias("rank"))
    )
    tops = {"served GNN": served}
    for name, fn in [("similar_disease", similar_disease), ("popularity", popularity)]:
        tops[name] = top_predictions(fn(split, drugs, all_diseases), drugs, diseases, known, k=50)
    rand = random_top(drugs, diseases, known, 50, args.seed)
    tops["random drugs"] = rand.with_columns(
        (pl.int_range(pl.len()).over("disease") + 1).alias("rank")
    )

    results = {
        "hetionet_commit": HETIONET_COMMIT,
        "mondo_commit": MONDO_COMMIT,
        "split": "random",
        "seed": args.seed,
        "counts": {
            "ctd": len(ctd),
            "mapped": pairs.height,
            "in_primekg": pairs.height - truth.height,
            "contraindicated": contra.height,
            "hetionet_only": truth.height,
            "diseases": len(diseases),
        },
        "methods": evaluate(tops, truth, [20, 50]),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    update_card(card_section(results))
    for name, r in results["methods"].items():
        print(f"{name:16s} top20 {r['top20']['hits']:3d}  top50 {r['top50']['hits']:3d}")


if __name__ == "__main__":
    main()
