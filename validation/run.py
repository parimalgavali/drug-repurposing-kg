"""Clinical-trial check: do the model's novel top-k drugs for held-out diseases have registered
studies more often than popularity's or random drugs'? Also stores trial counts on the served
PREDICTED_INDICATION relationships of the sampled diseases.

    uv run --extra ml --extra graph python -m validation.run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from validation.ctgov import CTGov

RESULTS = Path("ml/results/ctgov.json")
MODEL_CARD = Path("ml/model_card.md")
CHECKPOINT = Path("artifacts/gnn/disease_seed0_text.pt")
ANNOTATE = """
UNWIND $rows AS row
MATCH (:Drug {id: row.drug})-[p:PREDICTED_INDICATION]->(:Disease {id: row.disease})
SET p.ct_studies = row.studies, p.ct_max_phase = row.max_phase
"""
SERVED = """
MATCH (d:Drug)-[p:PREDICTED_INDICATION]->(s:Disease)
WHERE s.id IN $diseases AND p.rank <= $k
RETURN d.id AS drug, d.name AS drug_name, s.id AS disease, s.name AS disease_name
"""


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def random_top(drugs: np.ndarray, diseases: np.ndarray, known: pl.DataFrame, k: int, seed: int):
    rng = np.random.default_rng(seed)
    known_set = set(zip(known["drug"], known["disease"], strict=True))
    rows = []
    for disease in diseases:
        pool = [d for d in drugs if (d, disease) not in known_set]
        rows += [(int(d), int(disease)) for d in rng.choice(pool, size=k, replace=False)]
    return pl.DataFrame(rows, schema=["drug", "disease"], orient="row")


def card_table(results: dict) -> str:
    lines = [
        "| Method | Pairs | With ≥ 1 study | Share (95% Wilson CI) | Max phase ≥ 2 |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for name, r in results["methods"].items():
        lo, hi = r["ci"]
        lines.append(
            f"| {name} | {r['pairs']} | {r['hits']} | {r['share']:.3f} ({lo:.3f}–{hi:.3f}) "
            f"| {r['phase2_plus']} |"
        )
    return "\n".join(lines)


def update_card(table: str, card: Path = MODEL_CARD) -> None:
    start, end = "<!-- ctgov:start -->", "<!-- ctgov:end -->"
    text = card.read_text()
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    card.write_text(f"{head}{start}\n{table}\n{end}{tail}")


def main() -> None:
    from ml.baselines import popularity, similar_disease
    from ml.gnn import load_checkpoint, score_fn
    from ml.run import get_split
    from ml.write_predictions import top_predictions

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diseases", type=int, default=30, help="held-out diseases to sample")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--annotate", type=int, default=10, help="served ranks to annotate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    edges = pl.read_parquet(args.processed / "edges.parquet")
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    all_diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    name = dict(zip(nodes["node_index"], nodes["node_name"], strict=True))
    split = get_split("disease", args.seed, args.processed, args.processed / "splits")
    rng = np.random.default_rng(args.seed)
    test = np.sort(split.test["disease"].unique().to_numpy())
    sample = np.sort(rng.choice(test, size=min(args.diseases, len(test)), replace=False))
    # Every recorded use is excluded, so only novel pairs are checked.
    known = edges.filter(pl.col("rel_type").is_in(["INDICATION", "OFF_LABEL_USE"])).select(
        pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease")
    )

    model, graph = load_checkpoint(CHECKPOINT, split, nodes)
    scorers = {
        "gnn (text)": score_fn(model, graph, drugs),
        "similar_disease": similar_disease(split, drugs, all_diseases),
        "popularity": popularity(split, drugs, all_diseases),
    }
    tops = {m: top_predictions(s, drugs, sample, known, k=args.k) for m, s in scorers.items()}
    tops["random drugs"] = random_top(drugs, sample, known, args.k, args.seed)

    ct = CTGov()
    matched = {int(d) for d in sample if ct.lookup(name[int(d)], "").studies > 0}
    methods = {}
    for method, top in tops.items():
        found = [
            ct.lookup(name[d], name[s]) for d, s in zip(top["drug"], top["disease"], strict=True)
        ]
        hits = sum(f.studies > 0 for f in found)
        methods[method] = {
            "pairs": len(found),
            "hits": hits,
            "share": hits / len(found),
            "ci": wilson(hits, len(found)),
            "phase2_plus": sum((f.max_phase or 0) >= 2 for f in found),
        }
        print(f"{method:16s} {hits}/{len(found)} pairs with a registered study", flush=True)
    results = {
        "split": "disease",
        "seed": args.seed,
        "k": args.k,
        "diseases": len(sample),
        "diseases_with_any_study": len(matched),
        "methods": methods,
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    update_card(card_table(results))

    from kg.client import KGClient, connect

    with KGClient() as kg:
        served = kg.read(SERVED, {"diseases": [int(d) for d in sample], "k": args.annotate})
    rows = []
    for r in served:
        f = ct.lookup(r["disease_name"], r["drug_name"])
        rows.append({**r, "studies": f.studies, "max_phase": f.max_phase})
    with connect() as driver, driver.session() as s:
        s.run(ANNOTATE, rows=rows).consume()
    print(f"annotated {len(rows)} served predictions; {ct.requests} CT.gov requests")


if __name__ == "__main__":
    main()
