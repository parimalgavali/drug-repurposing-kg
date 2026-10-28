"""Write each disease's top-k non-indicated drugs to Neo4j as
(:Drug)-[:PREDICTED_INDICATION {score, rank, model_version}]->(:Disease).

Existing PREDICTED_INDICATION relationships are deleted first, so reruns are idempotent.

    uv run --extra ml --extra graph python -m ml.write_predictions --model gnn --split random
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:  # keeps the Neo4j CI job free of the ml extra
    from neo4j import Driver

    from ml.evaluate import ScoreFn

DELETE = """
MATCH ()-[r:PREDICTED_INDICATION]->()
CALL (r) { DELETE r } IN TRANSACTIONS OF 10000 ROWS
"""
CREATE = """
UNWIND $rows AS row
MATCH (d:Drug {id: row.drug}), (s:Disease {id: row.disease})
CREATE (d)-[:PREDICTED_INDICATION {score: row.score, rank: row.rank,
                                   model_version: $version}]->(s)
"""


def top_predictions(
    score: ScoreFn,
    drugs: np.ndarray,
    diseases: np.ndarray,
    known: pl.DataFrame,
    k: int = 50,
    batch: int = 512,
) -> pl.DataFrame:
    drug_pos = {int(d): i for i, d in enumerate(drugs)}
    known_by: dict[int, list[int]] = {}
    for (disease,), g in known.group_by("disease"):
        known_by[int(disease)] = [drug_pos[d] for d in g["drug"]]
    frames = []
    for start in range(0, len(diseases), batch):
        chunk = diseases[start : start + batch]
        scores = np.asarray(score(chunk), dtype=np.float64).copy()
        for row, disease in zip(scores, chunk, strict=True):
            row[known_by.get(int(disease), [])] = -np.inf
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        frames.append(
            pl.DataFrame(
                {
                    "disease": np.repeat(chunk, k),
                    "drug": drugs[top].ravel(),
                    "score": np.take_along_axis(scores, top, axis=1).ravel(),
                    "rank": np.tile(np.arange(1, k + 1), len(chunk)),
                }
            )
        )
    preds = pl.concat(frames)
    return preds.filter(pl.col("score").is_finite())


def write_predictions(
    driver: Driver,
    preds: pl.DataFrame,
    version: str,
    database: str = "neo4j",
    batch: int = 10_000,
) -> int:
    rows = preds.select("drug", "disease", "score", "rank").to_dicts()
    with driver.session(database=database) as session:
        session.run(DELETE).consume()
        for i in range(0, len(rows), batch):
            session.run(CREATE, rows=rows[i : i + batch], version=version).consume()
        query = "MATCH ()-[r:PREDICTED_INDICATION]->() RETURN count(r) AS n"
        return session.run(query).single()["n"]


def main() -> None:
    from kg.client import connect
    from ml.run import get_split, models

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gnn")
    parser.add_argument("--split", default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    edges = pl.read_parquet(args.processed / "edges.parquet")
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    split = get_split(args.split, args.seed, args.processed, args.processed / "splits")
    variant = {"gnn": "text", "gnn_notext": "notext"}.get(args.model, "")
    ckpt = Path(f"artifacts/gnn/{args.split}_seed{args.seed}_{variant}.pt")
    if args.model.startswith("gnn") and ckpt.exists():
        from ml.gnn import load_checkpoint, score_fn

        model, graph = load_checkpoint(ckpt, split, nodes)
        score = score_fn(model, graph, drugs)
    else:
        score = models()[args.model](split, drugs, diseases)
    known = edges.filter(pl.col("rel_type") == "INDICATION").select(
        pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease")
    )
    preds = top_predictions(score, drugs, diseases, known, k=args.k)
    version = f"{args.model}-{args.split}-seed{args.seed}"
    with connect() as driver:
        n = write_predictions(driver, preds, version)
    print(f"wrote {n} PREDICTED_INDICATION relationships ({version})")


if __name__ == "__main__":
    main()
