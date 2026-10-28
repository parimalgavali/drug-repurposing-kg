"""Write each disease's top-k non-indicated drugs to Neo4j as
(:Drug)-[:PREDICTED_INDICATION {score, rank, model_version}]->(:Disease).

Existing PREDICTED_INDICATION relationships are deleted first, so reruns are idempotent.

Diseases with no known indication are the unseen-disease case, so by default they are ranked by
the model `ml.routing` picked on held-out-disease validation (separately for diseases with and
without gene/phenotype links); `model_version` records which model produced each relationship.

    uv run --extra ml --extra graph python -m ml.write_predictions --model gnn_notext --split random
"""

from __future__ import annotations

import argparse
import json
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
                                   model_version: coalesce(row.version, $version)}]->(s)
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
    cols = [
        "drug",
        "disease",
        "score",
        "rank",
        *(["version"] if "version" in preds.columns else []),
    ]
    rows = preds.select(cols).to_dicts()
    with driver.session(database=database) as session:
        session.run(DELETE).consume()
        for i in range(0, len(rows), batch):
            session.run(CREATE, rows=rows[i : i + batch], version=version).consume()
        query = "MATCH ()-[r:PREDICTED_INDICATION]->() RETURN count(r) AS n"
        return session.run(query).single()["n"]


GNN_TAGS = {"gnn": "text", "gnn_notext": "notext", "gnn_dmask": "dmask", "gnn_pool": "pool"}


def scorer(model: str, split, nodes: pl.DataFrame, drugs: np.ndarray, diseases: np.ndarray):
    from ml.run import models

    ckpt = Path(f"artifacts/gnn/{split.name}_seed{split.seed}_{GNN_TAGS.get(model, '')}.pt")
    if model in GNN_TAGS and ckpt.exists():
        from ml.gnn import load_checkpoint, score_fn

        net, graph = load_checkpoint(ckpt, split, nodes)
        return score_fn(net, graph, drugs)
    return models()[model](split, drugs, diseases)


def unseen_groups(edges: pl.DataFrame, diseases: np.ndarray) -> dict[str, np.ndarray]:
    """Diseases without any known INDICATION, split by whether they have gene/phenotype links."""
    from ml.routing import has_genes

    indicated = set(edges.filter(pl.col("rel_type") == "INDICATION")["dst_index"].to_list())
    genes = has_genes(edges)
    unseen = [int(d) for d in diseases if int(d) not in indicated]
    return {
        "with_genes": np.array([d for d in unseen if d in genes], dtype=diseases.dtype),
        "without_genes": np.array([d for d in unseen if d not in genes], dtype=diseases.dtype),
    }


def full_split(edges: pl.DataFrame):
    """Every known indication as training data, for baselines that rank unseen diseases."""
    from ml.splits import PAIR, Split, indications

    ind = indications(edges)
    return Split("all", 0, edges, ind, ind.head(0).select(PAIR), ind.head(0).select(PAIR))


def main() -> None:
    from kg.client import connect
    from ml.run import get_split

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gnn")
    parser.add_argument("--split", default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--unseen-model",
        default="auto",
        help="model for diseases with no known drugs: auto (ml/results/routing.json), "
        "a model name, or none to use --model everywhere",
    )
    args = parser.parse_args()

    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    edges = pl.read_parquet(args.processed / "edges.parquet")
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    split = get_split(args.split, args.seed, args.processed, args.processed / "splits")
    known = edges.filter(pl.col("rel_type") == "INDICATION").select(
        pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease")
    )
    version = f"{args.model}-{args.split}-seed{args.seed}"
    route = {}
    if args.unseen_model == "auto":
        from ml.routing import OUT

        route = json.loads(OUT.read_text())["picked"]
    elif args.unseen_model != "none":
        route = {"with_genes": args.unseen_model, "without_genes": args.unseen_model}
    groups = unseen_groups(edges, diseases) if route else {}
    routed = {int(d) for g in groups.values() for d in g}
    rest = np.array([d for d in diseases if int(d) not in routed], dtype=diseases.dtype)
    score = scorer(args.model, split, nodes, drugs, diseases)
    frames = [
        top_predictions(score, drugs, rest, known, k=args.k).with_columns(version=pl.lit(version))
    ]
    full = full_split(edges) if route else None
    for group, model in route.items():
        if model in GNN_TAGS:  # GNNs: the served split's checkpoint of that variant
            model_split, tag = split, f"{model}-{args.split}-seed{args.seed}"
        else:
            model_split, tag = full, f"{model}-all"
        s = scorer(model, model_split, nodes, drugs, diseases)
        frames.append(
            top_predictions(s, drugs, groups[group], known, k=args.k).with_columns(
                version=pl.lit(tag)
            )
        )
        print(f"{group}: {len(groups[group])} diseases without known drugs -> {tag}")
    preds = pl.concat(frames)
    with connect() as driver:
        n = write_predictions(driver, preds, version)
    print(f"wrote {n} PREDICTED_INDICATION relationships ({version} + {route or 'no routing'})")


if __name__ == "__main__":
    main()
