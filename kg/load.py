"""Load node/edge frames with batched UNWIND. Meant for the test fixture, not the full graph
(use kg.build_import for that): `uv run python -m kg.load` loads data/fixtures/primekg_mini."""

from __future__ import annotations

import re

import polars as pl
from neo4j import Driver

from kg.client import connect
from primekg.fixture import read_fixture
from primekg.prepare import LABELS

_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe label/type: {name!r}")
    return name


def load_frames(
    driver: Driver,
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    database: str = "neo4j",
    batch: int = 2000,
) -> tuple[int, int]:
    with driver.session(database=database) as session:
        for (label,), group in nodes.group_by("label", maintain_order=True):
            rows = group.select(
                pl.col("node_index").alias("id"),
                pl.col("node_id").alias("source_id"),
                pl.col("node_source").alias("source"),
                pl.col("node_name").alias("name"),
            ).to_dicts()
            query = f"UNWIND $rows AS r CREATE (n:{_ident(label)}) SET n = r"
            for i in range(0, len(rows), batch):
                session.run(query, rows=rows[i : i + batch]).consume()
        keys = ["rel_type", "src_type", "dst_type"]
        for (rel, src_type, dst_type), group in edges.group_by(keys, maintain_order=True):
            rows = group.select(pl.col("src_index").alias("s"), pl.col("dst_index").alias("d"))
            rows = rows.to_dicts()
            src, dst = _ident(LABELS[src_type]), _ident(LABELS[dst_type])
            query = (
                f"UNWIND $rows AS r MATCH (a:{src} {{id: r.s}}), (b:{dst} {{id: r.d}}) "
                f"CREATE (a)-[:{_ident(rel)}]->(b)"
            )
            for i in range(0, len(rows), batch):
                session.run(query, rows=rows[i : i + batch]).consume()
    return nodes.height, edges.height


def node_count(driver: Driver, database: str = "neo4j") -> int:
    with driver.session(database=database) as session:
        return session.run("MATCH (n) RETURN count(n) AS c").single()["c"]


if __name__ == "__main__":
    nodes, edges = read_fixture()
    with connect() as driver:
        if node_count(driver):
            raise SystemExit("database is not empty; refusing to load the fixture on top of it")
        n, e = load_frames(driver, nodes, edges)
        print(f"loaded fixture: {n:,} nodes, {e:,} relationships")
