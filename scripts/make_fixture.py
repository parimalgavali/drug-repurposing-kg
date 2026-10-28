"""Write the committed test fixture data/fixtures/primekg_mini/{nodes,edges}.csv.

Usage: uv run python scripts/make_fixture.py   (needs data/processed/ from scripts/prepare.py)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from primekg.fixture import build_fixture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/fixtures/primekg_mini"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    edges = pl.read_parquet(args.processed / "edges.parquet")
    sub_nodes, sub_edges = build_fixture(nodes, edges, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    sub_nodes.write_csv(args.out / "nodes.csv")
    sub_edges.write_csv(args.out / "edges.csv")
    print(f"nodes: {sub_nodes.height:,}  edges: {sub_edges.height:,}")
    print(sub_edges.group_by("rel_type").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
