"""Convert data/raw/primekg/ into canonical parquet tables in data/processed/.

Usage: uv run python scripts/prepare.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from primekg.prepare import prepare_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/primekg"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--no-check", action="store_true", help="skip expected-count assertions")
    args = parser.parse_args()

    counts = prepare_all(args.raw, args.out, check_counts=not args.no_check)
    for name, n in counts.items():
        print(f"{name:>18}: {n:,}")
    edges = pl.read_parquet(args.out / "edges.parquet")
    print(edges.group_by("rel_type").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
