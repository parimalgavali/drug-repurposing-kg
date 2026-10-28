"""Download PrimeKG into data/raw/primekg/ and verify checksums.

Usage: uv run python scripts/fetch_data.py {primekg,features,all}
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primekg.fetch import GROUPS, fetch  # noqa: E402

RAW_DIR = Path("data/raw/primekg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=sorted(GROUPS))
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    for path in fetch(args.group, args.dest):
        print(f"ok  {path}  {path.stat().st_size:,} B")


if __name__ == "__main__":
    main()
