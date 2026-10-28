"""Write the processed PrimeKG tables as neo4j-admin import CSVs and optionally run the import.

    uv run python -m kg.build_import                 # CSVs only → data/import/
    uv run python -m kg.build_import --run --admin "sudo -u neo4j neo4j-admin"
    uv run python -m kg.build_import --run --admin "docker compose run --rm neo4j neo4j-admin" \
        --prefix /import
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl

ID_SPACE = "Node"
NODE_HEADER = {
    "node_index": f"id:ID({ID_SPACE})",
    "node_id": "source_id",
    "node_source": "source",
    "node_name": "name",
}
REL_HEADER = {"src_index": f":START_ID({ID_SPACE})", "dst_index": f":END_ID({ID_SPACE})"}


@dataclass(frozen=True)
class ImportFile:
    kind: str  # "nodes" | "relationships"
    name: str  # label or relationship type
    filename: str
    rows: int


def write_import_files(nodes: pl.DataFrame, edges: pl.DataFrame, out_dir: Path) -> list[ImportFile]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()
    files: list[ImportFile] = []
    for label in sorted(nodes["label"].unique()):
        df = (
            nodes.filter(pl.col("label") == label)
            .select(NODE_HEADER.keys())
            .sort("node_index")
            .rename(NODE_HEADER)
        )
        filename = f"nodes_{label}.csv"
        df.write_csv(out_dir / filename)
        files.append(ImportFile("nodes", label, filename, df.height))
    for rel in sorted(edges["rel_type"].unique()):
        df = (
            edges.filter(pl.col("rel_type") == rel)
            .select(REL_HEADER.keys())
            .sort(["src_index", "dst_index"])
            .rename(REL_HEADER)
        )
        filename = f"rels_{rel}.csv"
        df.write_csv(out_dir / filename)
        files.append(ImportFile("relationships", rel, filename, df.height))
    return files


def import_args(
    files: list[ImportFile],
    prefix: str,
    database: str = "neo4j",
    report_file: str = "/tmp/repurposekg-import.report",
) -> list[str]:
    args = [
        "database",
        "import",
        "full",
        database,
        "--overwrite-destination=true",
        "--id-type=integer",
        "--skip-duplicate-nodes=false",
        "--bad-tolerance=0",
        f"--report-file={report_file}",
    ]
    args += [f"--{f.kind}={f.name}={prefix.rstrip('/')}/{f.filename}" for f in files]
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/import"))
    parser.add_argument("--run", action="store_true", help="run neo4j-admin after writing CSVs")
    parser.add_argument("--admin", default="neo4j-admin", help="neo4j-admin command prefix")
    parser.add_argument("--prefix", help="path of --out as seen by neo4j-admin (default: --out)")
    parser.add_argument("--database", default="neo4j")
    args = parser.parse_args()

    nodes = pl.read_parquet(args.processed / "nodes.parquet")
    edges = pl.read_parquet(args.processed / "edges.parquet")
    files = write_import_files(nodes, edges, args.out)
    n_nodes = sum(f.rows for f in files if f.kind == "nodes")
    n_rels = sum(f.rows for f in files if f.kind == "relationships")
    print(f"wrote {len(files)} files to {args.out}: {n_nodes:,} nodes, {n_rels:,} relationships")

    cmd = [
        *shlex.split(args.admin),
        *import_args(files, args.prefix or str(args.out.resolve()), args.database),
    ]
    if args.run:
        subprocess.run(cmd, check=True)
    else:
        print("import command:\n  " + shlex.join(cmd))


if __name__ == "__main__":
    main()
