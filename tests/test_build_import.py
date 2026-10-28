import polars as pl

from kg.build_import import import_args, write_import_files
from primekg.fixture import read_fixture


def test_writes_one_file_per_label_and_type_with_admin_headers(tmp_path):
    nodes, edges = read_fixture()
    files = write_import_files(nodes, edges, tmp_path)

    node_files = [f for f in files if f.kind == "nodes"]
    rel_files = [f for f in files if f.kind == "relationships"]
    assert {f.name for f in node_files} == set(nodes["label"].unique())
    assert {f.name for f in rel_files} == set(edges["rel_type"].unique())
    assert sum(f.rows for f in node_files) == nodes.height
    assert sum(f.rows for f in rel_files) == edges.height

    drug = (tmp_path / "nodes_Drug.csv").read_text().splitlines()[0]
    assert drug == "id:ID(Node),source_id,source,name"
    ind = (tmp_path / "rels_INDICATION.csv").read_text().splitlines()[0]
    assert ind == ":START_ID(Node),:END_ID(Node)"


def test_relationship_files_keep_one_direction_per_edge(tmp_path):
    nodes, edges = read_fixture()
    for f in write_import_files(nodes, edges, tmp_path):
        if f.kind != "relationships":
            continue
        df = pl.read_csv(tmp_path / f.filename).rename(
            {":START_ID(Node)": "s", ":END_ID(Node)": "d"}
        )
        reverse = df.select(pl.col("d").alias("s"), pl.col("s").alias("d"))
        assert df.join(reverse, on=["s", "d"]).filter(pl.col("s") != pl.col("d")).is_empty()
        assert not df.is_duplicated().any()


def test_rewrites_stale_files(tmp_path):
    (tmp_path / "rels_OLD.csv").write_text("x\n")
    nodes, edges = read_fixture()
    write_import_files(nodes, edges, tmp_path)
    assert not (tmp_path / "rels_OLD.csv").exists()


def test_import_args_reference_prefix(tmp_path):
    nodes, edges = read_fixture()
    files = write_import_files(nodes, edges, tmp_path)
    args = import_args(files, "/import/")
    assert args[:4] == ["database", "import", "full", "neo4j"]
    assert "--id-type=integer" in args
    assert "--nodes=Drug=/import/nodes_Drug.csv" in args
    assert "--relationships=INDICATION=/import/rels_INDICATION.csv" in args
