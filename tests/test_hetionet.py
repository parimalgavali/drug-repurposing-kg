import gzip

import polars as pl
import pytest

from validation.hetionet import evaluate, map_pairs, read_ctd, read_doid_mondo, recorded

NODES = pl.DataFrame(
    {
        "node_index": [1, 2, 10, 11, 12],
        "node_id": ["DB00001", "DB00002", "4979", "8099_12497", "5000"],
        "node_type": ["drug", "drug", "disease", "disease", "disease"],
        "node_source": ["DrugBank", "DrugBank", "MONDO", "MONDO_grouped", "MONDO"],
    }
)


@pytest.fixture
def raw(tmp_path):
    with gzip.open(tmp_path / "edges.sif.gz", "wt") as f:
        f.write("source\tmetaedge\ttarget\n")
        f.write("Compound::DB00001\tCtD\tDisease::DOID:2841\n")
        f.write("Compound::DB00002\tCtD\tDisease::DOID:9\n")
        f.write("Compound::DB99999\tCtD\tDisease::DOID:2841\n")
        f.write("Compound::DB00001\tCpD\tDisease::DOID:9\n")
    (tmp_path / "map.tsv").write_text(
        "# curie_map:\nsubject_id\tsubject_label\tpredicate_id\tobject_id\n"
        "MONDO:0004979\tasthma\tskos:exactMatch\tDOID:2841\n"
        "MONDO:0012497\tx\tskos:exactMatch\tDOID:9\n"
    )
    return tmp_path


def test_maps_drugbank_and_doid_via_mondo_including_grouped(raw):
    ctd = read_ctd(raw / "edges.sif.gz")
    assert ctd == [("DB00001", "DOID:2841"), ("DB00002", "DOID:9"), ("DB99999", "DOID:2841")]
    pairs = map_pairs(ctd, read_doid_mondo(raw / "map.tsv"), NODES)
    assert pairs.rows() == [(1, 10), (2, 11)]


def test_recorded_reads_both_edge_directions():
    edges = pl.DataFrame(
        {
            "rel_type": ["INDICATION", "INDICATION", "TARGET"],
            "src_index": [1, 11, 1],
            "src_type": ["drug", "disease", "drug"],
            "dst_index": [10, 2, 99],
            "dst_type": ["disease", "drug", "gene/protein"],
        }
    )
    assert sorted(recorded(edges, ["INDICATION"]).rows()) == [(1, 10), (2, 11)]


def test_evaluate_precision_and_recall():
    truth = pl.DataFrame({"drug": [1, 2], "disease": [10, 10]})
    top = pl.DataFrame({"drug": [1, 3, 2], "disease": [10, 10, 10], "rank": [1, 2, 3]})
    r = evaluate({"m": top}, truth, [2, 3])["m"]
    assert (r["top2"]["hits"], r["top2"]["precision"], r["top2"]["recall"]) == (1, 0.5, 0.5)
    assert (r["top3"]["hits"], r["top3"]["recall"]) == (2, 1.0)
