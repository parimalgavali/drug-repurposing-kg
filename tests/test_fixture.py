import polars as pl
import pytest

from primekg.fixture import SEED_DISEASES, build_fixture, read_fixture, seed_disease_ids


@pytest.fixture(scope="module")
def mini():
    return read_fixture()


def test_every_seed_disease_has_an_indication(mini):
    nodes, edges = mini
    ind = edges.filter(pl.col("rel_type") == "INDICATION")
    for disease in seed_disease_ids(nodes):
        assert ind.filter(pl.col("dst_index") == disease).height >= 1


def test_edges_reference_known_nodes_and_size_is_small(mini):
    nodes, edges = mini
    ids = set(nodes["node_index"].to_list())
    assert set(edges["src_index"]) <= ids and set(edges["dst_index"]) <= ids
    assert 500 <= edges.height <= 2500
    assert edges.select("rel_type", "src_index", "dst_index").is_duplicated().sum() == 0


def test_fixture_is_deterministic(mini):
    nodes, edges = mini
    a = build_fixture(nodes, edges, max_edges=800)
    b = build_fixture(nodes, edges, max_edges=800)
    assert a[1].equals(b[1]) and a[0].equals(b[0])


def test_missing_seed_disease_raises(mini):
    nodes, _ = mini
    with pytest.raises(ValueError, match="not a disease"):
        seed_disease_ids(nodes, (*SEED_DISEASES, "not a disease"))
