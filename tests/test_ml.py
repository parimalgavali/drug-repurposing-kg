import numpy as np
import polars as pl
import pytest

pytest.importorskip("sklearn")

from ml.baselines import popularity, shared_neighbour, similar_disease  # noqa: E402
from ml.evaluate import evaluate  # noqa: E402
from ml.splits import CLINICAL, PRUNED, disease_groups, make_split  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402


@pytest.fixture(scope="module")
def fixture():
    nodes, edges = read_fixture()
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    diseases = np.sort(nodes.filter(pl.col("node_type") == "disease")["node_index"].to_numpy())
    return nodes, edges, drugs, diseases


@pytest.mark.parametrize("name", ["disease", "random"])
def test_no_leakage(fixture, name):
    nodes, edges, *_ = fixture
    split = make_split(name, edges, disease_groups(nodes), seed=0)
    held = pl.concat([split.test, split.valid])
    clinical = split.graph.filter(pl.col("rel_type").is_in(CLINICAL))
    both_ways = pl.concat(
        [
            clinical.select(
                pl.col("src_index").alias("drug"), pl.col("dst_index").alias("disease")
            ),
            clinical.select(
                pl.col("dst_index").alias("drug"), pl.col("src_index").alias("disease")
            ),
        ]
    )
    assert held.join(both_ways, on=["drug", "disease"]).is_empty()
    assert held.join(split.train, on=["drug", "disease"]).is_empty()
    assert not split.graph["rel_type"].is_in(PRUNED).any()
    if name == "disease":
        held_dis = held["disease"].unique().implode()
        assert clinical.filter(pl.col("dst_index").is_in(held_dis)).is_empty()
        assert split.train.filter(pl.col("disease").is_in(held_dis)).is_empty()


def test_split_is_seeded(fixture):
    nodes, edges, *_ = fixture
    a = make_split("disease", edges, disease_groups(nodes), seed=1)
    b = make_split("disease", edges, disease_groups(nodes), seed=1)
    assert a.test.equals(b.test) and a.graph.equals(b.graph)


def test_evaluator_known_answer():
    drugs = np.array([10, 11, 12, 13])
    test = pl.DataFrame({"drug": [11, 12], "disease": [1, 1]})
    known = pl.DataFrame({"drug": [13], "disease": [1]})
    # scores: drug 13 (known, filtered) is highest; 11 ranks 1st, 12 ranks 2nd behind 10
    scores = np.array([[0.5, 0.9, 0.4, 1.0]])
    m = evaluate(lambda chunk: scores, drugs, test, known, ks=(1, 10))
    assert m["mrr"] == pytest.approx((1 + 1 / 2) / 2)
    assert m["hits@1"] == 0.5 and m["hits@10"] == 1.0
    constant = evaluate(lambda chunk: np.zeros((1, 4)), drugs, test, known, ks=(1,))
    assert constant["mrr"] == pytest.approx(1 / 1.5)  # tie with the one negative → rank 1.5


@pytest.mark.parametrize("model", [popularity, shared_neighbour, similar_disease])
def test_baselines_are_deterministic(fixture, model):
    nodes, edges, drugs, diseases = fixture
    split = make_split("random", edges, disease_groups(nodes), seed=0)
    chunk = split.test["disease"].unique().sort().to_numpy()
    a = model(split, drugs, diseases)(chunk)
    b = model(split, drugs, diseases)(chunk)
    assert a.shape == (len(chunk), len(drugs))
    np.testing.assert_array_equal(a, b)


def test_strict_split_has_no_ontology_neighbour_across_train_and_test(fixture):
    from ml.splits import ontology_groups, ontology_leakage

    nodes, edges, *_ = fixture
    groups = ontology_groups(edges, disease_groups(nodes))
    split = make_split("disease_strict", edges, disease_groups(nodes), seed=0)
    assert split.name == "disease_strict" and split.test.height > 0
    assert ontology_leakage(edges, split)["test_pairs_leaked"] == 0
    group = dict(groups.iter_rows())
    held = set(pl.concat([split.test, split.valid])["disease"])
    train = set(split.train["disease"])
    assert {group[d] for d in held}.isdisjoint({group[d] for d in train})
    dd = edges.filter(pl.col("rel_type") == "DISEASE_DISEASE")
    for a, b in dd.select("src_index", "dst_index").iter_rows():
        assert not ((a in held and b in train) or (b in held and a in train))


def test_ontology_leakage_counts_neighbour_drugs():
    from ml.splits import Split, ontology_leakage

    edges = pl.DataFrame(
        {
            "rel_type": ["DISEASE_DISEASE"],
            "src_index": [1],
            "src_type": ["disease"],
            "dst_index": [2],
            "dst_type": ["disease"],
        }
    )
    train = pl.DataFrame({"drug": [10], "disease": [2]})
    test = pl.DataFrame({"drug": [10, 11], "disease": [1, 1]})
    split = Split("disease", 0, edges, train, test.head(0), test)
    leak = ontology_leakage(edges, split)
    assert leak == {"test_pairs_leaked": 0.5, "test_diseases_with_neighbour": 1.0}
