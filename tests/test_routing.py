import numpy as np
import polars as pl

from ml.routing import has_genes, pick, subgroup
from ml.write_predictions import full_split, unseen_groups


def _edges(rows):
    return pl.DataFrame(
        rows, schema=["rel_type", "src_index", "src_type", "dst_index", "dst_type"], orient="row"
    )


EDGES = _edges(
    [
        ("INDICATION", 10, "drug", 1, "disease"),
        ("DISEASE_PROTEIN", 1, "disease", 100, "gene/protein"),
        ("DISEASE_PROTEIN", 2, "disease", 100, "gene/protein"),
        ("DISEASE_PHENOTYPE_POSITIVE", 3, "disease", 200, "effect/phenotype"),
        ("DISEASE_DISEASE", 4, "disease", 1, "disease"),
    ]
)


def test_unseen_groups_split_by_gene_links():
    groups = unseen_groups(EDGES, np.array([1, 2, 3, 4]))
    assert groups["with_genes"].tolist() == [2, 3]
    assert groups["without_genes"].tolist() == [4]
    assert has_genes(EDGES) == {1, 2, 3}


def test_full_split_trains_on_every_indication():
    split = full_split(EDGES)
    assert split.train.rows() == [(10, 1)] and split.test.height == split.valid.height == 0


def test_subgroup_and_pick_use_validation():
    pairs = pl.DataFrame({"drug": [10, 11], "disease": [1, 4]})
    assert subgroup(pairs, {1}, "with_genes")["disease"].to_list() == [1]
    assert subgroup(pairs, {1}, "without_genes")["disease"].to_list() == [4]

    def run(valid_with, valid_without, test=0.9):
        return {
            "with_genes": {"valid": {"pairs": 5, "mrr": valid_with}, "test": {"mrr": test}},
            "without_genes": {"valid": {"pairs": 5, "mrr": valid_without}, "test": {"mrr": test}},
        }

    results = {"disease_strict": {"a": [run(0.3, 0.1)], "b": [run(0.2, 0.2, test=0.0)]}}
    assert pick(results) == {"with_genes": "a", "without_genes": "b"}
