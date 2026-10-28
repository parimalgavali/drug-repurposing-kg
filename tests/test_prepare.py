import polars as pl
import pytest

from primekg.prepare import canonical_edges, normalise_rel, prepare_features, prepare_nodes


def kg_rows(rows):
    cols = ["relation", "display_relation", "x_index", "x_type", "y_index", "y_type"]
    return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows]).with_columns(
        pl.col("x_index").cast(pl.String), pl.col("y_index").cast(pl.String)
    )


@pytest.mark.parametrize(
    ("relation", "display", "expected"),
    [
        ("indication", "indication", "INDICATION"),
        ("off-label use", "off-label use", "OFF_LABEL_USE"),
        ("drug_protein", "target", "TARGET"),
        ("drug_protein", "enzyme", "ENZYME"),
        ("disease_phenotype_positive", "phenotype present", "DISEASE_PHENOTYPE_POSITIVE"),
    ],
)
def test_normalise_rel(relation, display, expected):
    assert normalise_rel(relation, display) == expected


def test_both_directions_collapse_to_one_with_type_order():
    kg = kg_rows(
        [
            ("indication", "indication", 10, "disease", 1, "drug"),
            ("indication", "indication", 1, "drug", 10, "disease"),
        ]
    )
    out = canonical_edges(kg.lazy()).collect()
    assert out.rows() == [("INDICATION", 1, "drug", 10, "disease")]


def test_same_type_relation_uses_smaller_index_as_source():
    kg = kg_rows(
        [
            ("protein_protein", "ppi", 9, "gene/protein", 3, "gene/protein"),
            ("protein_protein", "ppi", 3, "gene/protein", 9, "gene/protein"),
        ]
    )
    out = canonical_edges(kg.lazy()).collect()
    assert out.select("src_index", "dst_index").rows() == [(3, 9)]


def test_drug_protein_subtypes_are_kept_apart():
    kg = kg_rows(
        [
            ("drug_protein", "target", 1, "drug", 5, "gene/protein"),
            ("drug_protein", "target", 5, "gene/protein", 1, "drug"),
            ("drug_protein", "enzyme", 1, "drug", 5, "gene/protein"),
            ("drug_protein", "enzyme", 5, "gene/protein", 1, "drug"),
        ]
    )
    out = canonical_edges(kg.lazy()).collect()
    assert sorted(out["rel_type"].to_list()) == ["ENZYME", "TARGET"]


def test_prepare_nodes_adds_labels():
    raw = pl.DataFrame(
        {
            "node_index": ["2", "1"],
            "node_id": ["a", "b"],
            "node_type": ["gene/protein", "drug"],
            "node_name": ["TP53", "Aspirin"],
            "node_source": ["NCBI", "DrugBank"],
        }
    )
    out = prepare_nodes(raw)
    assert out["node_index"].to_list() == [1, 2]
    assert out["label"].to_list() == ["Drug", "Gene"]


def test_prepare_features_keeps_first_duplicate():
    raw = pl.DataFrame({"node_index": ["5", "5", "4"], "description": ["first", "second", "x"]})
    out = prepare_features(raw)
    assert out.rows() == [(4, "x"), (5, "first")]
