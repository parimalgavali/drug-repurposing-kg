import polars as pl

from kg.labels_de import match, normalise, pick, read_doid_mondo

SSSOM = (
    "# curie_map:\nsubject_id\tsubject_label\tpredicate_id\tobject_id\n"
    "MONDO:0004979\tasthma\tskos:exactMatch\tDOID:2841\n"
    "MONDO:0005180\tparkinson\tskos:exactMatch\tDOID:14330\n"
    "MONDO:0000001\tx\tskos:broadMatch\tDOID:4\n"
)


def test_normalise_ids():
    assert normalise("MONDO", "MONDO:0004979") == "4979"
    assert normalise("MONDO", "MONDO_0010086") == "10086"
    assert normalise("DrugBank", "DB01050") == "DB01050"
    assert normalise("DrugBank", "BADID") is None
    assert normalise("DOID", "DOID:2841") == "DOID:2841"


def test_pick_prefers_direct_mondo_then_doid_and_shortest_label():
    rows = [
        ("DrugBank", "DB01050", "Ibuprofen"),
        ("DrugBank", "DB01050", "(RS)-Ibuprofen"),
        ("MONDO", "MONDO:0005180", "Parkinson-Krankheit"),
        ("DOID", "DOID:14330", "Morbus Parkinson"),
        ("DOID", "DOID:2841", "Asthma"),
        ("DOID", "DOID:4", "Krankheit"),
    ]
    doid_mondo = read_doid_mondo(SSSOM)
    assert doid_mondo == {"DOID:2841": ["4979"], "DOID:14330": ["5180"]}
    labels = pick(rows, doid_mondo)
    assert labels.rows() == [
        ("DrugBank", "DB01050", "Ibuprofen", "P715"),
        ("MONDO", "4979", "Asthma", "P699"),
        ("MONDO", "5180", "Parkinson-Krankheit", "P5270"),
    ]


def test_match_skips_grouped_diseases():
    nodes = pl.DataFrame(
        {
            "id": [14287, 33675, 27160],
            "source": ["DrugBank", "MONDO", "MONDO_grouped"],
            "source_id": ["DB01050", "4979", "4979_5180"],
        }
    )
    labels = pl.DataFrame(
        {"source": ["DrugBank", "MONDO"], "source_id": ["DB01050", "4979"], "name_de": ["I", "A"]}
    )
    assert sorted(match(nodes, labels).rows()) == [(14287, "DrugBank", "I"), (33675, "MONDO", "A")]
