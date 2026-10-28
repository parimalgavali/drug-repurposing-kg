import pytest

from explain.paths import Path, rank_metapaths, to_cytoscape


def node(key, label, name=None):
    return {"key": key, "id": key, "label": label, "name": name or key}


DRUG, DIS = node("d", "Drug"), node("s", "Disease")
G1, G2, P = node("g1", "Gene"), node("g2", "Gene"), node("p", "Phenotype")


def toy():
    paths = [
        Path([DRUG, G1, DIS], ["TARGET", "DISEASE_PROTEIN"]),
        Path([DRUG, G2, DIS], ["TARGET", "DISEASE_PROTEIN"]),
        Path([DRUG, P, DIS], ["DRUG_EFFECT", "DISEASE_PHENOTYPE_POSITIVE"]),
    ]
    degree = {
        ("d", "TARGET"): 2, ("g1", "TARGET"): 1, ("g2", "TARGET"): 4,
        ("g1", "DISEASE_PROTEIN"): 1, ("g2", "DISEASE_PROTEIN"): 1, ("s", "DISEASE_PROTEIN"): 2,
        ("d", "DRUG_EFFECT"): 4, ("p", "DRUG_EFFECT"): 4,
        ("p", "DISEASE_PHENOTYPE_POSITIVE"): 1, ("s", "DISEASE_PHENOTYPE_POSITIVE"): 1,
    }  # fmt: skip
    return paths, degree


def test_metapath_grouping_and_dwpc():
    paths, degree = toy()
    ranked = rank_metapaths(paths, degree, w=0.5)
    assert [m.metapath for m in ranked] == [
        "Drug-TARGET-Gene-DISEASE_PROTEIN-Disease",
        "Drug-DRUG_EFFECT-Phenotype-DISEASE_PHENOTYPE_POSITIVE-Disease",
    ]
    gene = ranked[0]
    # (2·1)^-.5·(1·2)^-.5 = 0.5 via g1; (2·4)^-.5·(1·2)^-.5 = 0.25 via g2
    assert gene.count == 2 and gene.dwpc == pytest.approx(0.75)
    assert [p.nodes[1]["key"] for p in gene.paths] == ["g1", "g2"]
    assert ranked[1].dwpc == pytest.approx(0.25)  # (4·4)^-.5 · (1·1)^-.5


def test_top_and_examples_limits_and_cytoscape():
    paths, degree = toy()
    ranked = rank_metapaths(paths, degree, top=1, examples=1, w=0.5)
    assert len(ranked) == 1 and len(ranked[0].paths) == 1 and ranked[0].count == 2
    cy = to_cytoscape(ranked)
    assert {n["data"]["id"] for n in cy["nodes"]} == {"d", "g1", "s"}
    assert {e["data"]["label"] for e in cy["edges"]} == {"TARGET", "DISEASE_PROTEIN"}
