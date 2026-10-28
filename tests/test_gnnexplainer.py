import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData  # noqa: E402

from explain.gnnexplainer import (  # noqa: E402
    PairScore,
    computation_edges,
    edge_key,
    explain_pair,
    initial_features,
    overlap,
    path_edges,
    without,
)
from ml.gnn import Graph, HeteroGNN  # noqa: E402

torch.set_num_threads(1)


def toy_graph() -> Graph:
    """drug 10, 11 · gene 20, 21, 22 · disease 30, 31 (global ids)."""
    data = HeteroData()
    index = {
        "drug": np.array([10, 11]),
        "gene_protein": np.array([20, 21, 22]),
        "disease": np.array([30, 31]),
    }
    for t, ids in index.items():
        data[t].num_nodes = len(ids)
    pairs = {
        ("drug", "TARGET", "gene_protein"): [(0, 0), (1, 1), (1, 2)],
        ("disease", "DISEASE_PROTEIN", "gene_protein"): [(0, 0), (1, 2)],
        ("gene_protein", "PROTEIN_PROTEIN", "gene_protein"): [(0, 1)],
        ("drug", "INDICATION", "disease"): [(1, 1)],
    }
    for (s, r, d), e in pairs.items():
        ei = torch.tensor(e).T
        data[s, r, d].edge_index = ei
        data[d, f"rev_{r}", s].edge_index = ei.flip(0)
    return Graph(data, index)


def toy_model(graph: Graph) -> HeteroGNN:
    torch.manual_seed(0)
    cfg = {"hidden": 8, "layers": 2, "dropout": 0.0, "disease_id_embedding": False}
    model = HeteroGNN(graph, cfg)
    with torch.no_grad():
        model.encode(graph.data.edge_index_dict)
    return model.eval()


def test_edge_key_is_undirected_and_drops_rev_prefix():
    assert edge_key("TARGET", 20, 10) == edge_key("rev_TARGET", 10, 20) == ("TARGET", 10, 20)


def test_path_edges_from_explain_result():
    result = {
        "metapaths": [
            {
                "paths": [
                    {
                        "nodes": [{"id": 10}, {"id": 20}, {"id": 30}],
                        "rels": ["TARGET", "DISEASE_PROTEIN"],
                    }
                ]
            }
        ]
    }
    assert path_edges(result) == {("TARGET", 10, 20), ("DISEASE_PROTEIN", 20, 30)}


def test_overlap_against_random_baseline():
    ranked = [("A", 1, 2), ("B", 2, 3), ("C", 3, 4), ("D", 4, 5)]
    r = overlap(ranked, {("B", 2, 3), ("D", 4, 5)}, candidates=8, k=2)
    assert r["hits"] == 1 and r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["random_precision"] == 0.25 and r["lift"] == 2.0


def test_receptive_field_keeps_the_pair_logit_exact():
    graph = toy_graph()
    model = toy_model(graph)
    scorer, x = PairScore(model), initial_features(model)
    d, s = torch.tensor([0]), torch.tensor([0])
    full = graph.data.edge_index_dict
    sub = computation_edges(full, {"drug": [0], "disease": [0]})
    with torch.no_grad():
        expected = model.decode(model.encode(full)["drug"][d], model.encode(full)["disease"][s])
        assert torch.allclose(scorer(x, full, d, s), expected)
        assert torch.allclose(scorer(x, sub, d, s), expected)
    assert sum(e.shape[1] for e in sub.values()) < sum(e.shape[1] for e in full.values())


def test_without_removes_both_directions():
    graph = toy_graph()
    edges = without(graph.data.edge_index_dict, graph, {("TARGET", 10, 20)})
    assert edges[("drug", "TARGET", "gene_protein")].shape[1] == 2
    assert edges[("gene_protein", "rev_TARGET", "drug")].shape[1] == 2


def test_explain_pair_runs_gnnexplainer_on_the_toy_graph():
    graph = toy_graph()
    model = toy_model(graph)
    for p in model.parameters():
        p.requires_grad_(False)
    paths = {("TARGET", 10, 20), ("DISEASE_PROTEIN", 20, 30)}
    out = explain_pair(PairScore(model), initial_features(model), graph, 10, 30, paths, epochs=5)
    assert out["path_edges_in_field"] == 2
    assert out["candidate_edges"] >= 2
    assert {e["rel"] for e in out["top_edges"]} <= {
        "TARGET",
        "DISEASE_PROTEIN",
        "PROTEIN_PROTEIN",
        "INDICATION",
    }
    assert set(out["logit_after_removing"]) == {"paths", "gnnexplainer", "random"}
