import time

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from ml.gnn import IND, build_graph, load_checkpoint, load_config, score_fn, train  # noqa: E402
from ml.splits import disease_groups, make_split  # noqa: E402
from primekg.fixture import read_fixture  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(1)  # the fixture graph is tiny; extra threads only add contention
    nodes, edges = read_fixture()
    split = make_split("disease", edges, disease_groups(nodes), seed=0)
    drugs = np.sort(nodes.filter(pl.col("node_type") == "drug")["node_index"].to_numpy())
    return nodes, split, drugs


def test_no_test_edges_in_message_passing(setup):
    nodes, split, _ = setup
    graph = build_graph(nodes, split)
    held = pl.concat([split.test, split.valid])
    drug = graph.local("drug", held["drug"].to_numpy()).tolist()
    disease = graph.local("disease", held["disease"].to_numpy()).tolist()
    held_pairs = set(zip(drug, disease, strict=True))
    assert held_pairs  # the fixture has held-out pairs
    for et, ei in graph.data.edge_index_dict.items():
        if et[0] == "drug" and et[2] == "disease":
            assert not held_pairs & set(map(tuple, ei.t().tolist())), et
        if et[0] == "disease" and et[2] == "drug":
            assert not held_pairs & set(map(tuple, ei.flip(0).t().tolist())), et
    assert graph.data[IND].edge_index.shape[1] == split.train.height


def test_forward_backward_fast(setup):
    nodes, split, drugs = setup
    cfg = load_config(epochs=4, eval_every=2, text=False)
    start = time.perf_counter()
    model, graph, info = train(split, nodes, drugs, cfg, text_path=None, log=lambda _: None)
    scores = score_fn(model, graph, drugs)(split.test["disease"].unique().sort().to_numpy())
    assert time.perf_counter() - start < 10
    assert np.isfinite(scores).all() and scores.shape[1] == len(drugs)
    assert info["epochs_run"] == 4 and len(info["history"]) == 2


def test_checkpoint_round_trip(setup, tmp_path):
    nodes, split, drugs = setup
    cfg = load_config(epochs=2, eval_every=2, text=False)
    model, graph, info = train(split, nodes, drugs, cfg, text_path=None, log=lambda _: None)
    path = tmp_path / "gnn.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg, "info": info}, path)
    loaded, graph2 = load_checkpoint(path, split, nodes, text_path=None)
    diseases = split.test["disease"].unique().sort().to_numpy()
    model.eval()
    np.testing.assert_allclose(
        score_fn(model, graph, drugs)(diseases),
        score_fn(loaded, graph2, drugs)(diseases),
        rtol=1e-5,
    )


def test_disease_pooling_trains_and_gates_unseen_diseases(setup):
    from ml.gnn import disease_neighbours

    nodes, split, drugs = setup
    graph = build_graph(nodes, split)
    idx, w = disease_neighbours(graph, split, k=3)
    pool = set(graph.local("disease", split.train["disease"].unique().to_numpy()).tolist())
    assert idx.shape == w.shape == (len(graph.index["disease"]), min(3, len(pool)))
    assert set(idx.ravel().tolist()) <= pool
    np.testing.assert_allclose(w.sum(1)[w.sum(1) > 0], 1, rtol=1e-5)
    cfg = load_config(epochs=4, eval_every=2, text=False, supervision="disease", pool_k=3)
    model, graph, info = train(split, nodes, drugs, cfg, text_path=None, log=lambda _: None)
    scores = score_fn(model, graph, drugs)(split.test["disease"].unique().sort().to_numpy())
    assert np.isfinite(scores).all() and info["epochs_run"] == 4
    assert model.gate.grad is not None
