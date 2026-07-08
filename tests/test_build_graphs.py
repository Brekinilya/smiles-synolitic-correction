"""Contract tests for stage 2 (build_graphs.py).

Uses the team's official dummy fixtures from synolitic.common.dummy so the
tests are symmetric with what the rest of the pipeline uses.
Run with: uv run pytest tests/test_build_graphs.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from synolitic.common import dummy, schemas
from synolitic.common.io import load_artifact, save_artifact
from synolitic.stage2_graphs.build_graphs import build_graphs, fit_pairwise_ensemble


def _fit(hs: dict):
    split = np.asarray(hs["split"])
    x_train = np.asarray(hs["X"], dtype=np.float64)[split == schemas.SPLIT_TRAIN]
    y_train = np.asarray(hs["is_correct"])[split == schemas.SPLIT_TRAIN]
    return fit_pairwise_ensemble(x_train, y_train)


# ------------------------------------------------------------------
# Main contract test
# ------------------------------------------------------------------

def test_graphs_pass_validate_graphs():
    """Output of build_graphs must pass the real schemas.validate_graphs."""
    hs = dummy.dummy_hidden_states(n=300, seed=1)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)   # raises via assert_valid if invalid
    assert schemas.validate_graphs(artifact) == []


def test_n_graphs_equals_n_samples():
    hs = dummy.dummy_hidden_states(n=200, seed=2)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    assert len(artifact["graphs"]) == 200


def test_node_feature_shape():
    hs = dummy.dummy_hidden_states(n=100, seed=3)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    for g in artifact["graphs"][:10]:
        assert g.x.shape == (schemas.D_H, schemas.N_NODE_FEATURES)
        assert g.x.dtype == torch.float32
        assert torch.isfinite(g.x).all()


def test_edges_are_undirected():
    """For every (u, v) in edge_index there must be a matching (v, u)."""
    hs = dummy.dummy_hidden_states(n=50, seed=4)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    for g in artifact["graphs"][:10]:
        pairs = set(map(tuple, g.edge_index.t().tolist()))
        for (u, v) in list(pairs)[:30]:
            assert (v, u) in pairs


def test_split_codes_preserved():
    """Each graph's split code must match the original hidden_states split."""
    hs = dummy.dummy_hidden_states(n=200, seed=5)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    orig_split = np.asarray(hs["split"])
    for g in artifact["graphs"]:
        idx = int(g.idx)
        assert int(g.split) == int(orig_split[idx])


def test_y_matches_is_correct():
    hs = dummy.dummy_hidden_states(n=200, seed=6)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    orig_ic = np.asarray(hs["is_correct"])
    for g in artifact["graphs"]:
        assert g.y.item() == float(orig_ic[int(g.idx)])


# ------------------------------------------------------------------
# Sparsification
# ------------------------------------------------------------------

def test_sparsify_top_k_controls_edge_count():
    hs = dummy.dummy_hidden_states(n=100, seed=7)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble, sparsify_top_k=50)
    assert artifact["meta"]["sparsify_top_k"] == 50
    for g in artifact["graphs"][:10]:
        assert g.edge_index.shape[1] == 50 * 2  # both directions


# ------------------------------------------------------------------
# Leakage protocol
# ------------------------------------------------------------------

def test_classifiers_fitted_on_train_split_only():
    """Verify that changing cal/test labels does not affect the ensemble."""
    hs = dummy.dummy_hidden_states(n=300, seed=8)
    ens_original = _fit(hs)

    hs_corrupted = {k: v.clone() if torch.is_tensor(v) else v for k, v in hs.items()}
    split = np.asarray(hs["split"])
    cal_test_mask = torch.tensor(split != schemas.SPLIT_TRAIN)
    hs_corrupted["is_correct"] = hs["is_correct"].clone()
    hs_corrupted["is_correct"][cal_test_mask] ^= 1  # flip cal/test labels

    ens_corrupted = _fit(hs_corrupted)  # should produce identical ensemble
    assert np.allclose(ens_original.W, ens_corrupted.W, atol=1e-5)
    assert np.allclose(ens_original.b, ens_corrupted.b, atol=1e-5)


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_fit_raises_if_train_split_has_one_class():
    x_train = np.random.randn(100, 8)
    y_train = np.ones(100, dtype=np.int8)
    with pytest.raises(ValueError):
        fit_pairwise_ensemble(x_train, y_train)


# ------------------------------------------------------------------
# Fix regression tests
# ------------------------------------------------------------------

def test_decisiveness_symmetry():
    """Fix 1: score near 0 and near 1 should both produce high node strength.
    Before the fix, score=0.02 (error-leaning) produced near-zero strength
    because raw score was used as topology weight instead of decisiveness."""
    from synolitic.stage2_graphs.build_graphs import SynoliticEnsemble, _build_sample_graph
    import itertools

    d = 4
    pairs = list(itertools.combinations(range(d), 2))  # 6 pairs
    W = np.zeros((len(pairs), 2), dtype=np.float32)
    b = np.zeros(len(pairs), dtype=np.float32)
    x_vec = np.zeros(d, dtype=np.float32)

    for score_val, label in [(0.98, "correct-leaning"), (0.02, "error-leaning")]:
        scores_row = np.full(len(pairs), 0.5, dtype=np.float32)
        scores_row[0] = score_val  # pair (0,1) is very decisive
        feats, _, _ = _build_sample_graph(x_vec, pairs, scores_row, d, top_k=1)
        # nodes 0 and 1 must have strength > 0.9 (decisiveness ~ 0.96)
        assert feats[0, 2] > 0.9, f"{label}: node 0 strength={feats[0,2]:.3f}, expected >0.9"
        assert feats[1, 2] > 0.9, f"{label}: node 1 strength={feats[1,2]:.3f}, expected >0.9"


def test_degree_in_unit_interval():
    """Fix 2: degree normalised by (d-1) must lie in [0, 1].
    Before the fix, normalisation by top_k gave values far below 1
    even for highly connected nodes."""
    hs = dummy.dummy_hidden_states(n=100, seed=42)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    for g in artifact["graphs"][:20]:
        degrees = g.x[:, 1]
        assert (degrees >= 0).all() and (degrees <= 1.0 + 1e-5).all(), \
            f"degree out of [0,1]: min={degrees.min():.4f} max={degrees.max():.4f}"


def test_edge_attr_is_raw_score_not_decisiveness():
    """edge_attr should be raw p(is_correct) so GNN sees direction.
    Shape must be [E, 1] (preferred for PyG layer compatibility).
    Values must lie in (0, 1) and include both sides of 0.5."""
    hs = dummy.dummy_hidden_states(n=200, seed=43)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    for g in artifact["graphs"][:10]:
        assert g.edge_attr.ndim == 2 and g.edge_attr.shape[1] == 1, \
            f"edge_attr must be [E, 1], got {tuple(g.edge_attr.shape)}"
    all_attrs = torch.cat([g.edge_attr.view(-1) for g in artifact["graphs"]])
    assert all_attrs.min() >= 0.0 and all_attrs.max() <= 1.0
    assert (all_attrs < 0.5).any() and (all_attrs > 0.5).any()


# ------------------------------------------------------------------
# IO round-trip
# ------------------------------------------------------------------

def test_save_and_reload_still_validates(tmp_path):
    hs = dummy.dummy_hidden_states(n=150, seed=9)
    ensemble = _fit(hs)
    artifact = build_graphs(hs, ensemble)
    path = save_artifact(artifact, tmp_path / "graphs.pt")
    reloaded = load_artifact(path)
    assert schemas.validate_graphs(reloaded) == []
    assert len(reloaded["graphs"]) == 150
