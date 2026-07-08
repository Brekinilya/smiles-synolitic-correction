"""Stage-4 tests: Theorem-1 bound functions, Fisher projector, Algorithm-1
calibration, split discipline, end-to-end on dummy artifacts."""

import numpy as np
import pytest

from synolitic.common import dummy, schemas
from synolitic.stage4_corrector import corrector as c4


def test_rho_psi_basic_properties():
    assert 0.0 <= c4.rho(0.9, 10) <= 0.9
    assert c4.rho(0.9, 2000) > 0.85          # tightens toward Delta
    assert c4.rho(0.9, 2000) < 0.9
    assert c4.rho(0.5, 0) == 0.0
    assert c4.psi(0.0, 2000) < 0.05          # small when CDF value is 0
    assert c4.psi(0.3, 10) >= 0.3
    # monotone in the sample size
    assert c4.rho(0.8, 1000) > c4.rho(0.8, 100) > c4.rho(0.8, 20)
    assert c4.psi(0.1, 1000) < c4.psi(0.1, 100)


def test_bounds_hold_on_fresh_gaussian_data():
    """Monte-Carlo check of Lemma 1 / Theorem 1 semantics with a fixed seed:
    empirical rates on a large fresh sample must respect the bounds."""
    rng = np.random.default_rng(0)
    m_minus, m_plus = 300, 700
    # errors project LOW, correct HIGH (Algorithm-1 orientation)
    t_minus_cal = rng.normal(0.0, 1.0, m_minus)
    t_plus_cal = rng.normal(2.5, 1.0, m_plus)

    phi = np.concatenate([t_minus_cal, t_plus_cal, rng.normal(0.0, 1.0, 20000),
                          rng.normal(2.5, 1.0, 20000)])
    is_correct = np.concatenate([np.zeros(m_minus), np.ones(m_plus),
                                 np.zeros(20000), np.ones(20000)]).astype(np.int8)
    split = np.concatenate([np.full(m_minus + m_plus, schemas.SPLIT_CAL),
                            np.full(40000, schemas.SPLIT_TEST)]).astype(np.int8)

    corr = c4.fit_corrector(phi, is_correct, split, delta=0.8)
    res = c4.evaluate_corrector(corr, phi, is_correct, split)

    assert res["emp_reject_given_error"] >= res["bound_reject_given_error"]
    assert res["emp_accept_given_correct"] >= res["bound_accept_given_correct"]
    assert res["bound_reject_given_error"] > 0.5      # meaningful at M-=300
    assert res["bound_accept_given_correct"] > 0.8    # good separation here


def test_fisher_projects_correct_high():
    rng = np.random.default_rng(1)
    mu = np.zeros(8)
    mu_shift = np.full(8, 1.5)
    phi_correct = rng.normal(0, 1, (400, 8)) + mu_shift
    phi_incorrect = rng.normal(0, 1, (150, 8)) + mu
    w = c4.fisher_projector(phi_correct, phi_incorrect)
    assert (phi_correct @ w).mean() > (phi_incorrect @ w).mean()
    assert np.isclose(np.linalg.norm(w), 1.0)


def test_corrector_end_to_end_on_dummy_scores():
    sc = dummy.dummy_scores(n=4000, seed=0)
    phi = c4.phi_from_scores(sc)
    is_correct = sc["is_correct"].numpy()
    split = sc["split"].numpy()

    corr = c4.fit_corrector(phi, is_correct, split, delta=0.8,
                            feature_name="gnn_score")
    assert corr.w is None  # 1-D feature -> identity projector, whole cal used
    res = c4.evaluate_corrector(corr, phi, is_correct, split)

    assert res["emp_reject_given_error"] >= res["bound_reject_given_error"]
    assert res["emp_accept_given_correct"] >= res["bound_accept_given_correct"]
    assert res["precision_on_accepted"] > res["base_accuracy"]
    assert 0 < res["accepted_fraction"] < 1


def test_corrector_on_multid_graph_features_uses_fisher():
    hs = dummy.dummy_hidden_states(n=600, attn_subsample=8, seed=3)
    gr = dummy.dummy_graphs(hidden_states=hs, seed=3)
    phi = c4.phi_from_graphs(gr)
    assert phi.shape == (600, schemas.D_H * 4)
    is_correct = hs["is_correct"].numpy()
    split = hs["split"].numpy()

    corr = c4.fit_corrector(phi, is_correct, split, delta=0.7, seed=1,
                            feature_name="graph_topo_fisher")
    assert corr.w is not None and corr.w.shape == (schemas.D_H * 4,)
    res = c4.evaluate_corrector(corr, phi, is_correct, split)
    for key in ("bound_accept_given_correct", "bound_reject_given_error",
                "emp_accept_given_correct", "emp_reject_given_error"):
        assert 0.0 <= res[key] <= 1.0


def test_split_discipline_test_rows_do_not_affect_fit():
    sc = dummy.dummy_scores(n=2000, seed=5)
    phi = c4.phi_from_scores(sc)
    is_correct = sc["is_correct"].numpy().copy()
    split = sc["split"].numpy()

    corr_a = c4.fit_corrector(phi, is_correct, split, delta=0.8)

    corrupted = is_correct.copy()
    test_mask = split == schemas.SPLIT_TEST
    corrupted[test_mask] = 1 - corrupted[test_mask]
    corr_b = c4.fit_corrector(phi, corrupted, split, delta=0.8)

    assert corr_a.theta == corr_b.theta
    assert corr_a.m_minus == corr_b.m_minus
    assert corr_a.accept_bound == corr_b.accept_bound


def test_quantile_matches_algorithm_one_convention():
    """theta = inf{s : F-(s) >= Delta} on the error projections."""
    phi = np.concatenate([np.arange(1, 11, dtype=float), np.full(10, 100.0)])
    is_correct = np.concatenate([np.zeros(10), np.ones(10)]).astype(np.int8)
    split = np.full(20, schemas.SPLIT_CAL, dtype=np.int8)
    corr = c4.fit_corrector(phi, is_correct, split, delta=0.8)
    # errors are 1..10; F-(8) = 0.8 -> theta = 8 exactly
    assert corr.theta == 8.0
    assert corr.m_minus == 10 and corr.m_plus == 10


def test_youden_threshold_separates_but_carries_no_bounds():
    rng = np.random.default_rng(7)
    z = np.concatenate([rng.normal(0, 1, 300), rng.normal(2, 1, 700)])
    y = np.concatenate([np.zeros(300), np.ones(700)]).astype(np.int8)
    thr = c4.youden_threshold(z, y)
    accept = z >= thr
    yb = y.astype(bool)
    assert accept[yb].mean() + (~accept[~yb]).mean() > 1.5  # J >> 0
    assert 0.0 < thr < 2.0  # lies between the class means


def test_phi_graph_summary_shape_and_coverage():
    hs = dummy.dummy_hidden_states(n=120, attn_subsample=4, seed=9)
    gr = dummy.dummy_graphs(hidden_states=hs, seed=9)
    phi = c4.phi_graph_summary(gr)
    assert phi.shape == (120, 3 * schemas.N_NODE_FEATURES)
    assert np.isfinite(phi).all()
    gr["graphs"] = gr["graphs"][1:]  # missing idx must be detected
    with pytest.raises(ValueError):
        c4.phi_graph_summary(gr)


def test_phi_raw_x_shape():
    hs = dummy.dummy_hidden_states(n=64, attn_subsample=4, seed=11)
    phi = c4.phi_raw_x(hs)
    assert phi.shape == (64, schemas.D_H)
