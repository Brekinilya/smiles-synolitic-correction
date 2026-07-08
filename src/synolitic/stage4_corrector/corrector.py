"""Stage 4: AI error corrector with distribution-agnostic guarantees.

Implements Algorithm 1 and Theorem 1 of Tyukin et al., "Weakly Supervised
Learners for Correction of AI Errors with Provable Performance Guarantees"
(IJCNN 2024, arXiv:2402.00899); the journal version (Information Sciences
678:120856, 2024) states the analogous bounds. Formulas below follow the
arXiv text verbatim.

Construction (single-class specialization: we moderate every answer of the
reversal transformer, so the paper's per-class index j collapses to one
class; S+ = samples the transformer got right, S- = its errors):

* projector h maps the feature vector Phi(u) to R, oriented so that CORRECT
  samples project HIGH;
* threshold theta = F-_dagger(Delta), the Delta-quantile of the projections
  of S- (errors);
* corrector: REJECT the model's answer iff h(Phi(u)) <= theta (Algorithm 1).

Theorem 1 bounds (both distribution-agnostic, DKW-based):

* P(accept | model correct) >= 1 - psi(F+(theta), M+),
* P(reject | model wrong)  >= rho(Delta, M-),

  rho(a, d) = sup_eps  max{a - eps, 0} * (1 - 2 exp(-2 d eps^2)),
  psi(a, d) = inf_eps  2 exp(-2 d eps^2) + min{1, a + eps},

where F+ is the empirical CDF of correct-sample projections and M+/M- are
the calibration counts.

Independence requirement of Theorem 1: h must be chosen independently of
the calibration sample S used for F+/F-. Therefore:

* 1-D features (GNN score, softmax confidence) use the identity projector —
  fixed a priori, the whole cal split serves as S;
* multi-D features (synolitic graph topology) fit a Fisher discriminant on
  one half of the cal split (cal-A) and compute theta/F+ on the other half
  (cal-B), so the projector stays independent of S = cal-B.

A second, bounds-free operating point is provided by ``youden_threshold``
(threshold maximizing TPR + TNR on the calibration sample). It is often a
good empirical trade-off but is tuned on both classes at once, so it does
NOT satisfy the premise of Theorem 1 — report empirical rates only for it.

Split discipline: everything here uses split == SPLIT_CAL only; report
final numbers on split == SPLIT_TEST (see ``evaluate_corrector``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from synolitic.common.schemas import SPLIT_CAL, SPLIT_TEST

_EPS_GRID = np.linspace(1e-4, 1.0, 20000)


def rho(a: float, d: int) -> float:
    """Lower bound on P(new sample <= empirical a-quantile), Theorem 1."""
    if d <= 0:
        return 0.0
    vals = np.maximum(a - _EPS_GRID, 0.0) * (1.0 - 2.0 * np.exp(-2.0 * d * _EPS_GRID**2))
    return float(np.clip(vals.max(), 0.0, 1.0))


def psi(a: float, d: int) -> float:
    """Upper bound on P(new sample <= theta) given empirical CDF value a."""
    if d <= 0:
        return 1.0
    vals = 2.0 * np.exp(-2.0 * d * _EPS_GRID**2) + np.minimum(1.0, a + _EPS_GRID)
    return float(np.clip(vals.min(), 0.0, 1.0))


def fisher_projector(
    phi_correct: np.ndarray, phi_incorrect: np.ndarray, ridge: float = 1e-3
) -> np.ndarray:
    """Fisher linear discriminant w ~ (S_w + ridge*I)^-1 (mu+ - mu-).

    Oriented so that CORRECT samples project HIGH (matches Algorithm 1's
    reject-if-low convention). Ridge is scaled by mean(trace)/dim to keep
    the solve stable for high-dimensional Phi with modest sample counts.
    """
    mu_p = phi_correct.mean(axis=0)
    mu_m = phi_incorrect.mean(axis=0)
    d = phi_correct.shape[1]
    sw = np.cov(phi_correct, rowvar=False) * (len(phi_correct) - 1)
    sw += np.cov(phi_incorrect, rowvar=False) * (len(phi_incorrect) - 1)
    sw += np.eye(d) * max(ridge * np.trace(sw) / d, 1e-12)
    w = np.linalg.solve(sw, mu_p - mu_m)
    norm = np.linalg.norm(w)
    return w / norm if norm > 0 else w


def youden_threshold(projected: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximizing TPR + TNR (Youden's J) on the given sample.

    Data-driven operating point tuned on both classes at once — often a good
    empirical trade-off, but it does NOT satisfy the premise of Theorem 1
    (theta must be the Delta-quantile of error projections with Delta fixed
    a priori), so no bound is claimed for it.
    """
    z = np.asarray(projected, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    thresholds = np.unique(z)
    accept = z[None, :] >= thresholds[:, None]
    tpr = accept[:, y].mean(axis=1) if y.any() else np.zeros(len(thresholds))
    tnr = (~accept[:, ~y]).mean(axis=1) if (~y).any() else np.zeros(len(thresholds))
    return float(thresholds[int(np.argmax(tpr + tnr))])


@dataclass
class Corrector:
    """Fitted corrector: projector + threshold + Theorem-1 bounds."""

    delta: float
    theta: float
    m_plus: int
    m_minus: int
    f_plus_at_theta: float          # empirical CDF of correct projections at theta
    accept_bound: float             # 1 - psi(F+(theta), M+)
    reject_bound: float             # rho(Delta, M-)
    w: np.ndarray | None = None     # None => identity projector (1-D Phi)
    feature_name: str = "phi"
    notes: dict = field(default_factory=dict)

    def project(self, phi: np.ndarray) -> np.ndarray:
        t = np.asarray(phi, dtype=np.float64)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if self.w is not None:
            t = t @ self.w
        return t

    def accept_mask(self, phi: np.ndarray) -> np.ndarray:
        """True = accept the model's answer; False = reject (Algorithm 1)."""
        return self.project(phi) > self.theta


def _quantile_inf(values: np.ndarray, level: float) -> float:
    """Generalized inverse CDF F_dagger(level) = inf{s : F(s) >= level}."""
    return float(np.quantile(values, level, method="inverted_cdf"))


def fit_corrector(
    phi: np.ndarray,
    is_correct: np.ndarray,
    split: np.ndarray,
    delta: float = 0.8,
    fisher_frac: float = 0.5,
    seed: int = 0,
    feature_name: str = "phi",
) -> Corrector:
    """Fit Algorithm 1 on the calibration split.

    Args:
        phi: [N, d] or [N] features. 1-D features must already be oriented
            "higher = more likely correct" (true for GNN score = P(correct)
            and for softmax confidence).
        is_correct / split: [N] int8 arrays aligned with phi.
        delta: target quantile of error projections (the paper's Delta).
        fisher_frac: fraction of cal rows used to fit the Fisher projector
            when phi is multi-dimensional (rest forms S for theta/F+/bounds).
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 1:
        phi = phi[:, None]
    is_correct = np.asarray(is_correct).astype(bool)
    cal = np.asarray(split) == SPLIT_CAL
    if not cal.any():
        raise ValueError("no calibration rows (split == SPLIT_CAL) found")

    cal_idx = np.flatnonzero(cal)
    w = None
    notes: dict = {}
    if phi.shape[1] > 1:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(cal_idx))
        n_a = int(round(fisher_frac * len(cal_idx)))
        idx_a, idx_b = cal_idx[perm[:n_a]], cal_idx[perm[n_a:]]
        y_a = is_correct[idx_a]
        if y_a.all() or not y_a.any():
            raise ValueError("cal-A has a single class; cannot fit Fisher projector")
        w = fisher_projector(phi[idx_a][y_a], phi[idx_a][~y_a])
        notes["fisher_fit_on"] = int(len(idx_a))
        s_idx = idx_b
    else:
        s_idx = cal_idx  # identity projector is a priori => whole cal is S

    t = phi[s_idx] @ w if w is not None else phi[s_idx, 0]
    y = is_correct[s_idx]
    t_minus, t_plus = t[~y], t[y]
    if len(t_minus) == 0 or len(t_plus) == 0:
        raise ValueError("S+ or S- is empty on the calibration split")

    theta = _quantile_inf(t_minus, delta)
    f_plus_at_theta = float((t_plus <= theta).mean())
    m_plus, m_minus = int(len(t_plus)), int(len(t_minus))

    return Corrector(
        delta=delta,
        theta=theta,
        m_plus=m_plus,
        m_minus=m_minus,
        f_plus_at_theta=f_plus_at_theta,
        accept_bound=float(np.clip(1.0 - psi(f_plus_at_theta, m_plus), 0.0, 1.0)),
        reject_bound=rho(delta, m_minus),
        w=w,
        feature_name=feature_name,
        notes=notes,
    )


def evaluate_corrector(
    corrector: Corrector,
    phi: np.ndarray,
    is_correct: np.ndarray,
    split: np.ndarray,
) -> dict:
    """Empirical test-split performance vs the Theorem-1 bounds."""
    test = np.asarray(split) == SPLIT_TEST
    if not test.any():
        raise ValueError("no test rows (split == SPLIT_TEST) found")
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 1:
        phi = phi[:, None]
    y = np.asarray(is_correct).astype(bool)[test]
    t = corrector.project(phi[test])
    accept = t > corrector.theta

    try:
        from sklearn.metrics import roc_auc_score

        auc_test = float(roc_auc_score(y, t))
    except ValueError:
        auc_test = float("nan")

    base_accuracy = float(y.mean())
    n_acc = int(accept.sum())
    return {
        "feature": corrector.feature_name,
        "delta": corrector.delta,
        "theta": corrector.theta,
        "m_plus_cal": corrector.m_plus,
        "m_minus_cal": corrector.m_minus,
        "bound_accept_given_correct": corrector.accept_bound,
        "bound_reject_given_error": corrector.reject_bound,
        "emp_accept_given_correct": float(accept[y].mean()) if y.any() else float("nan"),
        "emp_reject_given_error": float((~accept[~y]).mean()) if (~y).any() else float("nan"),
        "auc_test": auc_test,
        "base_accuracy": base_accuracy,
        "precision_on_accepted": float(y[accept].mean()) if n_acc else float("nan"),
        "accepted_fraction": float(accept.mean()),
        "n_test": int(test.sum()),
    }


# ---------------------------------------------------------------------------
# Feature builders (Phi) from the pipeline artifacts
# ---------------------------------------------------------------------------

def phi_from_scores(scores_artifact: dict) -> np.ndarray:
    """1-D Phi = GNN score = P(is_correct). Higher = more likely correct."""
    return scores_artifact["scores"].numpy().astype(np.float64)


def phi_from_confidence(hidden_states: dict) -> np.ndarray:
    """1-D Phi = model softmax confidence (the baseline the paper-style
    corrector must beat)."""
    return hidden_states["confidence"].numpy().astype(np.float64)


def phi_from_graphs(graphs_artifact: dict, feature_cols: tuple[int, ...] = (1, 2, 3, 4)) -> np.ndarray:
    """Multi-D Phi = concatenated topological node features of the synolitic
    graph (defaults to degree/strength/closeness/betweenness columns; column
    0 is the raw pooled hidden-state value). Rows ordered by ``data.idx``."""
    graphs = graphs_artifact["graphs"]
    order = np.argsort([int(g.idx) for g in graphs])
    return np.stack(
        [graphs[i].x[:, list(feature_cols)].numpy().ravel() for i in order]
    ).astype(np.float64)


def phi_graph_summary(graphs_artifact: dict) -> np.ndarray:
    """Compact per-graph Phi [N, 15]: mean/std/max over the 64 nodes of the
    5 node features. Coarser than ``phi_from_graphs`` (which keeps per-node
    structure) but far lower-dimensional — keep both as ablations."""
    graphs = graphs_artifact["graphs"]
    n = len(graphs)
    width = 3 * graphs[0].x.shape[1]
    feats = np.full((n, width), np.nan)
    for g in graphs:
        idx = int(g.idx)
        if not 0 <= idx < n:
            raise ValueError(f"graph idx {idx} outside 0..{n - 1}")
        x = g.x.numpy()
        feats[idx] = np.concatenate([x.mean(axis=0), x.std(axis=0), x.max(axis=0)])
    if np.isnan(feats).any():
        raise ValueError("graphs do not cover all indices 0..N-1")
    return feats.astype(np.float64)


def phi_raw_x(hidden_states: dict) -> np.ndarray:
    """Multi-D Phi = raw pooled hidden states X — the H1 comparison row
    (does the synolitic graph add anything over raw features?)."""
    return hidden_states["X"].numpy().astype(np.float64)
