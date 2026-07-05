"""Stage 4 interface: corrector + distribution-agnostic bounds.

Owner: role 4. Framework: Tyukin et al., "Coping with AI Errors with
Provable Guarantees" (Information Sciences 678, 2024), Theorem 1; also the
IJCNN 2024 paper for the weakly supervised construction.

Inputs available:
* ``scores.pt`` from stage 3 (1-D feature: GNN score), and/or
* topological feature vectors Phi extracted from ``graphs.pt`` (richer,
  paper-faithful choice: concatenated node degrees / centralities).

Split discipline: fit the Fisher projector and calibrate thresholds Delta_j
on ``split == SPLIT_CAL`` only; report empirical acceptance/rejection rates
and the theoretical bounds on ``split == SPLIT_TEST``.

Baseline for the same table: softmax-confidence thresholding using
``hidden_states["confidence"]``.

Can be developed today on synthetic data:
    uv run python scripts/make_dummy_data.py
    sc = load_artifact("artifacts/dummy/scores.pt")
"""

from __future__ import annotations


def fisher_projector(phi_correct, phi_incorrect):
    """Fisher linear discriminant w ~ (S_+ + S_-)^{-1} (mu_+ - mu_-)."""
    raise NotImplementedError("role 4")


def calibrate_threshold(projected_cal, is_correct_cal, target: str = "balanced"):
    """Pick Delta on the calibration split (trade-off acceptance vs rejection)."""
    raise NotImplementedError("role 4")


def theoretical_bounds(delta, m_plus: int, m_minus: int):
    """Bounds psi/rho on P(correct acceptance) and P(correct rejection),
    Theorem 1 of Tyukin et al. (2024)."""
    raise NotImplementedError("role 4")


def evaluate_corrector(scores, is_correct, split, delta):
    """Empirical acceptance/rejection rates on the test split vs the bounds."""
    raise NotImplementedError("role 4")
