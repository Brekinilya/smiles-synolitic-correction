"""Run the stage-4 corrector on dummy or real artifacts and print the
empirical-vs-bounds comparison across every feature set (H1 table).

Rows: GNN score (stage 3), softmax confidence (baseline), Fisher on raw X
(H1 reference), Fisher on synolitic graph topology (per-node and compact
summary variants). For each: Theorem-1 bounds vs empirical test rates at
the requested Delta, plus the bounds-free Youden operating point.

Usage (dummy, default):
    uv run python scripts/make_dummy_data.py --n 4000
    uv run python scripts/run_corrector.py

Usage (real, once stage 3 delivers scores.pt):
    uv run python scripts/run_corrector.py \
        --scores artifacts/scores.pt \
        --hidden-states artifacts/hidden_states.pt \
        --graphs artifacts/graphs.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from synolitic.common import schemas
from synolitic.common.io import load_artifact
from synolitic.stage4_corrector import corrector as c4


def youden_line(name: str, phi, is_correct, split) -> None:
    """Empirical-only operating point (no Theorem-1 claim)."""
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 1:
        phi = phi[:, None]
    cal = np.asarray(split) == schemas.SPLIT_CAL
    test = np.asarray(split) == schemas.SPLIT_TEST
    y = np.asarray(is_correct).astype(bool)
    if phi.shape[1] > 1:
        y_cal = y[cal]
        w = c4.fisher_projector(phi[cal][y_cal], phi[cal][~y_cal])
        t = phi @ w
    else:
        t = phi[:, 0]
    thr = c4.youden_threshold(t[cal], y[cal])
    accept = t[test] > thr
    yt = y[test]
    print(f"{name + ' @youden':>22} {'-':>5} {'-':>6} "
          f"{float(accept[yt].mean()):>9.3f} {'-':>6} {float((~accept[~yt]).mean()):>9.3f} "
          f"{float(yt[accept].mean()) if accept.any() else float('nan'):>9.3f} "
          f"{float(accept.mean()):>6.3f}   (no bounds)")


def run_one(name: str, phi, is_correct, split, delta: float, seed: int) -> dict | None:
    try:
        corr = c4.fit_corrector(phi, is_correct, split, delta=delta, seed=seed,
                                feature_name=name)
        return c4.evaluate_corrector(corr, phi, is_correct, split)
    except ValueError as e:
        print(f"  [{name}] skipped: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, default=Path("artifacts/dummy/scores.pt"))
    p.add_argument("--hidden-states", type=Path, default=Path("artifacts/dummy/hidden_states.pt"))
    p.add_argument("--graphs", type=Path, default=Path("artifacts/dummy/graphs.pt"))
    p.add_argument("--deltas", type=float, nargs="+", default=[0.8, 0.9])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scores_art = load_artifact(args.scores)
    schemas.assert_valid("scores", schemas.validate_scores(scores_art))
    is_correct = scores_art["is_correct"].numpy()
    split = scores_art["split"].numpy()

    features: dict[str, object] = {"gnn_score": c4.phi_from_scores(scores_art)}
    if args.hidden_states.exists():
        hs = load_artifact(args.hidden_states)
        schemas.assert_valid("hidden_states", schemas.validate_hidden_states(hs))
        features["confidence"] = c4.phi_from_confidence(hs)
        features["raw_x_fisher"] = c4.phi_raw_x(hs)
    if args.graphs.exists():
        gr = load_artifact(args.graphs)
        schemas.assert_valid("graphs", schemas.validate_graphs(gr))
        features["graph_topo_fisher"] = c4.phi_from_graphs(gr)
        features["graph_summary_fisher"] = c4.phi_graph_summary(gr)

    header = f"{'feature':>22} {'Delta':>5} {'acc>=':>6} {'acc(emp)':>9} " \
             f"{'rej>=':>6} {'rej(emp)':>9} {'prec@acc':>9} {'kept':>6}   {'AUC(test)'}"
    base = None
    auc: dict[str, float] = {}
    print(header)
    print("-" * len(header))
    for delta in args.deltas:
        for name, phi in features.items():
            r = run_one(name, phi, is_correct, split, delta, args.seed)
            if r is None:
                continue
            base = r["base_accuracy"]
            auc[name] = r["auc_test"]
            print(f"{r['feature']:>22} {r['delta']:>5.2f} "
                  f"{r['bound_accept_given_correct']:>6.3f} {r['emp_accept_given_correct']:>9.3f} "
                  f"{r['bound_reject_given_error']:>6.3f} {r['emp_reject_given_error']:>9.3f} "
                  f"{r['precision_on_accepted']:>9.3f} {r['accepted_fraction']:>6.3f}   "
                  f"{r['auc_test']:.3f}")
        print()

    for name, phi in features.items():
        try:
            youden_line(name, phi, is_correct, split)
        except ValueError as e:
            print(f"  [{name} @youden] skipped: {e}")

    if base is not None:
        print(f"\nbase accuracy on test split (no corrector): {base:.3f}")
        print("reading: prec@acc must exceed base accuracy; emp columns must "
              "respect their bounds (acc >= acc-bound, rej >= rej-bound)")
    if "graph_topo_fisher" in auc and "raw_x_fisher" in auc:
        g, r = auc["graph_topo_fisher"], auc["raw_x_fisher"]
        verdict = "SUPPORTED" if g >= r else "NOT supported"
        print(f"H1 check (projected-feature AUC on test): "
              f"synolitic {g:.3f} vs raw X {r:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
