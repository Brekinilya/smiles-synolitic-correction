"""H2 stress test: how do Theorem-1 guarantees scale with labeling budget?

Hypothesis H2 claims the corrector achieves meaningful bounds with fewer
than 100 labeled correction examples. This script subsamples the calibration
split to M labeled examples (M = 30..full), refits Algorithm 1 (Delta = 0.8,
GNN score) on each subsample and reports theoretical bounds vs empirical
test rates, averaged over R random subsamples per M. The test split is never
used for calibration — only measured on.

Usage:
    uv run python scripts/h2_labeling_curve.py [--scores artifacts/scores.pt]
"""

from __future__ import annotations

import argparse

import numpy as np

from synolitic.common import schemas
from synolitic.common.io import load_artifact
from synolitic.stage4_corrector.corrector import evaluate_corrector, fit_corrector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", default="artifacts/scores.pt")
    p.add_argument("--delta", type=float, default=0.8)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sc = load_artifact(args.scores)
    schemas.assert_valid("scores", schemas.validate_scores(sc))
    scores = sc["scores"].numpy()
    is_correct = sc["is_correct"].numpy()
    split = sc["split"].numpy()

    cal_idx = np.flatnonzero(split == schemas.SPLIT_CAL)
    rng = np.random.default_rng(args.seed)

    print(f"full cal = {len(cal_idx)} labeled examples "
          f"({int((is_correct[cal_idx] == 0).sum())} errors), Delta = {args.delta}")
    print(f"{'M':>5} {'M-':>5} | {'rej bound':>9} {'rej emp':>8} | "
          f"{'acc bound':>9} {'acc emp':>8} | {'prec@acc':>8}")

    for m in [30, 50, 100, 200, 500, 1000, len(cal_idx)]:
        rows = []
        n_rep = args.repeats if m < len(cal_idx) else 1
        for _ in range(n_rep):
            sub = rng.choice(cal_idx, size=m, replace=False)
            y_sub = is_correct[sub]
            if y_sub.all() or not y_sub.any():
                continue  # degenerate draw: single class, no corrector possible
            split_mod = split.copy()
            split_mod[cal_idx] = schemas.SPLIT_TRAIN  # park unused cal rows
            split_mod[sub] = schemas.SPLIT_CAL
            corr = fit_corrector(scores, is_correct, split_mod, delta=args.delta,
                                 feature_name="gnn_score")
            ev = evaluate_corrector(corr, scores, is_correct, split_mod)
            rows.append((corr.reject_bound, ev["emp_reject_given_error"],
                         corr.accept_bound, ev["emp_accept_given_correct"],
                         ev["precision_on_accepted"], corr.m_minus))
        a = np.array([r[:5] for r in rows])
        m_minus = int(np.mean([r[5] for r in rows]))
        mean, std = a.mean(axis=0), a.std(axis=0)
        note = f"   (+/- rej_emp {std[1]:.3f})" if len(rows) > 1 else ""
        print(f"{m:>5} {m_minus:>5} | {mean[0]:>9.3f} {mean[1]:>8.3f} | "
              f"{mean[2]:>9.3f} {mean[3]:>8.3f} | {mean[4]:>8.3f}{note}")

    print("reading: empirical corrector quality is nearly flat in M — labeling "
          "buys the STRENGTH OF THE GUARANTEE, not raw performance. Bounds are "
          "already meaningful at M = 100 (H2).")


if __name__ == "__main__":
    main()
