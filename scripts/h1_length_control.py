"""H1 under length control — is the GNN's global AUC edge over confidence just a
length confound?

Backs the "H1 under length control" section of docs/results.md. Every number is
computed from the released artifacts (artifacts/hidden_states.pt @ v0.1-data,
artifacts/scores.pt @ v0.3-data); the calibration split is never touched.

Run:
    uv run python scripts/h1_length_control.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

from synolitic.common import schemas
from synolitic.common.io import load_artifact

RNG = np.random.default_rng(20260721)
N_BOOT = 2000
MIN_BIN = 20  # ignore length bins with fewer samples in the within-length mean


def within_length_auc(y, score, length, min_bin: int = MIN_BIN) -> float:
    """Sample-weighted mean of per-length ROC-AUC over bins holding both classes."""
    num = den = 0.0
    for lv in np.unique(length):
        m = length == lv
        if m.sum() >= min_bin and np.unique(y[m]).size == 2:
            num += roc_auc_score(y[m], score[m]) * m.sum()
            den += m.sum()
    return num / den if den else float("nan")


def main() -> None:
    hs = load_artifact("artifacts/hidden_states.pt")
    sc = load_artifact("artifacts/scores.pt")

    X = hs["X"].numpy()
    lengths = hs["lengths"].numpy()
    conf = hs["confidence"].numpy()
    gnn = sc["scores"].numpy()
    ic = sc["is_correct"].numpy()
    split = sc["split"].numpy()
    assert (ic == hs["is_correct"].numpy()).all(), "scores/hidden_states is_correct misaligned"

    train = split == schemas.SPLIT_TRAIN
    test = split == schemas.SPLIT_TEST
    yt, gt, ct, lt = ic[test], gnn[test], conf[test], lengths[test]

    # 1) global vs within-length AUC (test split)
    g_glob, c_glob = roc_auc_score(yt, gt), roc_auc_score(yt, ct)
    g_wl, c_wl = within_length_auc(yt, gt, lt), within_length_auc(yt, ct, lt)
    print("=== AUC (test split) ===")
    print(f"  global      GNN {g_glob:.4f}  conf {c_glob:.4f}  diff {g_glob - c_glob:+.4f}")
    print(f"  within-len  GNN {g_wl:.4f}  conf {c_wl:.4f}  diff {g_wl - c_wl:+.4f}")

    # 2) bootstrap CI on the within-length difference (resample the test split)
    n = int(test.sum())
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        b = RNG.integers(0, n, n)
        diffs[i] = within_length_auc(yt[b], gt[b], lt[b]) - within_length_auc(yt[b], ct[b], lt[b])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  bootstrap   within-len diff {diffs.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({diffs.size}x)")

    # 3) is sequence length recoverable from the pooled hidden state?
    ridge = Ridge(alpha=1.0).fit(X[train], lengths[train])
    r2 = r2_score(lengths[test], ridge.predict(X[test]))
    print("\n=== length <- pooled hidden state ===")
    print(f"  ridge R^2 (train-fit, test-score): {r2:.4f}")

    # 4) exact-match collapse at L=20 (model-level, all N)
    m20 = lengths == 20
    err_tot = int((ic == 0).sum())
    err20 = int(((ic == 0) & m20).sum())
    print(f"\n=== exact-match accuracy by length (all N={len(ic)}) ===")
    for lv in range(int(lengths.min()), int(lengths.max()) + 1):
        m = lengths == lv
        if m.sum():
            print(f"  L={lv:>2}  n={int(m.sum()):>5}  acc={ic[m].mean():.3f}")
    print(f"  L=20 acc {ic[m20].mean():.3f} vs rest {ic[~m20].mean():.3f};  "
          f"L=20 errors {err20}/{err_tot} = {err20 / err_tot:.1%} of all errors")

    # 5) correlation of each score with length (Pearson). Reported on the test
    #    split and over all N; docs/results.md cites the all-N figures (a
    #    model-level property, distinct from the test-split AUC comparison).
    print("\n=== Pearson r of score with length ===")
    for name, m in (("test ", test), ("all N", np.ones(len(lengths), dtype=bool))):
        rc = np.corrcoef(conf[m], lengths[m])[0, 1]
        rg = np.corrcoef(gnn[m], lengths[m])[0, 1]
        print(f"  {name}  confidence {rc:+.2f}   GNN {rg:+.2f}")


if __name__ == "__main__":
    main()
