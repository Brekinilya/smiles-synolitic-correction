"""H1 evidence: node feature distributions correct vs error (real data).

Run from the repo root:
    uv run python make_h1_figure.py

Output: figures/node_feature_distributions.png

Shows distributions of all 5 node features across correct and error
predictions on the test split. Differences between the two groups
support H1: synolitic graph topology encodes error-discriminative signal.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from synolitic.common import schemas
from synolitic.common.io import load_artifact


def main():
    art = load_artifact("artifacts/graphs.pt")
    graphs = art["graphs"]

    test    = [g for g in graphs if int(g.split) == schemas.SPLIT_TEST]
    correct = [g for g in test if g.y.item() == 1.0]
    errors  = [g for g in test if g.y.item() == 0.0]
    print(f"test split: {len(correct)} correct, {len(errors)} errors")

    feat_names = ["value", "degree", "strength", "closeness", "betweenness"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor("white")

    for col, (ax, fname) in enumerate(zip(axes, feat_names)):
        vals_c = np.concatenate([g.x[:, col].numpy() for g in correct])
        vals_e = np.concatenate([g.x[:, col].numpy() for g in errors])

        lo = min(np.percentile(vals_c, 1), np.percentile(vals_e, 1))
        hi = max(np.percentile(vals_c, 99), np.percentile(vals_e, 99))
        bins = np.linspace(lo, hi, 40)

        ax.hist(vals_c, bins=bins, alpha=0.6, color="#2E7D32",
                density=True, label=f"Correct (N={len(correct)})")
        ax.hist(vals_e, bins=bins, alpha=0.6, color="#E65100",
                density=True, label=f"Errors (N={len(errors)})")
        ax.set_title(fname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        if col == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # Mann-Whitney U test + mean difference
        stat, pval = stats.mannwhitneyu(vals_c, vals_e, alternative="two-sided")
        diff = vals_c.mean() - vals_e.mean()
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        ax.text(0.98, 0.97,
                f"Δmean={diff:+.3f}\np={pval:.2e} {stars}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#616161")
        print(f"  {fname}: mean correct={vals_c.mean():.4f}, "
              f"mean error={vals_e.mean():.4f}, p={pval:.2e} {stars}")

    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Node feature distributions: correct vs error predictions (test split)\n"
        "Statistical differences support H1: synolitic topology encodes error-discriminative signal",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = "figures/node_feature_distributions.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
