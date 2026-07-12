"""Visualize Stage 4 AI corrector results for the presentation.

Run from the repo root:
    uv run python make_corrector_figure.py

Outputs (in figures/):
    corrector_bounds.png   -- theoretical vs empirical accept/reject rates
    corrector_calibration.png -- projection score distribution (correct vs error)
"""
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from synolitic.common import schemas
from synolitic.common.io import load_artifact
from synolitic.stage4_corrector import corrector as c4


def make_bounds_chart(results: list[dict], out_path: str):
    """Bar chart: theoretical bounds vs empirical rates for each feature."""
    names = [r["feature"] for r in results]
    labels = {
        "gnn_score": "GNN score",
        "confidence": "Softmax confidence",
        "raw_x_fisher": "Raw X (Fisher)",
        "graph_summary_fisher": "Graph summary (Fisher)",
        "graph_topo_fisher": "Graph topology (Fisher)",
    }
    display = [labels.get(n, n) for n in names]

    acc_bound = [r["bound_accept_given_correct"] for r in results]
    acc_emp   = [r["emp_accept_given_correct"]   for r in results]
    rej_bound = [r["bound_reject_given_error"]   for r in results]
    rej_emp   = [r["emp_reject_given_error"]     for r in results]

    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    ax.bar(x - 1.5*w, acc_bound, w, label="Bound: P(accept | correct)",
           color="#A5D6A7", edgecolor="#2E7D32", linewidth=1.2)
    ax.bar(x - 0.5*w, acc_emp,   w, label="Empirical: P(accept | correct)",
           color="#2E7D32")
    ax.bar(x + 0.5*w, rej_bound, w, label="Bound: P(reject | error)",
           color="#FFCC80", edgecolor="#E65100", linewidth=1.2)
    ax.bar(x + 1.5*w, rej_emp,   w, label="Empirical: P(reject | error)",
           color="#E65100")

    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=9)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="#9E9E9E", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Theorem 1 bounds vs empirical rates (test split)\n"
        "Empirical bars must always meet or exceed their bound bars",
        fontsize=12, fontweight="bold")

    # annotate delta
    delta = results[0]["delta"]
    ax.text(0.01, 0.97, f"Δ = {delta:.2f}", transform=ax.transAxes,
            fontsize=10, color="#616161", va="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def make_calibration_plot(phi_1d: np.ndarray, is_correct: np.ndarray,
                          split: np.ndarray, theta: float, out_path: str,
                          feature_name: str = "GNN score"):
    """Distribution of projection scores for correct vs error samples."""
    test = split == schemas.SPLIT_TEST
    scores_c = phi_1d[test & (is_correct == 1)]
    scores_e = phi_1d[test & (is_correct == 0)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    bins = np.linspace(0, 1, 41)
    ax.hist(scores_c, bins=bins, alpha=0.6, color="#2E7D32",
            label=f"Correct predictions (N={len(scores_c)})")
    ax.hist(scores_e, bins=bins, alpha=0.6, color="#E65100",
            label=f"Model errors (N={len(scores_e)})")
    ax.axvline(theta, color="#1565C0", linestyle="--", linewidth=2,
               label=f"Threshold θ = {theta:.3f}")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100],
                     0, theta, alpha=0.08, color="#E65100")
    ax.text(theta / 2, 1, "REJECT", ha="center", fontsize=10,
            color="#E65100", fontweight="bold")
    ax.text((1 + theta) / 2, 1, "ACCEPT", ha="center", fontsize=10,
            color="#2E7D32", fontweight="bold")
    ax.set_xlabel(feature_name, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"Score distribution on test split\n"
        f"Threshold θ separates correct from error predictions (H2)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",       default="artifacts/scores.pt")
    parser.add_argument("--hidden-states",default="artifacts/hidden_states.pt")
    parser.add_argument("--graphs",       default="artifacts/graphs.pt")
    parser.add_argument("--delta",        type=float, default=0.8)
    parser.add_argument("--seed",         type=int,   default=0)
    args = parser.parse_args()

    scores_art = load_artifact(args.scores)
    is_correct = scores_art["is_correct"].numpy()
    split      = scores_art["split"].numpy()

    features = {"gnn_score": c4.phi_from_scores(scores_art)}
    if os.path.exists(args.hidden_states):
        hs = load_artifact(args.hidden_states)
        features["confidence"]   = c4.phi_from_confidence(hs)
        features["raw_x_fisher"] = c4.phi_raw_x(hs)
    if os.path.exists(args.graphs):
        gr = load_artifact(args.graphs)
        features["graph_summary_fisher"] = c4.phi_graph_summary(gr)

    results = []
    for name, phi in features.items():
        try:
            corr = c4.fit_corrector(phi, is_correct, split,
                                    delta=args.delta, seed=args.seed,
                                    feature_name=name)
            r = c4.evaluate_corrector(corr, phi, is_correct, split)
            results.append(r)
            print(f"{name}: bound_accept={r['bound_accept_given_correct']:.3f} "
                  f"emp_accept={r['emp_accept_given_correct']:.3f} | "
                  f"bound_reject={r['bound_reject_given_error']:.3f} "
                  f"emp_reject={r['emp_reject_given_error']:.3f} | "
                  f"AUC={r['auc_test']:.3f}")
        except ValueError as e:
            print(f"  [{name}] skipped: {e}")

    os.makedirs("figures", exist_ok=True)

    if results:
        make_bounds_chart(results, "figures/corrector_bounds.png")

    # calibration plot on GNN score (1-D, most interpretable)
    gnn_phi = c4.phi_from_scores(scores_art)
    try:
        corr_gnn = c4.fit_corrector(gnn_phi, is_correct, split,
                                    delta=args.delta, seed=args.seed,
                                    feature_name="gnn_score")
        make_calibration_plot(gnn_phi, is_correct, split,
                              corr_gnn.theta,
                              "figures/corrector_calibration.png",
                              feature_name="GNN score P(is_correct)")
    except ValueError as e:
        print(f"  calibration plot skipped: {e}")


if __name__ == "__main__":
    main()
