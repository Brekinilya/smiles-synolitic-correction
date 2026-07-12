"""Visualize synolitic graphs: correct vs error, on REAL data.

Run from the repo root:

    uv run python make_graph_figure.py
    uv run python make_graph_figure.py --top-k 80

Output: figures/correct_vs_error_graphs.png  (200 dpi, slide-ready)
"""
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from synolitic.common import schemas
from synolitic.common.io import load_artifact


def to_nx(g, top_k):
    ea = g.edge_attr.view(-1).numpy()
    ei = g.edge_index.numpy()
    decisiveness = np.abs(ea - 0.5) * 2
    fwd = np.arange(0, ei.shape[1], 2)
    top_idx = fwd[np.argsort(-decisiveness[fwd])[:top_k]]
    G = nx.Graph()
    G.add_nodes_from(range(g.x.shape[0]))
    for k in top_idx:
        u, v = int(ei[0, k]), int(ei[1, k])
        G.add_edge(u, v, score=float(ea[k]))
    return G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=80)
    args = parser.parse_args()

    art = load_artifact("artifacts/graphs.pt")
    graphs = art["graphs"]

    test_graphs = [g for g in graphs if int(g.split) == schemas.SPLIT_TEST]
    correct = [g for g in test_graphs if g.y.item() == 1.0]
    errors  = [g for g in test_graphs if g.y.item() == 0.0]

    strength_c = np.array([g.x[:, 2].mean().item() for g in correct])
    strength_e = np.array([g.x[:, 2].mean().item() for g in errors])
    print(f"test split: {len(correct)} correct, {len(errors)} errors")
    print(f"mean node strength | correct: {strength_c.mean():.3f} +- {strength_c.std():.3f}")
    print(f"mean node strength | errors:  {strength_e.mean():.3f} +- {strength_e.std():.3f}")

    correct_g = correct[int(np.argmax(strength_c))]
    error_g   = errors[int(np.argmin(strength_e))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, g, title, accent in [
        (axes[0], correct_g, "Correct prediction", "#2E7D32"),
        (axes[1], error_g,   "Model error",        "#C62828"),
    ]:
        G = to_nx(g, args.top_k)
        strength = g.x[:, 2].numpy()
        pos = nx.spring_layout(G, seed=3, k=0.35)
        edge_colors = ["#1565C0" if G[u][v]["score"] > 0.5 else "#E65100"
                       for u, v in G.edges()]
        node_sizes = 40 + 260 * (strength / (strength.max() + 1e-8))
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=1.2, edge_color=edge_colors)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=strength, cmap="viridis")
        n_err = sum(1 for u, v in G.edges() if G[u][v]["score"] < 0.5)
        ax.set_title(
            f"{title}\n(mean strength {strength.mean():.2f}, "
            f"{n_err}/{G.number_of_edges()} error-leaning edges)",
            fontsize=13, color=accent, fontweight="bold")
        ax.axis("off")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0],[0], color="#1565C0", lw=2, label="correct-leaning edge (score > 0.5)"),
        Line2D([0],[0], color="#E65100", lw=2, label="error-leaning edge (score < 0.5)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#440154",
               markersize=9, label="node size/color = strength"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle(
        "Synolitic graphs are sample-specific: topology differs between "
        "correct and error samples (real data, test split)", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs("figures", exist_ok=True)
    out = "figures/correct_vs_error_graphs.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
