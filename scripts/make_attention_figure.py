"""Visualize encoder self-attention from REAL hidden_states.pt.

Run from the repo root:
    uv run python make_attention_figure.py

Output: figures/attention_heatmaps.png
Shows Layer 2 (most specialized) for correct vs error sample.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from synolitic.common import schemas
from synolitic.common.io import load_artifact


def main():
    hs = load_artifact("artifacts/hidden_states.pt")
    enc = hs["enc_self_attn"]       # [n_sub, layers, heads, Ls, Ls]
    attn_idx = hs["attn_idx"]
    is_correct = hs["is_correct"]

    n_sub, layers, heads, ls, _ = enc.shape
    print(f"attention dump: {n_sub} samples, {layers} layers, {heads} heads, seq_len={ls}")

    correct_i = next(i for i in range(n_sub) if is_correct[int(attn_idx[i])] == 1)
    error_i   = next(i for i in range(n_sub) if is_correct[int(attn_idx[i])] == 0)

    # Show only the last layer (most specialized)
    layer = layers - 1

    fig, axes = plt.subplots(2, heads, figsize=(3.5 * heads, 3.5 * 2))
    fig.patch.set_facecolor("white")

    for row, (sample_i, label, color) in enumerate([
        (correct_i, "Correct prediction", "#2E7D32"),
        (error_i,   "Model error",        "#C62828"),
    ]):
        for h in range(heads):
            ax = axes[row, h]
            data = enc[sample_i, layer, h].numpy()
            ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
            ax.set_title(f"Head {h+1}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(label, fontsize=10, color=color, fontweight="bold")

    fig.suptitle(
        f"Layer {layer+1} encoder self-attention: correct vs error predictions\n"
        "Heads show distinct specialization patterns (groundwork for H3)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = "figures/attention_heatmaps.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
