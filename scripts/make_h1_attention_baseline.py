"""H1 test: synolitic graphs vs raw attention features.

Run from the repo root:
    uv run python make_h1_attention_baseline.py

Compares three approaches on the SAME samples (attention subsample):
1. LogReg on flat attention weights  (raw attention, no graph)
2. LogReg on flat synolitic features (graph features, no GNN)
3. GNN on attention-as-graph         (attention with graph structure)

The synolitic GNN AUC (0.839) is loaded from artifacts/scores.pt.
All three are compared against the softmax-confidence baseline (0.799).

Output:
  figures/h1_attention_comparison.png  -- bar chart of AUC values
  Printed table with all numbers.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from synolitic.common import schemas
from synolitic.common.io import load_artifact


# ------------------------------------------------------------------
# Build attention-based graphs
# ------------------------------------------------------------------

def build_attention_graphs(hs: dict, graphs_art: dict) -> list[Data]:
    """Build one graph per sample using attention weights as edges.

    Node i features: [mean attention received by token i across all
    heads/layers, std, max] -- shape [Ls, 3] then padded/truncated to
    match D_H=64 nodes by repeating across sequence positions.

    Edge (i, j): mean attention from i to j across all heads/layers.
    We keep the top-k most attended edges (same k as synolitic graphs).
    """
    enc = hs["enc_self_attn"].numpy()   # [n_sub, L, H, Ls, Ls]
    attn_idx = hs["attn_idx"].numpy()
    is_correct = hs["is_correct"].numpy()
    split = hs["split"].numpy()

    n_sub, L, H, Ls, _ = enc.shape
    d_h = schemas.D_H  # 64

    # mean attention matrix [n_sub, Ls, Ls]
    mean_attn = enc.mean(axis=(1, 2))  # avg over layers and heads

    # top-k edges to keep (same ratio as synolitic)
    n_pairs = Ls * (Ls - 1)
    top_k = max(1, int(0.125 * n_pairs))

    graphs = []
    for i in range(n_sub):
        mat = mean_attn[i]  # [Ls, Ls]

        # node features: for each token position, compute
        # [mean_received, std_received, max_received] -- 3 features
        # then tile to d_h=64 nodes
        received = mat.mean(axis=0)   # [Ls] -- how much each pos is attended to
        std_recv = mat.std(axis=0)
        max_recv = mat.max(axis=0)
        sent = mat.mean(axis=1)       # [Ls] -- how much each pos attends to others

        # repeat to d_h nodes (tile or truncate)
        def tile_to_dh(arr):
            reps = (d_h + len(arr) - 1) // len(arr)
            return np.tile(arr, reps)[:d_h].astype(np.float32)

        node_feats = np.stack([
            tile_to_dh(received),
            tile_to_dh(std_recv),
            tile_to_dh(max_recv),
            tile_to_dh(sent),
        ], axis=1)  # [d_h, 4]

        # edges: flatten upper triangle of attention matrix, pick top-k
        rows, cols = np.triu_indices(Ls, k=1)
        weights = mat[rows, cols]
        top_idx = np.argsort(-weights)[:top_k]

        edge_list, edge_w = [], []
        for idx in top_idx:
            r, c = int(rows[idx]), int(cols[idx])
            w = float(weights[idx])
            # map token positions to node indices (tile same as features)
            ri = r % d_h
            ci = c % d_h
            if ri != ci:
                edge_list += [[ri, ci], [ci, ri]]
                edge_w += [w, w]

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.int64).t().contiguous()
            edge_attr  = torch.tensor(edge_w, dtype=torch.float32).view(-1, 1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.int64)
            edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

        orig = int(attn_idx[i])
        graphs.append(Data(
            x=torch.from_numpy(node_feats),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([float(is_correct[orig])], dtype=torch.float32),
            idx=torch.tensor([orig], dtype=torch.int64),
            split=torch.tensor([int(split[orig])], dtype=torch.int8),
        ))
    return graphs


# ------------------------------------------------------------------
# Simple GNN for attention graphs
# ------------------------------------------------------------------

class AttentionGNN(nn.Module):
    def __init__(self, in_dim=4, hidden=32, heads=2):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, edge_dim=1, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, edge_dim=1, concat=False)
        self.head  = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = F.elu(self.conv2(h, edge_index, edge_attr))
        h = global_mean_pool(h, batch)
        return self.head(h).view(-1)


def train_eval_gnn(graphs, epochs=40, seed=0):
    torch.manual_seed(seed)
    train_g = [g for g in graphs if int(g.split) == schemas.SPLIT_TRAIN]
    test_g  = [g for g in graphs if int(g.split) == schemas.SPLIT_TEST]
    if len(train_g) < 5 or len(test_g) < 5:
        return None

    y_train = np.array([float(g.y) for g in train_g])
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos])

    model = AttentionGNN()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(train_g, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss_fn(logits, batch.y.view(-1)).backward()
            opt.step()

    model.eval()
    test_loader = DataLoader(test_g, batch_size=64, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            scores.append(torch.sigmoid(logits).numpy())
            labels.append(batch.y.view(-1).numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    return roc_auc_score(labels, scores)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    hs = load_artifact("artifacts/hidden_states.pt")
    graphs_art = load_artifact("artifacts/graphs.pt")
    scores_art = load_artifact("artifacts/scores.pt")

    enc = hs["enc_self_attn"].numpy()
    attn_idx = hs["attn_idx"].numpy()
    is_correct = hs["is_correct"].numpy()
    split = hs["split"].numpy()
    confidence = hs["confidence"].numpy()

    n_sub = len(attn_idx)
    sub_split   = split[attn_idx]
    sub_correct = is_correct[attn_idx]
    sub_conf    = confidence[attn_idx]
    train_mask  = sub_split == schemas.SPLIT_TRAIN
    test_mask   = sub_split == schemas.SPLIT_TEST

    print(f"Attention subsample: {n_sub} samples "
          f"({train_mask.sum()} train, {test_mask.sum()} test)")

    results = {}

    # 1. Softmax confidence baseline (on subsample)
    auc = roc_auc_score(sub_correct[test_mask], sub_conf[test_mask])
    results["Softmax confidence\n(baseline)"] = auc
    print(f"Softmax confidence baseline (subsample): {auc:.3f}")

    # 2. LogReg on flat attention weights
    flat = enc.reshape(n_sub, -1)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(flat[train_mask], sub_correct[train_mask])
    auc = roc_auc_score(sub_correct[test_mask],
                        clf.predict_proba(flat[test_mask])[:, 1])
    results["LogReg on\nflat attention"] = auc
    print(f"LogReg on flat attention: {auc:.3f}")

    # 3. LogReg on flat synolitic graph features (subsample only)
    graphs_all = graphs_art["graphs"]
    idx_to_graph = {int(g.idx): g for g in graphs_all}
    sub_graphs = [idx_to_graph[i] for i in attn_idx if i in idx_to_graph]
    if len(sub_graphs) == n_sub:
        flat_graph = np.stack([g.x.numpy().flatten() for g in sub_graphs])
        sub_split2 = np.array([int(g.split) for g in sub_graphs])
        sub_label2 = np.array([float(g.y) for g in sub_graphs])
        tm2 = sub_split2 == schemas.SPLIT_TRAIN
        tt2 = sub_split2 == schemas.SPLIT_TEST
        clf2 = LogisticRegression(max_iter=1000)
        clf2.fit(flat_graph[tm2], sub_label2[tm2])
        auc = roc_auc_score(sub_label2[tt2],
                            clf2.predict_proba(flat_graph[tt2])[:, 1])
        results["LogReg on\nsynolitic features"] = auc
        print(f"LogReg on synolitic flat features (subsample): {auc:.3f}")

    # 4. GNN on attention-as-graph
    print("Training GNN on attention-based graphs...")
    attn_graphs = build_attention_graphs(hs, graphs_art)
    auc = train_eval_gnn(attn_graphs, epochs=40)
    if auc is not None:
        results["GNN on\nattention graphs"] = auc
        print(f"GNN on attention-as-graph: {auc:.3f}")

    # 5. Synolitic GNN (full dataset AUC from scores.pt)
    sc = scores_art["scores"].numpy()
    ic = scores_art["is_correct"].numpy()
    sp = scores_art["split"].numpy()
    tt = sp == schemas.SPLIT_TEST
    auc = roc_auc_score(ic[tt], sc[tt])
    results["Synolitic GNN\n(full dataset)"] = auc
    print(f"Synolitic GNN (full N=10000): {auc:.3f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    labels = list(results.keys())
    aucs   = list(results.values())
    colors = []
    for l in labels:
        if "Synolitic" in l and "GNN" in l:
            colors.append("#E65100")
        elif "synolitic" in l:
            colors.append("#2E7D32")
        elif "baseline" in l.lower() or "confidence" in l.lower():
            colors.append("#9E9E9E")
        else:
            colors.append("#BDBDBD")

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(labels, aucs, color=colors, height=0.55)
    ax.axvline(results.get("Softmax confidence\n(baseline)", 0.799),
               color="#616161", linestyle="--", linewidth=1.2, alpha=0.7)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Test ROC-AUC", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "H1 test: synolitic graphs vs raw attention (attention subsample)\n"
        "If synolitic > attention-based: topology captures signal invisible in raw attention",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = "figures/h1_attention_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}")

    print("\nSummary:")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {k.replace(chr(10), ' ')}: {v:.3f}")


if __name__ == "__main__":
    main()
