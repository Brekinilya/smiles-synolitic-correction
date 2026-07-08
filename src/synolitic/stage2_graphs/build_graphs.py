"""Stage 2 interface: hidden_states.pt -> graphs.pt.

Owner: role 2. Method per the proposal (Zaikin et al., Technologies 2026):
1. Fit C(64, 2) = 2016 pairwise binary classifiers, one per feature pair
   (x_i, x_j), separating is_correct == 1 vs 0. Fit ONLY on rows with
   ``split == SPLIT_TRAIN``; score all rows.
2. For every sample build a weighted graph on 64 nodes: edge (i, j) carries
   that sample's classifier score for the pair (i, j). Sparsify (threshold
   or top-k) while keeping the graph connected enough for centralities.
3. Node features [value, degree, strength, closeness, betweenness]
   (see schemas.N_NODE_FEATURES; networkx provides the centralities).
4. Emit ``{"graphs": list[torch_geometric.data.Data], "meta": {...}}`` and
   validate with ``schemas.validate_graphs`` before saving.

Develop against the dummy artifact:
    uv run python scripts/make_dummy_data.py
    hs = load_artifact("artifacts/dummy/hidden_states.pt")

Implementation notes (role 2, this fill-in):
- Sparsification keeps, per sample, the `sparsify_top_k` edges whose score
  is farthest from 0.5 -- i.e. the pairwise classifiers most *decisive* at
  that sample's own point. Which edges get kept depends only on the fixed,
  already-trained classifiers' behavior on the sample's own features, not
  on its label, so this cannot leak is_correct into the topology.
- Distances for closeness/betweenness are derived from edge weight so a
  highly confident edge (weight near 0 or 1) counts as "close".
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data

from synolitic.common import schemas
from synolitic.common.io import load_artifact, save_artifact


@dataclass
class SynoliticEnsemble:
    """Fitted ensemble of C(d, 2) pairwise classifiers, one per feature pair."""
    pairs: list[tuple[int, int]]
    W: np.ndarray  # [n_pairs, 2] logistic-regression coefficients
    b: np.ndarray  # [n_pairs] intercepts
    d: int

    def score_all(self, X: np.ndarray) -> np.ndarray:
        """[N, d] -> [N, n_pairs]: sigmoid(w_i*x_i + w_j*x_j + b) for every
        pair's fixed classifier, each row scored at its own (x_i, x_j)."""
        idx_i = np.array([p[0] for p in self.pairs])
        idx_j = np.array([p[1] for p in self.pairs])
        logits = X[:, idx_i] * self.W[:, 0] + X[:, idx_j] * self.W[:, 1] + self.b
        return 1.0 / (1.0 + np.exp(-logits))


def fit_pairwise_ensemble(x_train: np.ndarray, y_train: np.ndarray) -> SynoliticEnsemble:
    """Fit the C(64, 2) = 2016 pairwise classifiers.

    Args:
        x_train: [N_train, 64] float32 -- rows of X with split == SPLIT_TRAIN.
        y_train: [N_train] int8 is_correct labels.
    Returns:
        Fitted ensemble able to score any [N, 64] matrix pair-wise.
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "x_train/y_train has only one class present -- cannot fit "
            "pairwise classifiers. Check the split==SPLIT_TRAIN rows."
        )
    d = x_train.shape[1]
    pairs = list(itertools.combinations(range(d), 2))
    W = np.zeros((len(pairs), 2), dtype=np.float32)
    b = np.zeros(len(pairs), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train[:, [i, j]], y_train)
        W[k] = clf.coef_[0]
        b[k] = clf.intercept_[0]
    return SynoliticEnsemble(pairs=pairs, W=W, b=b, d=d)


def _build_sample_graph(x_vec: np.ndarray, pairs, scores_row: np.ndarray, d: int, top_k: int):
    """One sample's sparsified graph + its [d, N_NODE_FEATURES] node features."""
    confidence = np.abs(scores_row - 0.5)
    top_idx = np.argsort(-confidence)[:top_k]

    G = nx.Graph()
    G.add_nodes_from(range(d))
    for k in top_idx:
        i, j = pairs[k]
        w = float(scores_row[k])
        dist = (1.0 - w) if w >= 0.5 else w
        G.add_edge(i, j, weight=w, distance=max(dist, 1e-6))

    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    if G.number_of_edges() > 0:
        closeness = nx.closeness_centrality(G, distance="distance")
        betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    else:
        closeness = {node: 0.0 for node in G.nodes()}
        betweenness = {node: 0.0 for node in G.nodes()}

    feats = np.zeros((d, schemas.N_NODE_FEATURES), dtype=np.float32)
    for node in range(d):
        feats[node, 0] = x_vec[node]
        feats[node, 1] = degree.get(node, 0) / max(top_k, 1)
        feats[node, 2] = strength.get(node, 0.0)
        feats[node, 3] = closeness.get(node, 0.0)
        feats[node, 4] = betweenness.get(node, 0.0)

    edge_index_list, edge_attr_list = [], []
    for u, v, data in G.edges(data=True):
        edge_index_list += [[u, v], [v, u]]
        edge_attr_list += [data["weight"], data["weight"]]

    edge_index = torch.tensor(edge_index_list, dtype=torch.int64).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    return feats, edge_index, edge_attr


def build_graphs(hidden_states: dict, ensemble: SynoliticEnsemble, sparsify_top_k: int | None = None) -> dict:
    """Build one synolitic graph per sample from the hidden-states artifact."""
    X = np.asarray(hidden_states["X"], dtype=np.float64)
    is_correct = np.asarray(hidden_states["is_correct"])
    split = np.asarray(hidden_states["split"])
    N, d = X.shape
    n_pairs_total = len(ensemble.pairs)
    top_k = sparsify_top_k or max(1, int(0.125 * n_pairs_total))

    scores = ensemble.score_all(X)
    X32 = X.astype(np.float32)

    graphs = []
    for n in range(N):
        feats, edge_index, edge_attr = _build_sample_graph(
            X32[n], ensemble.pairs, scores[n], d, top_k
        )
        graphs.append(Data(
            x=torch.from_numpy(feats),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([float(is_correct[n])], dtype=torch.float32),
            idx=torch.tensor([n], dtype=torch.int64),
            split=torch.tensor([int(split[n])], dtype=torch.int8),
        ))

    artifact = {
        "graphs": graphs,
        "meta": {
            "n_graphs": N,
            "d": d,
            "n_pairs_total": n_pairs_total,
            "sparsify_top_k": top_k,
        },
    }
    schemas.assert_valid("graphs", schemas.validate_graphs(artifact))
    return artifact


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states", default="artifacts/dummy/hidden_states.pt")
    parser.add_argument("--out", default="artifacts/dummy/graphs.pt")
    parser.add_argument("--sparsify-top-k", type=int, default=None)
    args = parser.parse_args()

    hs = load_artifact(args.hidden_states)
    schemas.assert_valid("hidden_states", schemas.validate_hidden_states(hs))

    split = np.asarray(hs["split"])
    train_mask = split == schemas.SPLIT_TRAIN
    x_train = np.asarray(hs["X"], dtype=np.float64)[train_mask]
    y_train = np.asarray(hs["is_correct"])[train_mask]

    ensemble = fit_pairwise_ensemble(x_train, y_train)
    artifact = build_graphs(hs, ensemble, sparsify_top_k=args.sparsify_top_k)
    save_artifact(artifact, args.out)
    print(f"wrote {args.out}: {artifact['meta']}")
