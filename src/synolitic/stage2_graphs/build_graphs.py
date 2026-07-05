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
"""

from __future__ import annotations


def fit_pairwise_ensemble(x_train, y_train):
    """Fit the 2016 pairwise classifiers (e.g. sklearn LogisticRegression).

    Args:
        x_train: [N_train, 64] float32 — rows of X with split == SPLIT_TRAIN.
        y_train: [N_train] int8 is_correct labels.

    Returns:
        Fitted ensemble able to score any [N, 64] matrix pair-wise.
    """
    raise NotImplementedError("role 2")


def build_graphs(hidden_states: dict, ensemble, sparsify_top_k: int | None = None) -> dict:
    """Build one synolitic graph per sample from the hidden-states artifact."""
    raise NotImplementedError("role 2")
