"""Stage 3 interface: graphs.pt -> scores.pt.

Owner: role 3. Train a GCN / GATv2 (torch_geometric) to predict ``y``
(is_correct) from the synolitic graphs.

Split discipline: train ONLY on graphs with ``split == SPLIT_TRAIN``;
produce scores for ALL graphs (cal and test rows are consumed downstream
by stage 4). Classes are imbalanced (~4:1 at 80% model accuracy) — use
class weights or balanced sampling; report ROC-AUC, not accuracy.

Baseline to include for the defense: logistic regression / MLP on raw
``hidden_states["X"]`` — the comparison "graph vs raw features" is
hypothesis H1 of the proposal.

Output: ``{"scores": [N] float32 = P(is_correct=1), "is_correct": [N] int8,
"split": [N] int8, "meta": dict}``, ordered by ``idx`` so that row i aligns
with row i of hidden_states.pt. Validate with ``schemas.validate_scores``.

Develop against the dummy artifact:
    uv run python scripts/make_dummy_data.py
    graphs = load_artifact("artifacts/dummy/graphs.pt")
"""

from __future__ import annotations


def train_gnn(graphs_artifact: dict, arch: str = "gatv2") -> dict:
    """Train the GNN and return the scores artifact (see module docstring)."""
    raise NotImplementedError("role 3")
