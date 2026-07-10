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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import roc_auc_score
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')

from synolitic.common import schemas
from synolitic.common.io import load_artifact, save_artifact


class FocalLoss(nn.Module):
    """
    For imbalanced classification.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GNNConfidence(nn.Module):
    """
    GNN with confidence as node feature.
    """
    def __init__(self, 
                 in_channels=6,
                 hidden=16,
                 out=1,
                 dropout=0.3,
                 use_edge_attr=True):
        super().__init__()
        
        self.use_edge_attr = use_edge_attr
        self.num_nodes = 64
        
        if use_edge_attr:
            self.conv1 = GATv2Conv(in_channels, hidden, heads=4, concat=False, dropout=dropout, edge_dim=1)
            self.conv2 = GATv2Conv(hidden, hidden, heads=2, concat=False, dropout=dropout, edge_dim=1)
        else:
            self.conv1 = GATv2Conv(in_channels, hidden, heads=4, concat=False, dropout=dropout)
            self.conv2 = GATv2Conv(hidden, hidden, heads=2, concat=False, dropout=dropout)
        
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        
        flat_dim = self.num_nodes * hidden
        self.proj = nn.Linear(flat_dim, 64)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, out)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            batch_size = batch.max().item() + 1
        else:
            batch_size = 1
        
        if self.use_edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = (data.edge_attr - 0.5) * 2
        else:
            edge_attr = None
        
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        
        if self.use_edge_attr and edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_edge_attr and edge_attr is not None:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_flat = x.view(batch_size, -1)
        
        x = F.elu(self.proj(x_flat))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lin2(x).squeeze(-1)
        
        return logits


def add_confidence_to_nodes(graphs, hidden_states_artifact):
    """
    Add confidence as 6th node feature.
    """
    confidence = hidden_states_artifact.get("confidence", None)
    if confidence is None:
        raise ValueError("confidence not found in hidden_states_artifact")
    
    if torch.is_tensor(confidence):
        confidence_np = confidence.numpy()
    else:
        confidence_np = np.array(confidence)
    
    print(f"Confidence: mean={confidence_np.mean():.3f}, std={confidence_np.std():.3f}")
    
    for g in graphs:
        idx = g.idx.item()
        original_x = g.x.numpy()
        confidence_col = np.full((64, 1), confidence_np[idx], dtype=np.float32)
        new_x = np.concatenate([original_x, confidence_col], axis=1)
        g.x = torch.from_numpy(new_x)
    
    return graphs


def split_train_for_early_stop(train_graphs, val_ratio=0.1, seed=42):
    """
    Split training data into train and validation subsets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_train = len(train_graphs)
    indices = np.random.permutation(n_train)
    n_val = int(n_train * val_ratio)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_subset = [train_graphs[i] for i in train_indices]
    val_subset = [train_graphs[i] for i in val_indices]
    
    print(f"  Train subset: {len(train_subset)}, Val subset: {len(val_subset)}")
    
    return train_subset, val_subset


def train_gnn(
    graphs_artifact: dict,
    hidden_states_artifact: dict = None,
    num_epochs: int = 300,
    batch_size: int = 512,
    lr: float = 0.001,
    hidden: int = 16,
    dropout: float = 0.3,
    weight_decay: float = 1e-3,
    use_edge_attr: bool = True,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    patience: int = 25,
    n_ensemble: int = 5,
    val_ratio: float = 0.1,
    device=None,
) -> dict:
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    graphs = graphs_artifact["graphs"]
    print(f"Total graphs: {len(graphs)}")
    
    graphs = sorted(graphs, key=lambda g: g.idx.item())
    
    graphs = add_confidence_to_nodes(graphs, hidden_states_artifact)
    
    all_splits = np.array([g.split.item() for g in graphs], dtype=np.int8)
    all_labels = np.array([g.y.item() for g in graphs], dtype=np.int8)
    
    train_graphs = [g for g in graphs if g.split == schemas.SPLIT_TRAIN]
    cal_graphs = [g for g in graphs if g.split == schemas.SPLIT_CAL]
    test_graphs = [g for g in graphs if g.split == schemas.SPLIT_TEST]
    
    print(f"Train: {len(train_graphs)}, Cal: {len(cal_graphs)}, Test: {len(test_graphs)}")
    print(f"Node features: {train_graphs[0].x.shape[1]} (including confidence)")
    
    train_subset, val_subset = split_train_for_early_stop(train_graphs, val_ratio=val_ratio)
    
    train_labels = [g.y.item() for g in train_subset]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"Class balance: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight.item():.2f}")
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    cal_loader = DataLoader(cal_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    print(f"\nTraining ensemble of {n_ensemble} models")
    print("Model\tBest_Val_AUC\tCal_AUC\tTest_AUC")
    
    all_ensemble_preds = []
    best_val_aucs = []
    cal_aucs_final = []
    test_aucs_list = []
    
    for ensemble_idx in range(n_ensemble):
        seed = 42 + ensemble_idx * 13
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = GNNConfidence(
            in_channels=6,
            hidden=hidden,
            dropout=dropout,
            use_edge_attr=use_edge_attr
        ).to(device)
        
        if ensemble_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model params: {total_params:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )
        
        if use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        best_val_auc = 0.0
        best_state = None
        wait = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model(batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_probs.extend(probs)
                    val_labels.extend(batch.y.cpu().numpy())
            
            val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
            scheduler.step()
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            
            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1:3d}: loss={total_loss/len(train_loader):.4f}, "
                      f"val_auc={val_auc:.4f}, best={best_val_auc:.4f}, lr={current_lr:.6f}")
            
            if wait >= patience:
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Cal
        model.eval()
        cal_probs, cal_labels = [], []
        with torch.no_grad():
            for batch in cal_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                cal_probs.extend(probs)
                cal_labels.extend(batch.y.cpu().numpy())
        cal_auc = roc_auc_score(cal_labels, cal_probs) if len(set(cal_labels)) > 1 else 0.5
        
        # Test
        test_probs, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_probs.extend(probs)
                test_labels.extend(batch.y.cpu().numpy())
        test_auc = roc_auc_score(test_labels, test_probs)
        
        print(f"  {ensemble_idx+1}\t{best_val_auc:.4f}\t\t{cal_auc:.4f}\t{test_auc:.4f}")
        
        best_val_aucs.append(best_val_auc)
        cal_aucs_final.append(cal_auc)
        test_aucs_list.append(test_auc)
        
        model_preds = np.zeros(len(graphs), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            all_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            start_idx = 0
            for batch in all_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                batch_size_actual = len(probs)
                model_preds[start_idx:start_idx + batch_size_actual] = probs
                start_idx += batch_size_actual
        
        all_ensemble_preds.append(model_preds)
    
    ensemble_preds = np.mean(all_ensemble_preds, axis=0)
    
    assert len(ensemble_preds) == len(all_labels), f"Shape mismatch: {len(ensemble_preds)} vs {len(all_labels)}"
    
    test_mask = all_splits == schemas.SPLIT_TEST
    ensemble_auc = roc_auc_score(all_labels[test_mask], ensemble_preds[test_mask])
    print(f"\nEnsemble Test ROC-AUC: {ensemble_auc:.4f}")
    
    scores_artifact = {
        "scores": torch.from_numpy(ensemble_preds).float(),
        "is_correct": torch.from_numpy(all_labels).to(torch.int8),
        "split": torch.from_numpy(all_splits).to(torch.int8),
        "meta": {
            "schema_version": schemas.SCHEMA_VERSION,
            "ensemble_auc": float(ensemble_auc),
            "test_aucs": [float(x) for x in test_aucs_list],
            "best_val_aucs": [float(x) for x in best_val_aucs],
            "cal_aucs_final": [float(x) for x in cal_aucs_final],
            "arch": "gatv2_confidence_as_node_feature_improved",
            "use_edge_attr": use_edge_attr,
            "use_focal_loss": use_focal_loss,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "hidden": hidden,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_ensemble": n_ensemble,
            "pos_weight": float(pos_weight.item()),
            "n_train": len(train_subset),
            "n_val": len(val_subset),
            "n_cal": len(cal_graphs),
            "n_test": len(test_graphs),
            "epochs": num_epochs,
            "val_ratio": val_ratio,
            "source": "gnn_improved_ensemble_5",
            "trained_on": "train_only",
        },
    }
    
    schemas.assert_valid("scores", schemas.validate_scores(scores_artifact))
    return scores_artifact


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default="artifacts/graphs.pt")
    parser.add_argument("--hidden-states", default="artifacts/hidden_states.pt")
    parser.add_argument("--out", default="artifacts/scores.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--no-edge-attr", action='store_true')
    parser.add_argument("--no-focal", action='store_true')
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    print(f"Use edge_attr: {not args.no_edge_attr}")
    print(f"Use focal loss: {not args.no_focal}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Hidden: {args.hidden}")
    print(f"Batch size: {args.batch}")
    
    graphs_artifact = load_artifact(args.graphs)
    schemas.assert_valid("graphs", schemas.validate_graphs(graphs_artifact))
    
    hidden_states_artifact = load_artifact(args.hidden_states)
    schemas.assert_valid("hidden_states", schemas.validate_hidden_states(hidden_states_artifact))
    
    scores = train_gnn(
        graphs_artifact=graphs_artifact,
        hidden_states_artifact=hidden_states_artifact,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        hidden=args.hidden,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        use_edge_attr=not args.no_edge_attr,
        use_focal_loss=not args.no_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        patience=args.patience,
        n_ensemble=args.ensemble,
        val_ratio=args.val_ratio,
    )
    
    save_artifact(scores, args.out)
    print(f"\n Saved scores to: {args.out}")
    print(f"   Ensemble AUC: {scores['meta']['ensemble_auc']:.4f}")