"""Stage 4 Corrector.

Fisher discriminant + Theorem 1 guarantees + confidence baseline.

Usage:
    uv run python src/synolitic/stage4_corrector/corrector.py \
        --scores artifacts/dummy/scores.pt \
        --graphs artifacts/dummy/graphs.pt \
        --hidden-states artifacts/dummy/hidden_states.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast
from numpy.typing import NDArray
import numpy as np
from synolitic.common.io import load_artifact
from synolitic.common.schemas import SPLIT_CAL, SPLIT_TEST
from sklearn.metrics import roc_auc_score

def fisher_projector(phi_correct, phi_incorrect):
    x_pos = np.asarray(phi_correct, dtype=float)
    x_neg = np.asarray(phi_incorrect, dtype=float)

    mu_pos = x_pos.mean(axis=0)
    mu_neg = x_neg.mean(axis=0)

    s_pos = np.atleast_2d(np.cov(x_pos, rowvar=False))
    s_neg = np.atleast_2d(np.cov(x_neg, rowvar=False))

    sw = s_pos + s_neg
    sw += 1e-6 * np.eye(sw.shape[0])
    w = np.linalg.solve(sw,mu_pos - mu_neg,)
    n = np.linalg.norm(w)

    if n > 0:
        w /= n

    return w


def project(phi, w):
    return np.asarray(phi) @ np.asarray(w)


def calibrate_threshold(
    projected,
    labels,
):

    z = np.asarray(projected)
    y = np.asarray(labels, dtype=bool)

    best_delta = z[0]
    best = -1

    for delta in np.unique(z):

        accept = z >= delta

        tpr = np.mean(
            accept[y]
        ) if np.any(y) else 0

        tnr = np.mean(
            ~accept[~y]
        ) if np.any(~y) else 0


        score = tpr + tnr

        if score > best:
            best = score
            best_delta = delta

    return float(best_delta)

def evaluate(scores,labels,split,delta):
    mask = split == SPLIT_TEST

    z = scores[mask]
    y = labels[mask].astype(bool)

    accept = z >= delta
    reject = ~accept

    correct = y
    incorrect = ~y

    return {
        "n_test": int(mask.sum()),
        "acceptance_rate":
            float(np.mean(accept)),
        "rejection_rate":
            float(np.mean(reject)),
        "p_accept_given_correct":
            float(np.mean(accept[correct]))
            if np.any(correct)
            else np.nan,
        "p_reject_given_incorrect":
            float(np.mean(reject[incorrect]))
            if np.any(incorrect)
            else np.nan,
        "precision_accept":
            float(np.mean(y[accept]))
            if np.any(accept)
            else np.nan,
        "precision_reject":
            float(np.mean(~y[reject]))
            if np.any(reject)
            else np.nan,
        "accepted_count":
            int(np.sum(accept)),
        "rejected_count":
            int(np.sum(reject)),
        "correct_total":
            int(np.sum(correct)),
        "incorrect_total":
            int(np.sum(incorrect)),
    }

def rho(a, d):
    eps = np.linspace(1e-8,1,10000)
    v = (np.maximum(0, a-eps)*(1-2*np.exp(-2*d*eps**2)))
    return float(v.max())

def psi(a, d):
    eps = np.linspace(1e-8,1,10000,)
    v = np.minimum(1,2*np.exp(-2*d*eps**2)+ a + eps)
    return float(v.min())

def empirical_cdf(x, t):
    return float(np.mean(np.asarray(x)<=t))

def empirical_quantile(x,q):
    x=np.sort(np.asarray(x))
    k=int(np.ceil(q*len(x)))-1
    k=max(0,min(k,len(x)-1))

    return float(x[k])

def theoretical_bounds(theta,correct,incorrect):
    delta_cdf = empirical_cdf(incorrect, theta)
    a = empirical_cdf(correct,theta)

    return {
        "accept_lb":1-psi(a,len(correct)),
        "reject_lb":rho(delta_cdf,len(incorrect)),
        "m_plus":len(correct),
        "m_minus":len(incorrect),
        "theta": theta,
        "delta": delta_cdf,
        "a_plus":a
    }

def graphs_to_features(graph_artifact, expected_n: int) -> NDArray[np.float64]:
    graphs = graph_artifact["graphs"]
    feats = np.full((expected_n, 15), np.nan) 
    
    for g in graphs:
        idx = int(g.idx.item()) 
        x = g.x.detach().cpu().numpy()
        feats[idx] = np.concatenate([x.mean(axis=0), x.std(axis=0), x.max(axis=0)])
        
    if np.isnan(feats).any():
        raise ValueError("Не все индексы из scores.pt покрыты графами!")
    return feats

def run_confidence_baseline(hidden):
    confidence = hidden["confidence"]

    correct = hidden["is_correct"]

    split = hidden["split"]

    if hasattr(confidence,"numpy"):
        confidence=confidence.numpy()

    if hasattr(correct,"numpy"):
        correct=correct.numpy()

    if hasattr(split,"numpy"):
        split=split.numpy()

    cal = split==SPLIT_CAL

    delta = calibrate_threshold(confidence[cal],correct[cal])

    return evaluate(confidence,correct,split,delta)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--graphs", type=Path, required=True)
    p.add_argument("--hidden-states", type=Path, required=True)
    args = p.parse_args()

    scores_art = cast(dict[str, Any], load_artifact(args.scores))
    scores = np.asarray(scores_art["scores"], dtype=float)
    labels = np.asarray(scores_art["is_correct"], dtype=bool)
    split = np.asarray(scores_art["split"])

    cal = split == SPLIT_CAL
    test = split == SPLIT_TEST

    print("\n" + "=" * 80)
    print(" " * 20 + "STAGE 4: SYNONLITIC ERROR CORRECTOR")
    print("=" * 80)

    print("\n[DATASET STATISTICS]")
    print(f"  Total samples        : {len(scores)}")
    print(f"  Calibration (split=1): {cal.sum():<5} (correct: {(cal & labels).sum()}, incorrect: {(cal & ~labels).sum()})")
    print(f"  Test (split=2)       : {test.sum():<5} (correct: {(test & labels).sum()}, incorrect: {(test & ~labels).sum()})")

    def print_method_block(name: str, auc_cal: float, auc_test: float, delta: float, 
                           feature_dim: int | None, eval_dict: dict, bounds_dict: dict | None = None):
        print("\n" + "-" * 80)
        print(f"[{name}]")
        print("-" * 80)
        if feature_dim is not None:
            print(f"  Feature dimension    : {feature_dim}")
        print(f"  Threshold Δ          : {delta:.6f}")
        print(f"  AUC (calibration)    : {auc_cal:.3f}")
        print(f"  AUC (test)           : {auc_test:.3f}")
        print(f"  Acceptance rate      : {eval_dict['acceptance_rate']:.3f}")
        print(f"  P(accept | correct)  : {eval_dict['p_accept_given_correct']:.3f}")
        print(f"  P(reject | error)    : {eval_dict['p_reject_given_incorrect']:.3f}")
        
        if bounds_dict is not None:
            print(f"\n  [Theoretical bounds (Tyukin Th.1)]")
            print(f"    Accept lower bound : {bounds_dict['accept_lb']:.3f}")
            print(f"    Reject lower bound : {bounds_dict['reject_lb']:.3f}")

    delta_gnn = calibrate_threshold(scores[cal], labels[cal])
    result_gnn = evaluate(scores, labels, split, delta_gnn)
    auc_gnn_cal = roc_auc_score(labels[cal], scores[cal])
    auc_gnn_test = roc_auc_score(labels[test], scores[test])

    print_method_block(
        name="1. GNN SCORE (baseline from Stage 3)",
        auc_cal=auc_gnn_cal,
        auc_test=auc_gnn_test,
        delta=delta_gnn,
        feature_dim=None,
        eval_dict=result_gnn
    )

    graph_art = load_artifact(args.graphs)
    phi = graphs_to_features(graph_art, expected_n=len(scores))

    w = fisher_projector(phi[cal & labels], phi[cal & (~labels)])
    z = project(phi, w)

    fisher_delta = calibrate_threshold(z[cal], labels[cal])
    fisher_eval = evaluate(z, labels, split, fisher_delta)
    bounds = theoretical_bounds(fisher_delta, z[cal & labels], z[cal & (~labels)])
    fisher_auc_cal = roc_auc_score(labels[cal], z[cal])
    fisher_auc_test = roc_auc_score(labels[test], z[test])

    print_method_block(
        name="2. FISHER ON SYNONLITIC GRAPH FEATURES (paper-faithful Φ)",
        auc_cal=fisher_auc_cal,
        auc_test=fisher_auc_test,
        delta=fisher_delta,
        feature_dim=phi.shape[-1],
        eval_dict=fisher_eval,
        bounds_dict=bounds
    )

    hidden = load_artifact(args.hidden_states)
    baseline = run_confidence_baseline(hidden)
    
    confidence = hidden["confidence"] #type: ignore
    if hasattr(confidence, "numpy"):
        confidence = confidence.numpy()
    confidence_auc_cal = roc_auc_score(labels[cal], confidence[cal])
    confidence_auc_test = roc_auc_score(labels[test], confidence[test])

    delta_conf = calibrate_threshold(confidence[cal], labels[cal])

    print_method_block(
        name="3. CONFIDENCE BASELINE (softmax rejection from Stage 1)",
        auc_cal=confidence_auc_cal,
        auc_test=confidence_auc_test,
        delta=delta_conf,
        feature_dim=None,
        eval_dict=baseline
    )

    X_raw = hidden["X"] #type: ignore
    if hasattr(X_raw, "numpy"):
        X_raw = X_raw.numpy()
    X_raw = np.asarray(X_raw, dtype=float)

    w_raw = fisher_projector(X_raw[cal & labels], X_raw[cal & (~labels)])
    z_raw = project(X_raw, w_raw)

    raw_delta = calibrate_threshold(z_raw[cal], labels[cal])
    raw_eval = evaluate(z_raw, labels, split, raw_delta)
    raw_auc_cal = roc_auc_score(labels[cal], z_raw[cal])
    raw_auc_test = roc_auc_score(labels[test], z_raw[test])

    print_method_block(
        name="4. FISHER ON RAW FEATURES X (H1 baseline)",
        auc_cal=raw_auc_cal,
        auc_test=raw_auc_test,
        delta=raw_delta,
        feature_dim=X_raw.shape[-1],
        eval_dict=raw_eval
    )

    print("\n" + "=" * 80)
    print(" " * 28 + "STAGE 4 SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Method':<26}{'AUC':<9}{'P(acc|corr)':<15}{'P(rej|incorr)':<16}{'Δ':<10}"
    print(header)
    print("-" * 80)
    
    print(f"{'GNN score':<26}{auc_gnn_test:<9.3f}{result_gnn['p_accept_given_correct']:<15.3f}"
          f"{result_gnn['p_reject_given_incorrect']:<16.3f}{delta_gnn:<10.3f}")
    
    print(f"{'Confidence':<26}{confidence_auc_test:<9.3f}{baseline['p_accept_given_correct']:<15.3f}"
          f"{baseline['p_reject_given_incorrect']:<16.3f}{delta_conf:<10.3f}")
    
    print(f"{'Fisher Raw X (H1)':<26}{raw_auc_test:<9.3f}{raw_eval['p_accept_given_correct']:<15.3f}"
          f"{raw_eval['p_reject_given_incorrect']:<16.3f}{raw_delta:<10.3f}")
    
    print(f"{'Fisher Synolitic Φ':<26}{fisher_auc_test:<9.3f}{fisher_eval['p_accept_given_correct']:<15.3f}"
          f"{fisher_eval['p_reject_given_incorrect']:<16.3f}{fisher_delta:<10.3f}")

    print("\n" + "-" * 80)
    print("[HYPOTHESIS H1 CHECK]")
    print("-" * 80)
    print(f"  H1: Synolitic graph features should outperform raw features X")
    print(f"  Fisher Synolitic Φ AUC : {fisher_auc_test:.3f}")
    print(f"  Fisher Raw X AUC       : {raw_auc_test:.3f}")
    
    if fisher_auc_test >= raw_auc_test:
        print(f"  ✅ H1 is SUPPORTED: Synolitic features AUC ({fisher_auc_test:.3f}) >= Raw features AUC ({raw_auc_test:.3f})")
    else:
        print(f"  ❌ H1 is NOT SUPPORTED: Synolitic features AUC ({fisher_auc_test:.3f}) < Raw features AUC ({raw_auc_test:.3f})")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()