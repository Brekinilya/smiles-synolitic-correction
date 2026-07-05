"""Shape- and semantics-faithful dummy artifacts for parallel development.

Stages 2-4 can be written and tested against these fakes; swapping in the
real files requires no code changes because both go through the same
validators in :mod:`synolitic.common.schemas`.

The fakes are internally consistent: ``is_correct`` really equals the exact
match of ``pred_tokens`` vs ``target_tokens``, ``X`` carries a weak class
signal, ``confidence`` is higher for correct samples, and graph/score labels
are aligned with the hidden-states artifact they were derived from.
"""

from __future__ import annotations

import torch

from synolitic.common.schemas import (
    BOS,
    D_H,
    EOS,
    FIRST_DIGIT,
    N_DIGITS,
    N_NODE_FEATURES,
    PAD,
    SCHEMA_VERSION,
    make_split,
)


def dummy_hidden_states(
    n: int = 512,
    attn_subsample: int = 64,
    min_len: int = 10,
    max_len: int = 20,
    n_layers: int = 2,
    n_heads: int = 4,
    accuracy: float = 0.8,
    seed: int = 0,
) -> dict:
    from synolitic.stage1_model.dataset import make_dataset

    g = torch.Generator().manual_seed(seed)
    src, tgt, lengths = make_dataset(n, min_len, max_len, g)
    ls = src.shape[1]
    t = ls + 1

    pred = torch.full((n, t), PAD, dtype=torch.int64)
    pred[:, :ls] = tgt
    pred[torch.arange(n), lengths] = EOS

    is_correct = (torch.rand(n, generator=g) < accuracy).to(torch.int8)
    for i in (is_correct == 0).nonzero(as_tuple=True)[0].tolist():
        pos = int(torch.randint(0, int(lengths[i]), (1,), generator=g))
        orig = int(pred[i, pos])
        offset = 1 + int(torch.randint(0, N_DIGITS - 1, (1,), generator=g))
        pred[i, pos] = FIRST_DIGIT + (orig - FIRST_DIGIT + offset) % N_DIGITS

    u = torch.rand(n, generator=g)
    confidence = torch.where(is_correct.bool(), 0.70 + 0.29 * u, 0.35 + 0.45 * u)

    w = torch.randn(D_H, generator=g)
    x = torch.randn(n, D_H, generator=g) + (is_correct.float() - 0.5).unsqueeze(1) * w * 0.8
    x_layers = torch.stack([x + 0.3 * torch.randn(n, D_H, generator=g), x], dim=1)

    n_sub = min(attn_subsample, n)
    attn_idx = torch.randperm(n, generator=g)[:n_sub].sort().values

    def _attn(q: int, k: int) -> torch.Tensor:
        return torch.softmax(torch.randn(n_sub, n_layers, n_heads, q, k, generator=g), dim=-1)

    return {
        "X": x.to(torch.float32),
        "X_layers": x_layers.to(torch.float32),
        "input_tokens": src,
        "target_tokens": tgt,
        "pred_tokens": pred,
        "lengths": lengths,
        "is_correct": is_correct,
        "confidence": confidence.to(torch.float32),
        "split": make_split(n, (0.6, 0.2, 0.2), g),
        "attn_idx": attn_idx,
        "enc_self_attn": _attn(ls, ls),
        "dec_self_attn": _attn(t, t),
        "cross_attn": _attn(t, ls),
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "d_h": D_H,
            "n": n,
            "seed": seed,
            "pooling": "dummy_random",
            "task": "reversal-dummy",
            "min_len": min_len,
            "max_len": max_len,
            "created_by": "synolitic.common.dummy.dummy_hidden_states",
        },
    }


def dummy_graphs(
    hidden_states: dict | None = None,
    n: int = 256,
    n_edges: int = 256,
    seed: int = 0,
) -> dict:
    from torch_geometric.data import Data

    g = torch.Generator().manual_seed(seed)
    if hidden_states is not None:
        n = hidden_states["X"].shape[0]
        is_correct = hidden_states["is_correct"]
        split = hidden_states["split"]
    else:
        is_correct = (torch.rand(n, generator=g) < 0.8).to(torch.int8)
        split = make_split(n, (0.6, 0.2, 0.2), g)

    shift = torch.randn(N_NODE_FEATURES, generator=g) * 0.4
    graphs = []
    for i in range(n):
        y = float(is_correct[i])
        x = torch.randn(D_H, N_NODE_FEATURES, generator=g) + shift * (y - 0.5)
        a = torch.randint(0, D_H, (n_edges,), generator=g)
        b = torch.randint(0, D_H - 1, (n_edges,), generator=g)
        b = b + (b >= a).long()  # uniform over nodes != a, no self-loops
        edge_index = torch.stack([torch.cat([a, b]), torch.cat([b, a])]).long()
        edge_attr = torch.rand(2 * n_edges, generator=g).to(torch.float32)
        graphs.append(
            Data(
                x=x.to(torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float32),
                idx=torch.tensor([i], dtype=torch.int64),
                split=split[i].reshape(1),
            )
        )
    return {
        "graphs": graphs,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "n": n,
            "seed": seed,
            "created_by": "synolitic.common.dummy.dummy_graphs",
        },
    }


def dummy_scores(hidden_states: dict | None = None, n: int = 512, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    if hidden_states is not None:
        n = hidden_states["X"].shape[0]
        is_correct = hidden_states["is_correct"]
        split = hidden_states["split"]
    else:
        is_correct = (torch.rand(n, generator=g) < 0.8).to(torch.int8)
        split = make_split(n, (0.6, 0.2, 0.2), g)
    noise = torch.randn(n, generator=g)
    scores = torch.sigmoid((is_correct.float() * 2 - 1) + noise)  # ROC-AUC ~ 0.76
    return {
        "scores": scores.to(torch.float32),
        "is_correct": is_correct,
        "split": split,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "n": n,
            "seed": seed,
            "created_by": "synolitic.common.dummy.dummy_scores",
        },
    }
