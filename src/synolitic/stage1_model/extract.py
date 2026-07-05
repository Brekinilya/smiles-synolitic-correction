"""Build the ``hidden_states.pt`` artifact (stage 1 -> 2 contract).

For every sample: greedy-decode the trained model on a freshly generated
evaluation set (seed disjoint from training), pool encoder hidden states,
record correctness, confidence and the train/cal/test split. Per-head
attention matrices are dumped for a random subsample in a second pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from synolitic.common.schemas import (
    BOS,
    D_H,
    PAD,
    SCHEMA_VERSION,
    exact_match,
    make_split,
)
from synolitic.stage1_model.dataset import make_dataset
from synolitic.stage1_model.model import ReversalTransformer
from synolitic.stage1_model.train import resolve_device


@dataclass
class ExtractConfig:
    n: int = 10000
    batch_size: int = 256
    min_len: int = 10
    max_len: int = 20
    seed: int = 20260705       # disjoint from training seeds by construction
    attn_subsample: int = 2000
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    device: str = "auto"


def _pool(states: torch.Tensor, src_kpm: torch.Tensor) -> torch.Tensor:
    """Mean over non-PAD source positions: [B, L, D] -> [B, D]."""
    mask = (~src_kpm).float().unsqueeze(-1)
    return (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def extract(model: ReversalTransformer, cfg: ExtractConfig) -> dict:
    device = resolve_device(cfg.device)
    model = model.to(device)
    model.eval()

    src, tgt, lengths = make_dataset(
        cfg.n, cfg.min_len, cfg.max_len, torch.Generator().manual_seed(cfg.seed)
    )
    ls = src.shape[1]
    t = ls + 1

    x_parts, xl_parts, pred_parts, conf_parts = [], [], [], []
    for s in range(0, cfg.n, cfg.batch_size):
        sb = src[s : s + cfg.batch_size].to(device)
        pred, conf, memory, src_kpm, hiddens = model.greedy_decode(sb, max_new_tokens=t)
        pred = nn.functional.pad(pred, (0, t - pred.shape[1]), value=PAD)
        x_parts.append(_pool(memory, src_kpm).cpu())
        xl_parts.append(torch.stack([_pool(h, src_kpm) for h in hiddens], dim=1).cpu())
        pred_parts.append(pred.cpu())
        conf_parts.append(conf.cpu())

    x = torch.cat(x_parts).to(torch.float32)
    x_layers = torch.cat(xl_parts).to(torch.float32)
    pred = torch.cat(pred_parts)
    confidence = torch.cat(conf_parts).clamp(0.0, 1.0)
    is_correct = exact_match(pred, tgt, lengths).to(torch.int8)
    split = make_split(cfg.n, cfg.split_fractions, torch.Generator().manual_seed(cfg.seed + 1))

    n_sub = min(cfg.attn_subsample, cfg.n)
    attn_idx = (
        torch.randperm(cfg.n, generator=torch.Generator().manual_seed(cfg.seed + 2))[:n_sub]
        .sort()
        .values
    )
    enc_parts, dec_parts, cross_parts = [], [], []
    for s in range(0, n_sub, cfg.batch_size):
        rows = attn_idx[s : s + cfg.batch_size]
        sb = src[rows].to(device)
        memory, src_kpm, _, enc_attns = model.encode(sb, need_attn=True)
        bos = torch.full((rows.shape[0], 1), BOS, dtype=torch.int64)
        tgt_in = torch.cat([bos, pred[rows][:, :-1]], dim=1).to(device)
        _, dec_attns, cross_attns = model.decode(tgt_in, memory, src_kpm, need_attn=True)
        enc_parts.append(torch.stack(enc_attns, dim=1).cpu())
        dec_parts.append(torch.stack(dec_attns, dim=1).cpu())
        cross_parts.append(torch.stack(cross_attns, dim=1).cpu())

    meta = {
        "schema_version": SCHEMA_VERSION,
        "d_h": D_H,
        "n": cfg.n,
        "seed": cfg.seed,
        "pooling": "mean over non-PAD source positions of final-LN encoder output",
        "hiddens_note": "X_layers[:, l] pools raw layer-l output (before the final LN)",
        "task": "reversal",
        "min_len": cfg.min_len,
        "max_len": cfg.max_len,
        "split_fractions": tuple(cfg.split_fractions),
        "model_config": asdict(model.config),
        "extract_config": asdict(cfg),
        "created_by": "synolitic.stage1_model.extract",
    }
    return {
        "X": x,
        "X_layers": x_layers,
        "input_tokens": src,
        "target_tokens": tgt,
        "lengths": lengths,
        "pred_tokens": pred,
        "is_correct": is_correct,
        "confidence": confidence.to(torch.float32),
        "split": split,
        "attn_idx": attn_idx,
        "enc_self_attn": torch.cat(enc_parts).to(torch.float32),
        "dec_self_attn": torch.cat(dec_parts).to(torch.float32),
        "cross_attn": torch.cat(cross_parts).to(torch.float32),
        "meta": meta,
    }
