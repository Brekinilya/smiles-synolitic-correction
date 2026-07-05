"""Training loop for the reversal transformer.

The corrector needs errors to detect, so the model must NOT be trained to
saturation: training stops once validation exact-match accuracy reaches
``target_acc`` and the checkpoint whose accuracy is closest to the target
(preferring the [band_low, band_high] window) is kept as ``model_best.pt``.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from synolitic.common.schemas import PAD, exact_match
from synolitic.stage1_model.dataset import make_dataset, make_targets
from synolitic.stage1_model.model import ModelConfig, ReversalTransformer


@dataclass
class TrainConfig:
    steps: int = 5000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    min_len: int = 10
    max_len: int = 20
    dropout: float = 0.1
    seed: int = 1
    eval_every: int = 250
    val_size: int = 500
    target_acc: float = 0.78
    band_low: float = 0.70
    band_high: float = 0.85
    device: str = "auto"
    out_dir: str = "artifacts/checkpoints"
    init_from: str | None = None  # warm-start from an existing checkpoint


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_exact_match(
    model: ReversalTransformer,
    val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> float:
    was_training = model.training
    model.eval()
    src, tgt, lengths = val
    max_new = src.shape[1] + 1
    matches = []
    for s in range(0, src.shape[0], batch_size):
        pred, _, _, _, _ = model.greedy_decode(src[s : s + batch_size].to(device), max_new)
        pred = nn.functional.pad(pred, (0, max_new - pred.shape[1]), value=PAD)
        matches.append(exact_match(pred.cpu(), tgt[s : s + batch_size], lengths[s : s + batch_size]))
    if was_training:
        model.train()
    return float(torch.cat(matches).float().mean())


def train(cfg: TrainConfig) -> dict:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(dropout=cfg.dropout)
    model = ReversalTransformer(model_cfg).to(device)
    if cfg.init_from:
        ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"warm start from {cfg.init_from} "
              f"(step {ckpt.get('step')}, val_acc {ckpt.get('val_acc', float('nan')):.3f})",
              flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    data_gen = torch.Generator().manual_seed(cfg.seed)
    val = make_dataset(cfg.val_size, cfg.min_len, cfg.max_len,
                       torch.Generator().manual_seed(cfg.seed + 1000))

    history: list[dict] = []
    loss_sum, loss_cnt = 0.0, 0
    since_eval = 0
    # Exact-match accuracy ramps up sharply once per-token accuracy is high;
    # evaluate more often near the band so a checkpoint lands inside it.
    next_eval = cfg.eval_every
    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        src, tgt, lengths = make_dataset(cfg.batch_size, cfg.min_len, cfg.max_len, data_gen)
        tgt_in, tgt_out = make_targets(tgt, lengths)
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        loss_sum += float(loss.detach())
        loss_cnt += 1
        since_eval += 1

        if since_eval >= next_eval or step == cfg.steps:
            since_eval = 0
            acc = evaluate_exact_match(model, val, device)
            avg_loss = loss_sum / max(loss_cnt, 1)
            loss_sum, loss_cnt = 0.0, 0
            ckpt_path = out_dir / f"step{step:06d}.pt"
            torch.save(
                {
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(cfg),
                    "step": step,
                    "val_acc": acc,
                },
                ckpt_path,
            )
            history.append({"step": step, "val_acc": acc, "loss": avg_loss,
                            "ckpt": str(ckpt_path)})
            print(f"step {step:5d}  loss {avg_loss:.4f}  val exact-match {acc:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if acc >= cfg.target_acc:
                break
            next_eval = cfg.eval_every if acc < 0.3 else min(100, cfg.eval_every)

    in_band = [h for h in history if cfg.band_low <= h["val_acc"] <= cfg.band_high]
    pool = in_band or history
    best = min(pool, key=lambda h: abs(h["val_acc"] - cfg.target_acc))
    best_path = out_dir / "model_best.pt"
    shutil.copyfile(best["ckpt"], best_path)
    for h in history:
        Path(h["ckpt"]).unlink(missing_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if not in_band:
        print(f"WARNING: no checkpoint landed in accuracy band "
              f"[{cfg.band_low}, {cfg.band_high}]; kept val_acc={best['val_acc']:.3f}. "
              f"Consider adjusting steps/eval_every and retraining.")
    print(f"best checkpoint: step {best['step']}, val exact-match {best['val_acc']:.3f} "
          f"-> {best_path}")
    return {"best_step": best["step"], "best_acc": best["val_acc"],
            "best_path": str(best_path), "history": history}
