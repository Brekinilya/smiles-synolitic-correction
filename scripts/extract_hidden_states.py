"""Extract hidden_states.pt (the stage 1 -> 2 artifact) from a trained model.

Usage:
    uv run python scripts/extract_hidden_states.py \
        [--ckpt artifacts/checkpoints/model_best.pt] [--out artifacts/hidden_states.pt] \
        [--n 10000] [--attn-subsample 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from synolitic.common import schemas
from synolitic.common.io import load_artifact, save_artifact
from synolitic.stage1_model.extract import ExtractConfig, extract
from synolitic.stage1_model.model import ModelConfig, ReversalTransformer


def main() -> None:
    defaults = ExtractConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("artifacts/checkpoints/model_best.pt"))
    p.add_argument("--out", type=Path, default=Path("artifacts/hidden_states.pt"))
    p.add_argument("--n", type=int, default=defaults.n)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--min-len", type=int, default=defaults.min_len)
    p.add_argument("--max-len", type=int, default=defaults.max_len)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--attn-subsample", type=int, default=defaults.attn_subsample)
    p.add_argument("--device", default=defaults.device)
    args = p.parse_args()

    ckpt = load_artifact(args.ckpt)
    model = ReversalTransformer(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {args.ckpt} (step {ckpt['step']}, val exact-match {ckpt['val_acc']:.3f})")

    cfg = ExtractConfig(
        n=args.n, batch_size=args.batch_size, min_len=args.min_len, max_len=args.max_len,
        seed=args.seed, attn_subsample=args.attn_subsample, device=args.device,
    )
    artifact = extract(model, cfg)
    schemas.assert_valid("hidden_states", schemas.validate_hidden_states(artifact))
    out = save_artifact(artifact, args.out)

    ic = artifact["is_correct"]
    print(f"contract v{schemas.SCHEMA_VERSION} OK -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print(f"N={cfg.n}  model accuracy on extraction set: {float(ic.float().mean()):.3f}")
    for code, name in schemas.SPLIT_NAMES.items():
        m = artifact["split"] == code
        print(f"  {name:5s}: {int(m.sum()):6d} samples, "
              f"accuracy {float(ic[m].float().mean()):.3f}")
    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(ic.numpy(), artifact["confidence"].numpy())
        print(f"baseline check — ROC-AUC of confidence vs is_correct: {auc:.3f}")
    except ValueError:
        print("baseline ROC-AUC unavailable (single class present)")


if __name__ == "__main__":
    main()
