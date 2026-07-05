"""Train the reversal transformer into the target accuracy band.

Usage:
    uv run python scripts/train_model.py [--steps 5000] [--target-acc 0.78] ...
"""

from __future__ import annotations

import argparse

from synolitic.stage1_model.train import TrainConfig, train


def main() -> None:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=defaults.steps)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--min-len", type=int, default=defaults.min_len)
    p.add_argument("--max-len", type=int, default=defaults.max_len)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--eval-every", type=int, default=defaults.eval_every)
    p.add_argument("--val-size", type=int, default=defaults.val_size)
    p.add_argument("--target-acc", type=float, default=defaults.target_acc)
    p.add_argument("--band-low", type=float, default=defaults.band_low)
    p.add_argument("--band-high", type=float, default=defaults.band_high)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--out-dir", default=defaults.out_dir)
    p.add_argument("--init-from", default=None,
                   help="warm-start from an existing checkpoint")
    args = p.parse_args()

    cfg = TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    result = train(cfg)
    print(f"done: best_acc={result['best_acc']:.3f} -> {result['best_path']}")


if __name__ == "__main__":
    main()
