"""Generate dummy artifacts for all pipeline stages.

Usage:
    uv run python scripts/make_dummy_data.py [--n 512] [--out-dir artifacts/dummy]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synolitic.common import dummy, schemas
from synolitic.common.io import save_artifact


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--attn-subsample", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/dummy"))
    args = p.parse_args()

    hs = dummy.dummy_hidden_states(n=args.n, attn_subsample=args.attn_subsample, seed=args.seed)
    schemas.assert_valid("hidden_states", schemas.validate_hidden_states(hs))
    path = save_artifact(hs, args.out_dir / "hidden_states.pt")
    acc = float(hs["is_correct"].float().mean())
    print(f"{path}  N={args.n}  model-accuracy={acc:.3f}")

    graphs = dummy.dummy_graphs(hidden_states=hs, seed=args.seed)
    schemas.assert_valid("graphs", schemas.validate_graphs(graphs))
    path = save_artifact(graphs, args.out_dir / "graphs.pt")
    print(f"{path}  graphs={len(graphs['graphs'])}")

    scores = dummy.dummy_scores(hidden_states=hs, seed=args.seed)
    schemas.assert_valid("scores", schemas.validate_scores(scores))
    path = save_artifact(scores, args.out_dir / "scores.pt")
    print(f"{path}  N={args.n}")

    print("all dummy artifacts pass their contracts")


if __name__ == "__main__":
    main()
