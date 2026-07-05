"""Artifact IO helpers.

All pipeline artifacts are plain ``torch.save`` files. Loading uses
``weights_only=False`` because ``graphs.pt`` contains ``torch_geometric.data.Data``
objects (not loadable under the safe-tensors-only mode that is the default
since torch 2.6). Only load artifacts produced by this team.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_artifact(obj: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return path


def load_artifact(path: str | Path) -> object:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
