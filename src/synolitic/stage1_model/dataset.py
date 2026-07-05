"""Sequence reversal task: synthetic data generation.

Data is unlimited and generated on the fly; disjoint seeds are all that is
needed to keep model-training data, validation data and the extraction set
independent.
"""

from __future__ import annotations

import torch
from torch import Tensor

from synolitic.common.schemas import BOS, EOS, FIRST_DIGIT, N_DIGITS, PAD


def make_dataset(
    n: int, min_len: int, max_len: int, generator: torch.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample ``n`` sequences padded to ``max_len``.

    Returns:
        src: [n, max_len] digit tokens, PAD beyond each length.
        tgt_digits: [n, max_len] reversed digit tokens, PAD beyond each length.
        lengths: [n] int64.
    """
    lengths = torch.randint(min_len, max_len + 1, (n,), generator=generator)
    pos = torch.arange(max_len).unsqueeze(0).expand(n, max_len)
    in_len = pos < lengths.unsqueeze(1)
    src = torch.randint(0, N_DIGITS, (n, max_len), generator=generator) + FIRST_DIGIT
    src = torch.where(in_len, src, torch.full_like(src, PAD))
    rev_idx = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0)
    tgt = torch.gather(src, 1, rev_idx)
    tgt = torch.where(in_len, tgt, torch.full_like(tgt, PAD))
    return src, tgt, lengths


def make_targets(tgt_digits: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    """Teacher-forcing pair: tgt_in = [BOS, y...], tgt_out = [y..., EOS]."""
    n, max_len = tgt_digits.shape
    tgt_in = torch.full((n, max_len + 1), PAD, dtype=torch.int64)
    tgt_in[:, 0] = BOS
    tgt_in[:, 1:] = tgt_digits
    tgt_out = torch.full((n, max_len + 1), PAD, dtype=torch.int64)
    tgt_out[:, :max_len] = tgt_digits
    tgt_out[torch.arange(n), lengths] = EOS
    return tgt_in, tgt_out
