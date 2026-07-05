"""Data contracts (v1.1) shared by all pipeline stages.

Single source of truth for:

* token ids of the reversal task (part of the stage-1 artifact contract),
* train/cal/test split codes and the split-usage rules,
* the exact-match correctness definition (``exact_match``),
* schemas and validators for the three inter-stage artifacts:

  - ``hidden_states.pt``  (stage 1 -> 2), see ``validate_hidden_states``
  - ``graphs.pt``         (stage 2 -> 3), see ``validate_graphs``
  - ``scores.pt``         (stage 3 -> 4), see ``validate_scores``

Split-usage rules (leakage protocol):

* ``SPLIT_TRAIN`` — fitting pairwise classifiers (stage 2) and the GNN (stage 3);
* ``SPLIT_CAL``   — calibrating corrector thresholds Delta_j (stage 4);
* ``SPLIT_TEST``  — final reported metrics only. Never fit or calibrate on it.

Every stage should validate artifacts it produces *and* consumes::

    from synolitic.common import schemas
    schemas.assert_valid("hidden_states", schemas.validate_hidden_states(art))
"""

from __future__ import annotations

import torch
from torch import Tensor

SCHEMA_VERSION = "1.1"

# Model / graph dimensions fixed by the proposal (Setup 1).
D_H = 64                # transformer hidden size == number of graph nodes
N_NODE_FEATURES = 5     # [value, degree, strength, closeness, betweenness]

# Reversal-task vocabulary. Token ids are part of the artifact contract.
PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3
N_DIGITS = 10
FIRST_DIGIT = N_SPECIAL             # digit d is encoded as token d + FIRST_DIGIT
VOCAB_SIZE = N_SPECIAL + N_DIGITS   # 13

SPLIT_TRAIN, SPLIT_CAL, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_CAL: "cal", SPLIT_TEST: "test"}


def exact_match(pred_tokens: Tensor, target_tokens: Tensor, lengths: Tensor) -> Tensor:
    """Contract definition of ``is_correct`` for the reversal task.

    A prediction is correct iff the tokens before the first EOS are exactly
    the target digit sequence (same length, same tokens). Missing EOS, extra
    or missing tokens, PAD/BOS emitted mid-sequence all count as incorrect.

    Args:
        pred_tokens: [N, T] generated tokens (EOS included, PAD after it).
        target_tokens: [N, L] target digit tokens, PAD beyond each length.
        lengths: [N] true sequence lengths.

    Returns:
        [N] bool tensor.
    """
    n = pred_tokens.shape[0]
    out = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        length = int(lengths[i])
        row = pred_tokens[i]
        eos = (row == EOS).nonzero(as_tuple=True)[0]
        seq = row[: int(eos[0])] if eos.numel() else row
        out[i] = seq.numel() == length and bool((seq == target_tokens[i, :length]).all())
    return out


def make_split(n: int, fractions: tuple[float, float, float], generator: torch.Generator) -> Tensor:
    """Random train/cal/test assignment as an int8 tensor of split codes."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    perm = torch.randperm(n, generator=generator)
    n_train = round(fractions[0] * n)
    n_cal = round(fractions[1] * n)
    codes = torch.empty(n, dtype=torch.int8)
    codes[perm[:n_train]] = SPLIT_TRAIN
    codes[perm[n_train : n_train + n_cal]] = SPLIT_CAL
    codes[perm[n_train + n_cal :]] = SPLIT_TEST
    return codes


def assert_valid(name: str, errors: list[str]) -> None:
    if errors:
        raise ValueError(f"{name} contract violations:\n- " + "\n- ".join(errors))


# --------------------------------------------------------------------------
# hidden_states.pt  (stage 1 -> 2)
# --------------------------------------------------------------------------

_HS_REQUIRED: dict[str, torch.dtype] = {
    "X": torch.float32,               # [N, D_H] pooled encoder hidden states
    "input_tokens": torch.int64,      # [N, Ls] source digits, PAD-padded
    "target_tokens": torch.int64,     # [N, Ls] reversed digits, PAD-padded
    "pred_tokens": torch.int64,       # [N, T]  greedy output incl. EOS
    "lengths": torch.int64,           # [N]
    "is_correct": torch.int8,         # [N] exact_match(pred, target, lengths)
    "confidence": torch.float32,      # [N] mean max-softmax over generated steps
    "split": torch.int8,              # [N] SPLIT_* codes
    "attn_idx": torch.int64,          # [N_sub] rows covered by attention dumps
    "enc_self_attn": torch.float32,   # [N_sub, layers, heads, Ls, Ls]
    "dec_self_attn": torch.float32,   # [N_sub, layers, heads, T, T]
    "cross_attn": torch.float32,      # [N_sub, layers, heads, T, Ls]
}

_REQUIRED_META = ("schema_version", "d_h", "seed", "pooling")


def validate_hidden_states(artifact: object) -> list[str]:
    """Return a list of contract violations (empty list == valid)."""
    if not isinstance(artifact, dict):
        return ["artifact must be a dict"]
    errors: list[str] = []
    for key, dtype in _HS_REQUIRED.items():
        if key not in artifact:
            errors.append(f"missing key '{key}'")
        elif not torch.is_tensor(artifact[key]):
            errors.append(f"'{key}' must be a torch.Tensor")
        elif artifact[key].dtype != dtype:
            errors.append(f"'{key}' dtype must be {dtype}, got {artifact[key].dtype}")
    meta = artifact.get("meta")
    if not isinstance(meta, dict):
        errors.append("missing 'meta' dict")
    if errors:
        return errors

    x = artifact["X"]
    n = x.shape[0]
    ls = artifact["input_tokens"].shape[1]
    t = artifact["pred_tokens"].shape[1]

    if x.ndim != 2 or x.shape[1] != D_H:
        errors.append(f"X must be [N, {D_H}], got {list(x.shape)}")
    if not torch.isfinite(x).all():
        errors.append("X contains non-finite values")

    for key in ("input_tokens", "target_tokens", "pred_tokens", "lengths",
                "is_correct", "confidence", "split"):
        if artifact[key].shape[0] != n:
            errors.append(f"'{key}' first dim must be N={n}")
    if artifact["target_tokens"].shape != (n, ls):
        errors.append("target_tokens must have the same shape as input_tokens")

    for key in ("input_tokens", "target_tokens", "pred_tokens"):
        v = artifact[key]
        if v.numel() and (v.min() < 0 or v.max() >= VOCAB_SIZE):
            errors.append(f"'{key}' has token ids outside [0, {VOCAB_SIZE})")

    lengths = artifact["lengths"]
    if lengths.numel() and (lengths.min() < 1 or lengths.max() > ls):
        errors.append(f"lengths must lie in [1, {ls}]")

    conf = artifact["confidence"]
    if conf.numel() and (conf.min() < 0 or conf.max() > 1):
        errors.append("confidence must lie in [0, 1]")

    ic = artifact["is_correct"]
    if not set(ic.unique().tolist()) <= {0, 1}:
        errors.append("is_correct must contain only 0/1")
    else:
        recomputed = exact_match(artifact["pred_tokens"], artifact["target_tokens"], lengths)
        n_bad = int((recomputed.to(torch.int8) != ic).sum())
        if n_bad:
            errors.append(
                f"is_correct disagrees with recomputed exact match on {n_bad}/{n} samples"
            )

    split = artifact["split"]
    codes = set(split.unique().tolist())
    if not codes <= set(SPLIT_NAMES):
        errors.append("split contains unknown codes")
    elif n >= 100 and codes != set(SPLIT_NAMES):
        errors.append("all three splits (train/cal/test) must be present for N >= 100")

    idx = artifact["attn_idx"]
    n_sub = idx.shape[0]
    if idx.numel() and (idx.min() < 0 or idx.max() >= n):
        errors.append("attn_idx out of range")
    if idx.unique().numel() != n_sub:
        errors.append("attn_idx must be unique")
    enc = artifact["enc_self_attn"]
    dec = artifact["dec_self_attn"]
    cross = artifact["cross_attn"]
    if enc.ndim != 5 or enc.shape[0] != n_sub or enc.shape[-2:] != (ls, ls):
        errors.append(f"enc_self_attn must be [N_sub, layers, heads, {ls}, {ls}]")
    else:
        layers, heads = enc.shape[1], enc.shape[2]
        if dec.shape != (n_sub, layers, heads, t, t):
            errors.append(f"dec_self_attn must be [N_sub, {layers}, {heads}, {t}, {t}]")
        if cross.shape != (n_sub, layers, heads, t, ls):
            errors.append(f"cross_attn must be [N_sub, {layers}, {heads}, {t}, {ls}]")
        for key in ("enc_self_attn", "dec_self_attn", "cross_attn"):
            v = artifact[key]
            if v.numel() and not torch.allclose(
                v.sum(dim=-1), torch.ones_like(v.sum(dim=-1)), atol=1e-2
            ):
                errors.append(f"'{key}' rows must sum to 1 (softmax weights)")

    if "X_layers" in artifact:
        xl = artifact["X_layers"]
        if not torch.is_tensor(xl) or xl.dtype != torch.float32 or xl.ndim != 3 \
                or xl.shape[0] != n or xl.shape[2] != D_H:
            errors.append(f"X_layers (optional) must be float32 [N, n_layers, {D_H}]")

    for key in _REQUIRED_META:
        if key not in meta:
            errors.append(f"meta missing '{key}'")
    if meta.get("d_h") not in (None, D_H):
        errors.append(f"meta.d_h must be {D_H}")
    return errors


# --------------------------------------------------------------------------
# graphs.pt  (stage 2 -> 3)
# --------------------------------------------------------------------------

def validate_graphs(artifact: object, max_errors: int = 20) -> list[str]:
    """Validate a graphs artifact.

    Preferred form is ``{"graphs": list[torch_geometric.data.Data], "meta": dict}``;
    a bare list is accepted for compatibility. Each Data must carry:
    ``x`` [D_H, N_NODE_FEATURES] float32, ``edge_index`` [2, E] int64 in [0, D_H),
    ``edge_attr`` [E] or [E, 1] float32, ``y`` [1] float32 in {0, 1},
    ``idx`` [1] int64 (row in hidden_states.pt), ``split`` [1] int8.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:  # pragma: no cover
        return ["torch_geometric is not installed"]

    if isinstance(artifact, dict):
        graphs = artifact.get("graphs")
        if not isinstance(artifact.get("meta"), dict):
            return ["graphs artifact dict must contain a 'meta' dict"]
        if not isinstance(graphs, list):
            return ["graphs artifact dict must contain a 'graphs' list"]
    elif isinstance(artifact, list):
        graphs = artifact
    else:
        return ["graphs artifact must be a dict {'graphs': [...], 'meta': {...}} or a list"]

    errors: list[str] = []
    if not graphs:
        return ["graph list is empty"]

    seen_idx: set[int] = set()
    for i, d in enumerate(graphs):
        if len(errors) >= max_errors:
            errors.append("... (further errors truncated)")
            break
        where = f"graph[{i}]"
        if not isinstance(d, Data):
            errors.append(f"{where}: not a torch_geometric Data object")
            continue
        if d.x is None or d.x.shape != (D_H, N_NODE_FEATURES) or d.x.dtype != torch.float32:
            errors.append(f"{where}: x must be float32 [{D_H}, {N_NODE_FEATURES}]")
        elif not torch.isfinite(d.x).all():
            errors.append(f"{where}: x contains non-finite values")
        ei = d.edge_index
        if ei is None or ei.dtype != torch.int64 or ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] == 0:
            errors.append(f"{where}: edge_index must be a non-empty int64 [2, E]")
        else:
            if ei.min() < 0 or ei.max() >= D_H:
                errors.append(f"{where}: edge_index values must lie in [0, {D_H})")
            ea = d.edge_attr
            e = ei.shape[1]
            if ea is None or ea.dtype != torch.float32 or ea.shape not in ((e,), (e, 1)):
                errors.append(f"{where}: edge_attr must be float32 [E] or [E, 1] with E={e}")
        if d.y is None or d.y.shape != (1,) or d.y.dtype != torch.float32 \
                or float(d.y) not in (0.0, 1.0):
            errors.append(f"{where}: y must be float32 [1] in {{0, 1}}")
        if getattr(d, "idx", None) is None or d.idx.shape != (1,) or d.idx.dtype != torch.int64:
            errors.append(f"{where}: idx must be int64 [1] (row in hidden_states.pt)")
        else:
            j = int(d.idx)
            if j in seen_idx:
                errors.append(f"{where}: duplicate idx {j}")
            seen_idx.add(j)
        sp = getattr(d, "split", None)
        if sp is None or sp.shape != (1,) or sp.dtype != torch.int8 \
                or int(sp) not in SPLIT_NAMES:
            errors.append(f"{where}: split must be int8 [1] with a valid SPLIT_* code")
    return errors


# --------------------------------------------------------------------------
# scores.pt  (stage 3 -> 4)
# --------------------------------------------------------------------------

_SCORES_REQUIRED: dict[str, torch.dtype] = {
    "scores": torch.float32,     # [N] GNN estimate of P(is_correct == 1)
    "is_correct": torch.int8,    # [N]
    "split": torch.int8,         # [N]
}


def validate_scores(artifact: object) -> list[str]:
    if not isinstance(artifact, dict):
        return ["artifact must be a dict"]
    errors: list[str] = []
    for key, dtype in _SCORES_REQUIRED.items():
        if key not in artifact:
            errors.append(f"missing key '{key}'")
        elif not torch.is_tensor(artifact[key]):
            errors.append(f"'{key}' must be a torch.Tensor")
        elif artifact[key].dtype != dtype:
            errors.append(f"'{key}' dtype must be {dtype}, got {artifact[key].dtype}")
    if not isinstance(artifact.get("meta"), dict):
        errors.append("missing 'meta' dict")
    if errors:
        return errors

    n = artifact["scores"].shape[0]
    for key in _SCORES_REQUIRED:
        if artifact[key].shape != (n,):
            errors.append(f"'{key}' must be a flat [N] tensor with N={n}")
    s = artifact["scores"]
    if s.numel() and (s.min() < 0 or s.max() > 1):
        errors.append("scores must lie in [0, 1] (probability of a correct prediction)")
    if not set(artifact["is_correct"].unique().tolist()) <= {0, 1}:
        errors.append("is_correct must contain only 0/1")
    if not set(artifact["split"].unique().tolist()) <= set(SPLIT_NAMES):
        errors.append("split contains unknown codes")
    return errors
