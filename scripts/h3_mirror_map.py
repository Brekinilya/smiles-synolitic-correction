"""H3 (replacement): mechanistic decomposition of the reversal model's errors.

The original H3 -- that synolitic graph topology correlates with attention-head
specialization -- is NOT supported: the quantities live in different spaces
(8 attention heads over token positions vs the 64 hidden dimensions the graph is
built from; a graph is a function of the input, not of a head), and the reported
Spearman correlations are a length confound. In its place, a mechanistic
analysis of the cross-attention decomposes the model's errors into three
independently measured sources. Every number in the docs/results.md "H3" section
is reproduced here. Uses only artifacts/hidden_states.pt (attention dumps +
tokens); no model, no graphs.

    uv run python scripts/h3_mirror_map.py
"""

from __future__ import annotations

import numpy as np

from synolitic.common.io import load_artifact
from synolitic.common.schemas import EOS

hs = load_artifact("artifacts/hidden_states.pt")
att = hs["cross_attn"].numpy()                 # [N_sub, layers, heads, T, Ls]
idx = hs["attn_idx"].numpy()
lengths = hs["lengths"].numpy()[idx]           # subsample lengths (attention set)
correct = hs["is_correct"].numpy()[idx].astype(int)
pred_all = hs["pred_tokens"].numpy()           # [N, T]   all N (token analysis)
tgt_all = hs["target_tokens"].numpy()          # [N, Ls]
lengths_all = hs["lengths"].numpy()
ic_all = hs["is_correct"].numpy().astype(int)
n_sub, n_layers, n_heads, T, Ls = att.shape


# ---- 1. the reversal circuit: mirror hit-rate per head --------------------
# decode step t must copy source position L-1-t; hit = fraction of steps whose
# cross-attention argmax lands on that mirror position (a fraction -> length
# invariant, no log-L ceiling). Identity (diagonal) hit shown as a contrast.
hit = np.zeros((n_sub, n_layers, n_heads))
ident = np.zeros((n_sub, n_layers, n_heads))
for s in range(n_sub):
    l = int(lengths[s])
    am = att[s, :, :, :l, :l].argmax(-1)
    hit[s] = (am == (l - 1 - np.arange(l))).mean(-1)
    ident[s] = (am == np.arange(l)).mean(-1)

chance = (1.0 / lengths).mean()
print(f"=== 1. reversal circuit: per-head hit (chance ~{chance:.3f}) ===")
for i in range(n_layers):
    print("  " + "   ".join(
        f"L{i+1}H{j+1} mir {hit[:, i, j].mean():.3f} id {ident[:, i, j].mean():.3f}"
        for j in range(n_heads)))
bi, bj = np.unravel_index(hit.mean(0).argmax(), (n_layers, n_heads))
mh = hit[:, bi, bj]
print(f"  -> all layer-2 heads mirror (~0.8), layer-1 & identity at chance: "
      f"REDUNDANT circuit (reference head L{bi+1}H{bj+1}, hit {mh.mean():.3f})")


# ---- 2. mirror fidelity predicts correctness WITHIN length (+bootstrap CI) --
strata = {}
num = den = 0.0
for L in range(int(lengths.min()), int(lengths.max()) + 1):
    m = lengths == L
    ok = np.where(m & (correct == 1))[0]
    er = np.where(m & (correct == 0))[0]
    if len(ok) >= 10 and len(er) >= 10:
        w = int(m.sum())
        strata[L] = (ok, er, w)
        num += (mh[ok].mean() - mh[er].mean()) * w
        den += w
gap = num / den
rng = np.random.default_rng(0)
boot = np.empty(2000)
for b in range(2000):
    n = d = 0.0
    for L, (ok, er, w) in strata.items():
        n += (mh[rng.choice(ok, len(ok), True)].mean()
              - mh[rng.choice(er, len(er), True)].mean()) * w
        d += w
    boot[b] = n / d
lo, hi = np.percentile(boot, [2.5, 97.5])
print("\n=== 2. mirror fidelity predicts correctness, within fixed length ===")
print(f"  gap (correct-error) {gap:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"(stratified bootstrap over {len(strata)} strata; {(boot <= 0).mean() * 100:.1f}% <= 0)")


# ---- 3. not a tautology: errors happen with the copy intact ----------------
n_err = int((correct == 0).sum())
hit_err = int(((mh > 0.9) & (correct == 0)).sum())
print("\n=== 3. not tautological ===")
print(f"  errors with mirror intact (hit>0.9): {hit_err} of {n_err} "
      f"({hit_err / n_err * 100:.0f}%) -> errors have sources the copy does not capture")


# ---- 4. mirror smooth vs accuracy sharp collapse at L=20 -------------------
print("\n=== 4. mirror (smooth) vs accuracy (sharp at L=20) ===")
La, mir, acc = [], [], []
for L in sorted(strata):
    m = lengths == L
    La.append(L)
    mir.append(mh[m].mean())
    acc.append(correct[m].mean())
La, mir, acc = np.array(La), np.array(mir), np.array(acc)
fm = La < 20
mres = mir[-1] - np.polyval(np.polyfit(La[fm], mir[fm], 1), 20)
ares = acc[-1] - np.polyval(np.polyfit(La[fm], acc[fm], 1), 20)
print(f"  L=20 mirror {mir[-1]:.3f} (resid {mres:+.3f}) vs accuracy {acc[-1]:.3f} (resid {ares:+.3f})")


# ---- 5-7. error decomposition on ALL N ------------------------------------
peos = (pred_all == EOS).any(1)
pred_len = np.where(peos, (pred_all == EOS).argmax(1), pred_all.shape[1])
print("\n=== 5-7. error decomposition (all N): termination / copy / correlation ===")
print(f"  {'L':>3} {'term%err':>9} {'tok_acc|term':>13} {'exact|term':>11} {'obs/indep':>10}")
for L in range(10, 21):
    m = lengths_all == L
    er = m & (ic_all == 0)
    to = m & (pred_len == L)
    termpct = (pred_len[er] != L).mean() * 100 if er.sum() else float("nan")
    tok = (pred_all[to, :L] == tgt_all[to, :L]).mean() if to.sum() else float("nan")
    exact_t = ic_all[to].mean() if to.sum() else float("nan")
    print(f"  {L:>3} {termpct:>8.0f}% {tok:>13.3f} {exact_t:>11.3f} {exact_t / tok ** L:>10.2f}")
m20 = lengths_all == 20
to20 = m20 & (pred_len == 20)
ft, et = to20.sum() / m20.sum(), ic_all[to20].mean()
print(f"  L=20: {(1 - ft) * 100:.0f}% termination-fail + {ft * 100:.0f}% term-ok x {et:.3f} correct "
      f"= {ft * et:.3f}  (observed exact {ic_all[m20].mean():.3f})")
