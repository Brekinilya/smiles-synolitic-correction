# Pre-defense results (2026-07-12)

Full pipeline run on real artifacts: reversal transformer → synolitic graphs
→ GNN error classifier → corrector with Theorem-1 guarantees.

## Provenance

| artifact | release | produced by | key numbers |
|---|---|---|---|
| `hidden_states.pt` | `v0.1-data` | stage 1 (`extract_hidden_states.py`) | N=10000, model exact-match 0.724, splits 6000/2000/2000 |
| `stage1_model_step7900.pt` | `v0.1-data` | stage 1 (two-pass training, see below) | 235 405 params, weights only, val exact-match 0.766 |
| `graphs.pt` | `v0.2-data` | stage 2 @ `eaaa6f9` | 64 nodes, top-252 edges, decisiveness topology |
| `scores.pt` | `v0.3-data` | stage 3 @ `fa3e948` | GATv2 ensemble (5 seeds), focal loss, confidence as 6th node feature, early stop on a 600-row train slice (cal untouched), **test AUC 0.839** |

### Stage 1: exact reproduction (bit-identical to `v0.1-data`)

The released `hidden_states.pt` requires a **two-pass** training schedule — the
single obvious command never reaches the target accuracy band. Re-running the
recipe below regenerates `hidden_states.pt` **bit-identical** to `v0.1-data`:
every tensor (`X`, `X_layers`, `pred_tokens`, `is_correct`, `confidence`, the
splits, …) matches with `max|Δ| = 0`.

```bash
# 1a  deliberate undertraining — default settings stall at val exact-match
#     0.046 (step 5000). This is intentional; it only seeds the warm start.
uv run python scripts/train_model.py --out-dir artifacts/checkpoints_pretrain
# 1b  warm start — higher LR + a fine eval grid catch the 0.70–0.85 band at
#     pass-1b step 2900 (cumulative 7900), val exact-match 0.766.
uv run python scripts/train_model.py \
    --init-from artifacts/checkpoints_pretrain/model_best.pt \
    --lr 1e-3 --steps 4000 --eval-every 50
# 1c  extract — N=10000, splits 6000/2000/2000, model accuracy 0.724
#     (train 0.720 / cal 0.733 / test 0.724; 535 errors in the cal split).
uv run python scripts/extract_hidden_states.py
```

The trained checkpoint is published as
[`stage1_model_step7900.pt`](https://github.com/Brekinilya/smiles-synolitic-correction/releases/download/v0.1-data/stage1_model_step7900.pt)
on `v0.1-data` (235 405 params, weights only, 963 317 B). The corrector's
operating point on `gnn_score` at Δ=0.8 is **θ = 0.6156**, **M⁻ = 535** (the
535 errors in the calibration split).

Reproduce the table below:

```bash
uv run python scripts/run_corrector.py \
    --scores artifacts/scores.pt \
    --graphs artifacts/graphs.pt \
    --hidden-states artifacts/hidden_states.pt
```

## Final corrector table (test split, N=2000, base accuracy 0.724)

`acc>=` / `rej>=` are the distribution-agnostic Theorem-1 bounds computed on
the calibration split; `acc(emp)` / `rej(emp)` are the observed test rates.
**Every empirical rate respects its bound.**

```
               feature Delta  acc>=  acc(emp)  rej>=  rej(emp)  prec@acc   kept   AUC(test)
-------------------------------------------------------------------------------------------
             gnn_score  0.80  0.650     0.704  0.722     0.821     0.911  0.559   0.839
            confidence  0.80  0.512     0.559  0.722     0.826     0.894  0.453   0.799
          raw_x_fisher  0.80  0.350     0.436  0.692     0.792     0.846  0.373   0.686
     graph_topo_fisher  0.80  0.279     0.332  0.692     0.792     0.807  0.298   0.631
  graph_summary_fisher  0.80  0.151     0.269  0.692     0.832     0.807  0.241   0.619

             gnn_score  0.90  0.459     0.488  0.822     0.906     0.931  0.380   0.839
            confidence  0.90  0.341     0.386  0.822     0.928     0.933  0.299   0.799
          raw_x_fisher  0.90  0.159     0.249  0.791     0.886     0.851  0.212   0.686
     graph_topo_fisher  0.90  0.170     0.203  0.791     0.862     0.795  0.185   0.631
  graph_summary_fisher  0.90  0.053     0.148  0.791     0.911     0.814  0.132   0.619

     gnn_score @youden     -      -     0.767      -     0.775     0.900  0.617   (no bounds)
    confidence @youden     -      -     0.763      -     0.679     0.862  0.641   (no bounds)
  raw_x_fisher @youden     -      -     0.700      -     0.609     0.824  0.615   (no bounds)
graph_topo_fisher @youden     -      -     0.556      -     0.658     0.810  0.497   (no bounds)
graph_summary_fisher @youden     -      -     0.809      -     0.362     0.769  0.761   (no bounds)
```

## Headlines

1. **The synolitic corrector keeps more answers at higher precision than the
   softmax baseline — but the raw-AUC gap is a length confound.** GNN on
   synolitic graphs (with model confidence as a node feature) reaches test
   AUC **0.839** vs **0.799** for confidence alone; this global gap **does not
   survive length control** — within-length it reverses (see "H1 under length
   control" below). As a corrector at Δ=0.8 it keeps **55.9%** of answers at
   **91.1% precision** (base accuracy 72.4%) vs 45.3% at 89.4% for the
   confidence corrector, rejecting **82.1%** of errors. At d=64 this
   operational gain partly rides the L=20 error cluster.
2. **Theorem-1 guarantees hold empirically.** All 20 bound comparisons
   (5 feature sets × 2 Δ × accept/reject) are satisfied on the untouched
   test split; thresholds and bounds were computed on the calibration split
   only (Fisher projectors on cal-A, thresholds on cal-B).
3. **Honest H1 reading.** Pure graph-topology features projected by a
   small-sample Fisher discriminant do not yet beat raw hidden states
   (AUC 0.631 vs 0.686). The synolitic advantage appears (a) when node
   identity is preserved (flat features 0.699 > pooled 0.634), and (b) in
   combination with model confidence (0.839 > 0.799; adding raw X on top of
   graph features adds nothing, 0.842 vs 0.841 in the linear probe). With
   d=64 this Setup-1 testbed is the smallest of the proposal's three setups;
   the blessing-of-dimensionality effects the method targets are expected to
   grow at d=379 (fMRI) and d=4096 (Llama 2).

### H1 under length control

The +0.040 global gap is partly a **length confound**. Sequence length is almost
perfectly recoverable from the pooled hidden state (ridge R² = 0.9995), and
exact-match accuracy varies non-monotonically with length, collapsing at L=20
(0.377 vs ~0.75 elsewhere; 19.5% of all errors). Confidence declines smoothly
with length (Pearson r = −0.73, all N) while accuracy does not, so its length
dependence is miscalibrated across strata and costs it globally; the GNN score
(r = −0.30) responds specifically to the L=20 collapse, which is genuinely
predictive. Stratifying by length removes both effects.

| comparison        | GNN   | confidence | GNN − conf |
|-------------------|-------|------------|------------|
| global AUC        | 0.839 | 0.799      | +0.040     |
| within-length AUC | 0.806 | 0.817      | −0.010     |

Sample-weighted over L=10..20, test split. Bootstrap (2000 resamples of the test
split): GNN − conf = −0.010, 95% CI [−0.018, −0.003]; the reversal is not noise.

At d=64 our current pipeline does not beat confidence once length is controlled.
Note this is a statement about the current implementation: the pairwise
classifiers are fitted without feature standardization under L2, which may bias
edge selection toward high-variance coordinates. Reproduce with:

    uv run python scripts/h1_length_control.py

## H2 stress test: guarantees vs labeling budget

H2 claims meaningful bounds with fewer than 100 labeled correction examples.
We subsample the calibration split to M labeled examples, refit Algorithm 1
(Δ = 0.8, GNN score) and read bounds vs empirical test rates (30 random
subsamples per M; the test split is never used for calibration):

```
    M    M- | rej bound  rej emp | acc bound  acc emp | prec@acc
   30     7 |     0.330    0.784 |     0.295    0.669 |    0.902
   50    13 |     0.416    0.772 |     0.456    0.726 |    0.898
  100    27 |     0.511    0.806 |     0.502    0.694 |    0.907
  200    55 |     0.586    0.803 |     0.550    0.707 |    0.906
  500   134 |     0.656    0.818 |     0.596    0.699 |    0.910
 1000   268 |     0.694    0.815 |     0.635    0.710 |    0.910
 2000   535 |     0.722    0.821 |     0.650    0.704 |    0.911
```

Empirical corrector quality is nearly flat in M (rejection ~0.78–0.82 and
precision ~0.90 already at M = 30): **labeling buys the strength of the
guarantee, not raw performance** — the certified lower bound grows
0.33 → 0.51 → 0.72. Bounds are already meaningful at M = 100, which is the
literal statement of H2. Every row respects its bound. Reproduce with:

```bash
uv run python scripts/h2_labeling_curve.py
```

## H3: mechanistic error decomposition (replaces the original H3)

The original H3 — that synolitic graph topology correlates with attention-head
specialization — is **not supported**. The two quantities live in different
spaces (8 attention heads over token positions vs the 64 hidden dimensions the
graph is built from; a graph is a function of the input, not of a head), and the
strong Spearman correlations reported for it in the original analysis (|r| up to
0.935) are a **length confound**: attention entropy is bounded by log L and
rises with length, while node centrality, derived from X, falls with length, so
any pair of them correlates mechanically; the association does not survive
stratification by length. In its place, a mechanistic analysis of the
cross-attention decomposes the model's errors into three independently measured
sources. Reproduce all of the below with:

    uv run python scripts/h3_mirror_map.py

**1. A redundant reversal circuit.** All four layer-2 cross-attention heads
implement the reversal, attending to the mirror position (decode step _t_ →
source position _L−1−t_): mirror hit-rate 0.79–0.86 vs a chance of 0.07, while
layer-1 heads and the identity (diagonal) pattern sit at chance. The reversal is
carried redundantly across the layer, not by a single specialized head.

**2. Mirror fidelity predicts correctness — within length, not as a confound.**
Within every fixed length, correct predictions have higher mirror fidelity than
errors: sample-weighted gap **+0.091, 95 % CI [+0.079, +0.104]** (stratified
bootstrap over the 11 length strata, 2000 resamples; 0/2000 ≤ 0). Not a
tautology: **~25 % of errors (140 / 554) occur with the copy intact** (mirror >
0.9), so errors have sources the copy circuit does not capture.

**3. Termination / EOS-timing is a dominant error source.** Roughly 50–90 % of
errors are the model emitting EOS at the wrong position (wrong output length),
not wrong content — the large majority at short lengths (~85–90 %). This is a
decoder mechanism distinct from the copy, and it accounts for the "copy intact
but wrong" errors above.

**4. Copy degradation at maximum length explains the L=20 collapse.** The mirror
hit-rate degrades only smoothly with length (L=20 residual −0.06 vs the L=10–19
trend), yet accuracy collapses sharply (L=20 residual −0.40) — the failure at
the boundary is in copy _precision_, not attention _position_. Per-token content
accuracy (correctly-terminated) falls from ~0.99 at short lengths to 0.941 at
L=20. Constant-rate compounding is ruled out — using the short-length ~0.987
per-token accuracy it predicts exact-match 0.77 vs the observed 0.38 — but the
measured per-token drop plus **error correlation** (token errors cluster in a
subset of hard examples; observed / independent rises 1.03 → 1.90 with length)
reproduces the collapse exactly: at L=20, 33 % termination failures + 67 %
correctly-terminated × 55.9 % fully correct = 0.377 = observed exact-match.

The bootstrap CI in (2) applies only to the reversal-circuit predictor; (3) and
(4) stand on exact decompositions, not mean estimates.

## Ablation history worth keeping (stage 3)

Five permutation-invariant GNN variants (GCN/GATv2 ×2–3 layers, ±BatchNorm,
±edge_attr, ±pos_weight) all plateaued at test AUC 0.59–0.66 — the
information ceiling of mean/max pooling measured by a linear probe is 0.634.
Switching to an identity-preserving readout (`x.view(B, 64·hidden)`) plus
confidence lifted the same graphs to 0.839. The lesson: synolitic graphs have
fixed node identity, and readouts must preserve it.
