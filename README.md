# Synolitic Correction of the Transformer Architecture

SMILES 2026 team project. We detect and reject errors of a Transformer by
(1) turning its hidden states into sample-specific **synolitic graphs**
(ensembles of pairwise classifiers → graph topology), (2) training a **GNN**
to predict whether the model erred, and (3) building an **AI error corrector**
with distribution-agnostic performance bounds (Tyukin et al., 2024).
Setup 1 (this repo, pre-defense target): a small from-scratch Transformer
(2+2 layers, 4 heads, d_model=64, ~236K params) on the sequence-reversal task.

## Pipeline & ownership

```
stage 1 (role 1)          stage 2 (role 2)          stage 3 (role 3)         stage 4 (role 4)
transformer + extraction  synolitic graphs          GNN error classifier     corrector + bounds
        │                        │                         │                        │
        └── hidden_states.pt ────┴──── graphs.pt ──────────┴──── scores.pt ─────────┘
```

Each stage develops against **dummy artifacts** with identical schemas, so all
four stages proceed in parallel; swapping in the real files needs no code
changes. The schemas and their validators live in
[`src/synolitic/common/schemas.py`](src/synolitic/common/schemas.py) — that
file is the single source of truth. Validate everything you produce _and_
consume:

```python
from synolitic.common import schemas
schemas.assert_valid("graphs", schemas.validate_graphs(artifact))
```

## Quickstart

```bash
uv sync                                        # env (torch, torch-geometric, sklearn, networkx)
uv run pytest -q                               # contract + model tests
uv run python scripts/make_dummy_data.py       # artifacts/dummy/{hidden_states,graphs,scores}.pt
uv run python scripts/run_corrector.py         # stage 4 demo: Theorem-1 bounds vs empirical rates
# stage 1 (real data) — three passes; reproduces artifacts/hidden_states.pt bit-for-bit:
#   1a  deliberate undertraining — default settings stall at ~0.046 exact-match
#       (step 5000); this pass only seeds the warm start.
uv run python scripts/train_model.py --out-dir artifacts/checkpoints_pretrain
#   1b  warm start (higher LR + fine eval grid) catches the 0.70–0.85 band at
#       pass-1b step 2900 (cumulative 7900), val exact-match 0.766 -> artifacts/checkpoints/
uv run python scripts/train_model.py \
    --init-from artifacts/checkpoints_pretrain/model_best.pt \
    --lr 1e-3 --steps 4000 --eval-every 50
#   1c  extract — N=10000, model accuracy 0.724, splits 6000/2000/2000
uv run python scripts/extract_hidden_states.py # -> artifacts/hidden_states.pt
# stage 4 on real artifacts:
uv run python scripts/run_corrector.py \
    --scores artifacts/scores.pt \
    --graphs artifacts/graphs.pt \
    --hidden-states artifacts/hidden_states.pt
```

Artifacts are git-ignored; real files are shared via GitHub release assets.

## Split discipline (leakage protocol)

`split` codes: `0=train`, `1=cal`, `2=test` (60/20/20). **This is the one rule
that keeps the theoretical bounds honest:**

| split   | who uses it                                                        |
| ------- | ------------------------------------------------------------------ |
| `train` | stage 2 fits the 2016 pairwise classifiers; stage 3 trains the GNN |
| `cal`   | stage 4 fits the Fisher projector and calibrates thresholds Δ      |
| `test`  | final reported numbers only — never fit/calibrate on it            |

Model training data is generated on the fly with seeds disjoint from the
extraction set, so it never overlaps any split.

## Data contracts (v1.1)

### `hidden_states.pt` (stage 1 → 2) — dict of tensors + `meta`

| key             | shape / dtype                   | meaning                                                              |
| --------------- | ------------------------------- | -------------------------------------------------------------------- |
| `X`             | `[N, 64] float32`               | mean over non-PAD source positions of the final-LN encoder output    |
| `X_layers`      | `[N, 2, 64] float32`            | same pooling per encoder layer (optional, for layer analysis)        |
| `input_tokens`  | `[N, Ls] int64`                 | source digits, PAD-padded                                            |
| `target_tokens` | `[N, Ls] int64`                 | reversed digits, PAD-padded                                          |
| `pred_tokens`   | `[N, T] int64`                  | greedy output incl. EOS, PAD after it                                |
| `lengths`       | `[N] int64`                     | true sequence lengths                                                |
| `is_correct`    | `[N] int8`                      | exact match of the full sequence (`schemas.exact_match`)             |
| `confidence`    | `[N] float32`                   | mean max-softmax over generated steps (baseline for stage 4)         |
| `split`         | `[N] int8`                      | 0/1/2, see split discipline                                          |
| `attn_idx`      | `[N_sub] int64`                 | rows covered by the attention dump                                   |
| `enc_self_attn` | `[N_sub, 2, 4, Ls, Ls] float32` | per-layer, per-head attention (H3 analysis + raw-attention baseline) |
| `dec_self_attn` | `[N_sub, 2, 4, T, T] float32`   |                                                                      |
| `cross_attn`    | `[N_sub, 2, 4, T, Ls] float32`  |                                                                      |

Vocabulary (also part of the contract): `PAD=0, BOS=1, EOS=2`, digit _d_ ↦
token `d+3`. Class balance is ~4:1 (model held at ~70–85% accuracy) — use
class weights and report ROC-AUC.

### `graphs.pt` (stage 2 → 3) — `{"graphs": list[torch_geometric.data.Data], "meta": dict}`

Each `Data`: `x [64, 5] float32` (node features `[value, degree, strength,
closeness, betweenness]`), `edge_index [2, E] int64` (undirected ⇒ include
both directions), `edge_attr [E] float32` (pairwise-classifier scores),
`y [1] float32` (= is_correct), `idx [1] int64` (row in hidden_states.pt),
`split [1] int8`. Pairwise classifiers are fitted on the train split only.

### `scores.pt` (stage 3 → 4) — dict

`scores [N] float32` = GNN estimate of P(is_correct=1), aligned by index with
`hidden_states.pt`; `is_correct [N] int8`; `split [N] int8`; `meta`.
Stage 4 may additionally consume topological features straight from
`graphs.pt` (the paper-faithful Φ).

**Changes vs the v1.0 chat proposal:** added `split` (leakage protocol),
`confidence` + attention dumps (baselines and H3 need them), replaced scalar
`y_pred` with `pred_tokens [N, T]` (reversal outputs are sequences), graphs
carry `idx`/`split`, scores fixed to "probability the model is correct".

## Layout

```
src/synolitic/
  common/           schemas + validators, artifact IO, dummy generators
  stage1_model/     reversal data, transformer, training, extraction   (role 1)
  stage2_graphs/    pairwise ensemble -> synolitic graphs              (role 2)
  stage3_gnn/       GCN/GATv2 error classifier                        (role 3)
  stage4_corrector/ Fisher discriminant, thresholds, bounds            (role 4)
scripts/            make_dummy_data / train_model / extract_hidden_states
tests/              contract tests + stage-1 smoke tests
```

## Timeline (pre-defense: July 12, 23:00 UTC+3)

| date    | milestone                                                             |
| ------- | --------------------------------------------------------------------- |
| Jul 5   | roles fixed, contracts v1.1, repo + dummies + contract tests          |
| Jul 6–8 | parallel development on dummies; real `hidden_states.pt` by Jul 8 EOD |
| Jul 9   | stages 2–4 switch to real data                                        |
| Jul 10  | integration day: full pipeline run, metrics table, figures            |
| Jul 11  | presentation draft + rehearsal                                        |
| Jul 12  | polish, submit                                                        |

Baselines for the results table: softmax-confidence rejection, logistic
regression / MLP on raw `X`, GNN on synolitic graphs, corrector on synolitic
features vs corrector on raw features (hypothesis H1).

## Results

Final pre-defense numbers live in [docs/results.md](docs/results.md).
Headlines: GNN on synolitic graphs reaches test AUC **0.839** vs 0.799 for
the softmax-confidence baseline; as a corrector at Δ=0.8 it keeps 55.9% of
answers at **91.1% precision** (base accuracy 72.4%) rejecting 82.1% of
errors; **all Theorem-1 bounds hold empirically** on the untouched test split.

## References

- Zaikin et al., _Overcoming the Curse of Dimensionality with Synolitic AI_, Technologies 14(2):84, 2026.
- Tyukin et al., _Coping with AI Errors with Provable Guarantees_, Information Sciences 678:120856, 2024.
- Tyukin et al., _Weakly Supervised Learners for Correction of AI Errors with Provable Performance Guarantees_, IJCNN 2024.
- Clark et al., _What Does BERT Look At?_, BlackboxNLP 2019.
- Vaswani et al., _Attention Is All You Need_, NeurIPS 2017.
