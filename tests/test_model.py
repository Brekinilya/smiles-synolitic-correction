import torch

from synolitic.common.schemas import VOCAB_SIZE
from synolitic.stage1_model.dataset import make_dataset, make_targets
from synolitic.stage1_model.model import ModelConfig, ReversalTransformer
from synolitic.stage1_model.train import TrainConfig, train


def test_parameter_count_matches_proposal():
    model = ReversalTransformer(ModelConfig())
    assert 200_000 < model.param_count() < 280_000  # proposal says ~236K


def test_dataset_reversal_is_correct():
    g = torch.Generator().manual_seed(0)
    src, tgt, lengths = make_dataset(16, 3, 7, g)
    for i in range(16):
        n = int(lengths[i])
        assert torch.equal(tgt[i, :n], src[i, :n].flip(0))
        assert (src[i, n:] == 0).all() and (tgt[i, n:] == 0).all()


def test_forward_and_greedy_decode_shapes():
    g = torch.Generator().manual_seed(0)
    src, tgt, lengths = make_dataset(8, 4, 6, g)
    tgt_in, tgt_out = make_targets(tgt, lengths)
    model = ReversalTransformer(ModelConfig())
    model.eval()

    logits = model(src, tgt_in)
    assert logits.shape == (8, 7, VOCAB_SIZE)
    assert tgt_out.shape == (8, 7)

    pred, conf, memory, src_kpm, hiddens = model.greedy_decode(src, max_new_tokens=7)
    assert pred.shape[0] == 8 and pred.shape[1] <= 7
    assert conf.shape == (8,) and conf.min() > 0 and conf.max() <= 1
    assert memory.shape == (8, 6, 64)
    assert len(hiddens) == 2 and src_kpm.shape == (8, 6)


def test_short_training_reduces_loss(tmp_path):
    cfg = TrainConfig(
        steps=300, batch_size=64, lr=1e-3, min_len=3, max_len=6,
        eval_every=150, val_size=128, target_acc=0.99, seed=0,
        out_dir=str(tmp_path),
    )
    result = train(cfg)
    history = result["history"]
    assert history[-1]["loss"] < history[0]["loss"] * 0.7
