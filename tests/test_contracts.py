"""Contract tests: dummy artifacts must satisfy the same validators the real
artifacts will be checked against — this is what makes dummy/real swap safe."""

import torch

from synolitic.common import dummy, schemas
from synolitic.common.io import load_artifact, save_artifact


def test_exact_match_semantics():
    eos, pad = schemas.EOS, schemas.PAD
    d = schemas.FIRST_DIGIT
    target = torch.tensor([[d + 5, d + 4, d + 3, pad]])
    lengths = torch.tensor([3])

    ok = torch.tensor([[d + 5, d + 4, d + 3, eos, pad]])
    assert schemas.exact_match(ok, target, lengths).tolist() == [True]

    early_eos = torch.tensor([[d + 5, d + 4, eos, pad, pad]])
    wrong_digit = torch.tensor([[d + 5, d + 4, d + 9, eos, pad]])
    no_eos = torch.tensor([[d + 5, d + 4, d + 3, d + 7, d + 1]])
    for bad in (early_eos, wrong_digit, no_eos):
        assert schemas.exact_match(bad, target, lengths).tolist() == [False]


def test_dummy_hidden_states_pass_contract():
    art = dummy.dummy_hidden_states(n=128, attn_subsample=8, seed=0)
    assert schemas.validate_hidden_states(art) == []


def test_hidden_states_validator_catches_corruption():
    art = dummy.dummy_hidden_states(n=64, attn_subsample=8, seed=0)
    art["X"] = art["X"].double()
    assert any("dtype" in e for e in schemas.validate_hidden_states(art))

    art = dummy.dummy_hidden_states(n=64, attn_subsample=8, seed=1)
    art["is_correct"] = 1 - art["is_correct"]
    assert any("exact match" in e for e in schemas.validate_hidden_states(art))

    art = dummy.dummy_hidden_states(n=64, attn_subsample=8, seed=2)
    del art["confidence"]
    assert any("confidence" in e for e in schemas.validate_hidden_states(art))


def test_dummy_graphs_pass_contract():
    hs = dummy.dummy_hidden_states(n=48, attn_subsample=4, seed=0)
    art = dummy.dummy_graphs(hidden_states=hs, seed=0)
    assert schemas.validate_graphs(art) == []
    assert len(art["graphs"]) == 48
    g0 = art["graphs"][0]
    assert g0.x.shape == (schemas.D_H, schemas.N_NODE_FEATURES)


def test_graphs_validator_catches_corruption():
    art = dummy.dummy_graphs(n=8, seed=0)
    art["graphs"][3].x = art["graphs"][3].x[:, :3]
    assert any("graph[3]" in e for e in schemas.validate_graphs(art))


def test_dummy_scores_pass_contract():
    art = dummy.dummy_scores(n=256, seed=0)
    assert schemas.validate_scores(art) == []
    art["scores"] = art["scores"] * 2
    assert any("[0, 1]" in e for e in schemas.validate_scores(art))


def test_artifacts_roundtrip_through_disk(tmp_path):
    hs = dummy.dummy_hidden_states(n=32, attn_subsample=4, seed=0)
    assert schemas.validate_hidden_states(load_artifact(save_artifact(hs, tmp_path / "h.pt"))) == []
    gr = dummy.dummy_graphs(hidden_states=hs, seed=0)
    assert schemas.validate_graphs(load_artifact(save_artifact(gr, tmp_path / "g.pt"))) == []
    sc = dummy.dummy_scores(hidden_states=hs, seed=0)
    assert schemas.validate_scores(load_artifact(save_artifact(sc, tmp_path / "s.pt"))) == []
