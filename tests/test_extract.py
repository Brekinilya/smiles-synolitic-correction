"""The extraction pipeline must produce a contract-valid artifact even from an
untrained model (correctness labels are then mostly 0 — that is fine)."""

from synolitic.common import schemas
from synolitic.common.io import load_artifact, save_artifact
from synolitic.stage1_model.extract import ExtractConfig, extract
from synolitic.stage1_model.model import ModelConfig, ReversalTransformer


def test_extraction_produces_valid_artifact(tmp_path):
    model = ReversalTransformer(ModelConfig())
    cfg = ExtractConfig(
        n=120, batch_size=32, min_len=4, max_len=6, seed=7, attn_subsample=10,
    )
    artifact = extract(model, cfg)
    assert schemas.validate_hidden_states(artifact) == []

    assert artifact["X"].shape == (120, schemas.D_H)
    assert artifact["X_layers"].shape == (120, 2, schemas.D_H)
    assert artifact["enc_self_attn"].shape[:3] == (10, 2, 4)

    reloaded = load_artifact(save_artifact(artifact, tmp_path / "hs.pt"))
    assert schemas.validate_hidden_states(reloaded) == []
