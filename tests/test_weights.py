"""safetensors export / load round trip."""
import torch

from transformer_shenshu.model import ModelConfig, ShenShuModel
from transformer_shenshu.weights import export_safetensors, load_model


def test_safetensors_roundtrip(tmp_path):
    cfg = ModelConfig(n_bars=64, d_model=32, nhead=2, dim_feedforward=64, num_encoder_layers=1)
    m = ShenShuModel(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"model": m.state_dict(), "config": cfg.__dict__, "step": 7, "val_bar_acc": 0.5}, ckpt)

    for dtype in (torch.float32, torch.float16):
        out = tmp_path / f"m-{dtype}.safetensors"
        export_safetensors(ckpt, out, dtype)
        loaded, info = load_model(out)
        assert info["step"] == 7
        assert loaded.cfg.n_bars == 64
        ref = m.state_dict()
        for k, v in loaded.state_dict().items():
            tol = 0 if dtype == torch.float32 else 1e-2
            assert torch.allclose(v, ref[k].float(), atol=tol), k

    pt_model, info = load_model(ckpt)  # weights_only=True path
    assert info["step"] == 7
