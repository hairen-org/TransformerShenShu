"""Tests for Shenshu-42M-v1 hybrid bar heads."""
from transformer_shenshu.model import ModelConfig, ShenShuModel
import torch


def test_bar_forward_cpu():
    cfg = ModelConfig(n_bars=128, num_decoder_layers=0, use_vq=False, use_lm=False)
    m = ShenShuModel(cfg)
    B = 2
    batch = {
        "sex": torch.ones(B, dtype=torch.long),
        "year_gan": torch.ones(B, dtype=torch.long),
        "year_zhi": torch.ones(B, dtype=torch.long),
        "month_gan": torch.ones(B, dtype=torch.long),
        "month_zhi": torch.ones(B, dtype=torch.long),
        "day_gan": torch.ones(B, dtype=torch.long),
        "day_zhi": torch.ones(B, dtype=torch.long),
        "hour_gan": torch.ones(B, dtype=torch.long),
        "hour_zhi": torch.ones(B, dtype=torch.long),
        "father_zodiac": torch.ones(B, dtype=torch.long),
        "mother_zodiac": torch.ones(B, dtype=torch.long),
        "brothers": torch.zeros(B, dtype=torch.long),
        "spouse_zodiac": torch.ones(B, dtype=torch.long),
        "sons": torch.zeros(B, dtype=torch.long),
        "daughters": torch.zeros(B, dtype=torch.long),
        "verse_ids": torch.randint(0, 128, (B, 6)),
    }
    out = m(batch)
    assert out["bar_logits"].shape == (B, 6, 128)
    assert out["loss_bar"].ndim == 0
    assert 0.0 <= float(out["bar_acc"]) <= 1.0
    assert "loss_true_ke" in out
    assert out["true_ke_logits"].shape == (B, 8)
    preds = m.predict_verse_ids(batch)
    assert preds.shape == (B, 6)


def test_v01_forward_still_works_with_vq_lm():
    cfg = ModelConfig(vocab_size=64, max_target_len=32, code_length=4, codebook_size=32,
                      use_vq=True, use_lm=True, num_decoder_layers=2, n_bars=32)
    m = ShenShuModel(cfg)
    B, T = 2, 8
    batch = {
        "sex": torch.zeros(B, dtype=torch.long),
        "year_gan": torch.ones(B, dtype=torch.long),
        "year_zhi": torch.ones(B, dtype=torch.long),
        "month_gan": torch.ones(B, dtype=torch.long),
        "month_zhi": torch.ones(B, dtype=torch.long),
        "day_gan": torch.ones(B, dtype=torch.long),
        "day_zhi": torch.ones(B, dtype=torch.long),
        "hour_gan": torch.ones(B, dtype=torch.long),
        "hour_zhi": torch.ones(B, dtype=torch.long),
        "father_zodiac": torch.ones(B, dtype=torch.long),
        "mother_zodiac": torch.ones(B, dtype=torch.long),
        "brothers": torch.zeros(B, dtype=torch.long),
        "spouse_zodiac": torch.ones(B, dtype=torch.long),
        "sons": torch.zeros(B, dtype=torch.long),
        "daughters": torch.zeros(B, dtype=torch.long),
        "tgt_in": torch.randint(0, 64, (B, T)),
        "tgt_out": torch.randint(0, 64, (B, T)),
        "verse_ids": torch.randint(0, 32, (B, 6)),
    }
    out = m(batch)
    assert out["bar_logits"].shape == (B, 6, 32)
    assert out["logits"].shape == (B, T, 64)
