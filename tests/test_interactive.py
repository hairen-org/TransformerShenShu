"""Interactive CLI flow with scripted input (tiny random model)."""
import builtins

from transformer_shenshu.cli import interactive as ui
from transformer_shenshu.model import ModelConfig, ShenShuModel


def _feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *_: next(it))


def _model():
    return ShenShuModel(ModelConfig(n_bars=64, d_model=32, nhead=2, dim_feedforward=64, num_encoder_layers=1)).eval()


def test_full_profile_gives_single_ke(monkeypatch):
    # longitude skipped → Beijing time, no correction
    _feed(monkeypatch, ["男", "S1990-05-15", "10:30", "", "猴", "兔", "2", "马", "1", ""])
    p = ui.collect_profile()
    assert p.pillars == ("庚午", "辛巳", "庚辰", "辛巳")
    assert p.qin == {"father": 9, "mother": 4, "brothers": 2, "spouse": 7, "sons": 1}
    ke, rows = ui.predict(_model(), p)
    assert ke is not None and len(rows) == 1 and len(rows[0]) == 6


def test_longitude_correction_changes_hour_pillar(monkeypatch):
    # 10:30 Beijing time in Urumqi (87.6°E) ≈ 08:24 true solar time → 辰时 instead of 巳时
    _feed(monkeypatch, ["男", "S1990-05-15", "10:30", "abc", "87.6", "", "", "", "", "", ""])
    p = ui.collect_profile()
    assert p.pillars == ("庚午", "辛巳", "庚辰", "庚辰")


def test_skipped_time_skips_longitude(monkeypatch):
    # time skipped → step 4 is not asked; the next answer goes to 父亲生肖
    _feed(monkeypatch, ["女", "C1990-04-21", "", "猴", "", "", "", "", ""])
    p = ui.collect_profile()
    assert p.qin == {"father": 9}


def test_skipped_family_lists_eight_ke(monkeypatch):
    _feed(monkeypatch, ["bad", "女", "P", "丙寅", "癸未", "壬戌", "辛亥", "", "兔", "9", "", "", "", ""])
    p = ui.collect_profile()
    assert p.sex == "F" and p.pillars == ("丙寅", "癸未", "壬戌", "辛亥")
    assert p.qin == {"mother": 4}  # "9" brothers rejected, then skipped
    ke, rows = ui.predict(_model(), p)
    assert ke is None and len(rows) == 8
