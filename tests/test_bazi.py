"""Birth-date parsing and 四柱 conversion."""
import pytest

from transformer_shenshu.bazi import BirthError, chart_from_birth, parse_birth, parse_time, validate_pillar


def test_solar_before_lichun_uses_previous_year_pillar():
    ch = chart_from_birth(parse_birth("S2000-01-01"), 10, 30)
    assert ch.pillars == "己卯 丙子 戊午 丁巳"  # 立春前仍属己卯年
    assert ch.zodiac == "兔"


def test_lunar_new_year_maps_to_solar():
    ch = chart_from_birth(parse_birth("C2000-01-01"), 12, 0)
    assert ch.solar.startswith("2000-02-05")
    assert (ch.year, ch.month) == ("庚辰", "戊寅")


@pytest.mark.parametrize("text", ["C2023-闰02-15", "c2023-L2-15", "C2023/闰02/15"])
def test_leap_month_syntax(text):
    b = parse_birth(text)
    assert (b.calendar, b.year, b.month, b.day, b.leap) == ("C", 2023, 2, 15, True)
    assert chart_from_birth(b, 0, 0).solar.startswith("2023-04-05")


@pytest.mark.parametrize(
    "text, msg",
    [
        ("2000-01-01", "格式不对"),
        ("S2023-02-29", "不存在"),
        ("S2023-闰02-01", "阳历没有闰月"),
        ("C2023-03-30", "只有 29 天"),
        ("C2022-闰02-01", "没有闰2月"),
        ("C1850-01-01", "1900–2100"),
        ("C2023-13-01", "1–12"),
    ],
)
def test_invalid_dates(text, msg):
    with pytest.raises(BirthError, match=msg):
        parse_birth(text)


def test_time_parsing():
    assert parse_time("10:30") == (10, 30)
    assert parse_time("23") == (23, 0)
    assert parse_time("午时") == (12, 0)
    assert parse_time("子") == (0, 0)
    for bad in ("25:00", "10:61", "noon"):
        with pytest.raises(BirthError):
            parse_time(bad)


def test_validate_pillar():
    assert validate_pillar("丙寅") == "丙寅"
    with pytest.raises(BirthError):
        validate_pillar("甲丑")
