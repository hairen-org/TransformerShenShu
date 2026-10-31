"""Birth-date parsing and 四柱 conversion."""
import datetime as _dt

import pytest

from transformer_shenshu.bazi import (
    BirthError,
    chart_from_birth,
    equation_of_time,
    parse_birth,
    parse_longitude,
    parse_time,
    true_solar_time,
    validate_pillar,
)


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


@pytest.mark.parametrize(
    "text, value",
    [
        ("116.4", 116.4),
        ("116.4E", 116.4),
        ("东经116°24′", 116.4),
        ("116度24分", 116.4),
        ("西经74", -74.0),
        ("74W", -74.0),
        ("-74.5", -74.5),
    ],
)
def test_parse_longitude(text, value):
    assert parse_longitude(text) == pytest.approx(value)


@pytest.mark.parametrize("bad", ["abc", "190", "116°75′"])
def test_parse_longitude_invalid(bad):
    with pytest.raises(BirthError):
        parse_longitude(bad)


def test_equation_of_time_extremes():
    # well-known values: ≈ +16.4 min early November, ≈ −14.2 min mid February
    assert equation_of_time(_dt.datetime(2023, 11, 3, 12)) == pytest.approx(16.4, abs=0.6)
    assert equation_of_time(_dt.datetime(2023, 2, 11, 12)) == pytest.approx(-14.2, abs=0.6)


def test_true_solar_time_on_standard_meridian_is_only_eot():
    fix = true_solar_time(_dt.datetime(2023, 11, 3, 12), 120.0)
    assert fix.longitude_min == 0
    assert (fix.true_solar - fix.clock).total_seconds() / 60 == pytest.approx(fix.eot_min)


def test_correction_can_cross_midnight():
    # 00:30 Beijing time at 87.6°E → previous evening (≈ 22:21) → day pillar and 亥时 change
    plain = chart_from_birth(parse_birth("S2000-01-01"), 0, 30)
    fixed = chart_from_birth(parse_birth("S2000-01-01"), 0, 30, longitude=87.6)
    assert fixed.solar.startswith("1999-12-31 22:")
    assert plain.day == "戊午" and fixed.day == "丁巳"
    assert fixed.hour[1] == "亥"


def test_lunar_input_with_correction():
    fixed = chart_from_birth(parse_birth("C2000-01-01"), 0, 10, longitude=87.6)
    # 农历正月初一 = 2000-02-05 00:10 → −129.6 min (longitude) −14 min (EoT) ≈ 2000-02-04 21:47
    assert fixed.solar == "2000-02-04 21:47"
    assert fixed.lunar.startswith("一九九九年腊月")  # crossed back into the previous lunar day


def test_validate_pillar():
    assert validate_pillar("丙寅") == "丙寅"
    with pytest.raises(BirthError):
        validate_pillar("甲丑")
