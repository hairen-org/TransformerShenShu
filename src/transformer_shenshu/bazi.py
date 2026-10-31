"""Birth date → 四柱八字 (节气-based), using lunar-python.

Date syntax:
    S2000-01-01        阳历（公历）
    C2000-01-01        中国农历
    C2023-闰02-15      农历闰月（也可写 C2023-L02-15）
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

GAN = "甲乙丙丁戊己庚辛壬癸"
ZHI = "子丑寅卯辰巳午未申酉戌亥"
SHICHEN = {z: (i * 2) % 24 for i, z in enumerate(ZHI)}  # 子=0:00, 丑=2:00, …, 亥=22:00
JIAZI = [GAN[i % 10] + ZHI[i % 12] for i in range(60)]

YEAR_MIN, YEAR_MAX = 1900, 2100
_DATE_RE = re.compile(r"^\s*([SsCc])\s*(\d{4})\s*[-/.]\s*(闰|[Ll])?\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*$")


class BirthError(ValueError):
    """Human-readable input error."""


@dataclass
class Birth:
    calendar: str  # "S" or "C"
    year: int
    month: int
    day: int
    leap: bool = False

    def label(self) -> str:
        if self.calendar == "S":
            return f"阳历 {self.year}-{self.month:02d}-{self.day:02d}"
        return f"农历 {self.year}-{'闰' if self.leap else ''}{self.month:02d}-{self.day:02d}"


@dataclass
class Chart:
    year: str
    month: str
    day: str
    hour: str
    solar: str
    lunar: str
    zodiac: str

    @property
    def pillars(self) -> str:
        return f"{self.year} {self.month} {self.day} {self.hour}"


def parse_birth(text: str) -> Birth:
    m = _DATE_RE.match(text or "")
    if not m:
        raise BirthError("格式不对。阳历写 S2000-01-01，农历写 C2000-01-01，农历闰月写 C2023-闰02-15")
    cal = m.group(1).upper()
    year, month, day = int(m.group(2)), int(m.group(4)), int(m.group(5))
    leap = bool(m.group(3))
    if not YEAR_MIN <= year <= YEAR_MAX:
        raise BirthError(f"年份需在 {YEAR_MIN}–{YEAR_MAX} 之间")
    if cal == "S":
        if leap:
            raise BirthError("阳历没有闰月；闰月只用于农历（C 开头）")
        try:
            _dt.date(year, month, day)
        except ValueError:
            raise BirthError(f"阳历 {year}-{month:02d}-{day:02d} 不存在") from None
        return Birth("S", year, month, day)

    from lunar_python import LunarYear

    months = {m.getMonth(): m.getDayCount() for m in LunarYear.fromYear(year).getMonthsInYear()}
    key = -month if leap else month
    if key not in months:
        if leap:
            leap_m = LunarYear.fromYear(year).getLeapMonth()
            hint = f"该年闰{leap_m}月" if leap_m else "该年没有闰月"
            raise BirthError(f"农历 {year} 年没有闰{month}月（{hint}）")
        raise BirthError("农历月份需在 1–12 之间")
    if not 1 <= day <= months[key]:
        raise BirthError(f"农历 {year} 年{'闰' if leap else ''}{month}月只有 {months[key]} 天")
    return Birth("C", year, month, day, leap)


def parse_time(text: str) -> tuple[int, int]:
    """'10:30' / '10' / '午' / '午时' → (hour, minute)."""
    t = (text or "").strip().rstrip("时時")
    if len(t) == 1 and t in SHICHEN:
        return SHICHEN[t], 0
    m = re.match(r"^(\d{1,2})(?:[:：](\d{1,2}))?$", t)
    if not m:
        raise BirthError("时间写 24 小时制，如 10:30 或 23；也可写时辰，如 午 / 午时")
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        raise BirthError("时间超出范围：小时 0–23，分钟 0–59")
    return h, mi


def chart_from_birth(b: Birth, hour: int, minute: int) -> Chart:
    from lunar_python import Lunar, Solar

    if b.calendar == "S":
        lunar = Solar.fromYmdHms(b.year, b.month, b.day, hour, minute, 0).getLunar()
    else:
        lunar = Lunar.fromYmdHms(b.year, -b.month if b.leap else b.month, b.day, hour, minute, 0)
    ec = lunar.getEightChar()
    return Chart(
        year=ec.getYear(),
        month=ec.getMonth(),
        day=ec.getDay(),
        hour=ec.getTime(),
        solar=lunar.getSolar().toYmd() + f" {hour:02d}:{minute:02d}",
        lunar=lunar.toString(),
        zodiac=lunar.getYearShengXiaoByLiChun(),
    )


def validate_pillar(text: str) -> str:
    p = (text or "").strip()
    if p not in JIAZI:
        raise BirthError("需为六十甲子中的一柱，如 甲子、丙寅（天干地支阴阳要相配，如没有“甲丑”）")
    return p
