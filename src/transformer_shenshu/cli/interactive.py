"""Interactive CLI: ``shenshu``.

Step-by-step prompts → 四柱（阳历 S / 农历 C 自动换算）+ 六亲 → 六门命书.
When 父 / 母 / 兄弟 / 配偶 are not all known, the true 刻 cannot be fixed, so the
eight candidate 刻 are listed (like the classical 考刻 step).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from transformer_shenshu import __version__
from transformer_shenshu.bazi import (
    BirthError,
    chart_from_birth,
    parse_birth,
    parse_time,
    validate_pillar,
)
from transformer_shenshu.cli.demo import SECTIONS, VERSE_OFFSET, ZODIAC_CN, load_lut, zodiac_index
from transformer_shenshu.data.dataset import parse_pillar
from transformer_shenshu.weights import RELEASE_TAG, VARIANTS, cache_dir, download, load_model

BANNER = r"""
  ______                      ____
 /_  __/________ _____  _____/ __/___  _________ ___  ___  _____
  / / / ___/ __ `/ __ \/ ___/ /_/ __ \/ ___/ __ `__ \/ _ \/ ___/
 / / / /  / /_/ / / / (__  ) __/ /_/ / /  / / / / / /  __/ /
/_/ /_/   \__,_/_/ /_/____/_/  \____/_/  /_/ /_/ /_/\___/_/
                 _____ __                  _____ __
                / ___// /_  ___  ____     / ___// /_  __  __
                \__ \/ __ \/ _ \/ __ \    \__ \/ __ \/ / / /
               ___/ / / / /  __/ / / /   ___/ / / / / /_/ /
              /____/_/ /_/\___/_/ /_/   /____/_/ /_/\__,_/
"""

ZODIAC_HELP = "鼠 牛 虎 兔 龙 蛇 马 羊 猴 鸡 狗 猪（也可写 属鼠 或英文 rat / ox / tiger …）"
KE_LABELS = ["第一刻", "第二刻", "第三刻", "第四刻", "第五刻", "第六刻", "第七刻", "第八刻"]
SKIP = object()


class Quit(Exception):
    pass


# ---------------------------------------------------------------- terminal helpers
def _setup_console() -> bool:
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    if color and os.name == "nt":
        os.system("")  # enable ANSI escape handling on Windows consoles
    return color


COLOR = False


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def ask(step: str, title: str, lines: list[str], *, optional: bool, parse, default_note: str = ""):
    """Prompt until ``parse`` accepts the answer. Returns SKIP when skipped."""
    tag = c("可跳过", "32") if optional else c("必填", "31")
    print()
    print(c(f"[{step}] {title}", "1;36") + f"  （{tag}）")
    for line in lines:
        print("   " + line)
    if optional:
        print("   " + c(f"直接回车跳过{('：' + default_note) if default_note else ''}", "2"))
    while True:
        try:
            raw = input(c("   > ", "1;33")).strip()
        except EOFError:
            raise Quit from None
        if raw.lower() in ("q", "quit", "exit"):
            raise Quit
        if not raw:
            if optional:
                return SKIP
            print(c("   这一项必填，请输入（输入 q 退出）", "31"))
            continue
        try:
            return parse(raw)
        except BirthError as e:
            print(c(f"   ✗ {e}", "31"))


# ---------------------------------------------------------------- parsers
def parse_sex(raw: str) -> str:
    v = raw.strip().lower()
    if v in ("男", "m", "male", "1"):
        return "M"
    if v in ("女", "f", "female", "2"):
        return "F"
    raise BirthError("请输入 男 / 女（或 M / F）")


def parse_zodiac(raw: str) -> int:
    i = zodiac_index(raw)
    if not i:
        raise BirthError("不认识这个生肖。可填：" + ZODIAC_HELP)
    return i


def int_parser(lo: int, hi: int, what: str):
    def _p(raw: str) -> int:
        if not raw.isdigit() or not lo <= int(raw) <= hi:
            raise BirthError(f"{what}请填 {lo}–{hi} 的整数")
        return int(raw)

    return _p


def parse_birth_or_manual(raw: str):
    if raw.strip().upper() == "P":
        return "P"
    return parse_birth(raw)


# ---------------------------------------------------------------- session
@dataclass
class Profile:
    sex: str = "M"
    pillars: tuple[str, str, str, str] = ("", "", "", "")
    chart_note: str = ""
    qin: dict[str, int] = field(default_factory=dict)  # father/mother/spouse (1..12), brothers/sons/daughters


def collect_profile() -> Profile:
    p = Profile()
    p.sex = ask(
        "1/9", "性别", ["填 男 或 女（也可写 M / F）"], optional=False, parse=parse_sex
    )

    birth = ask(
        "2/9",
        "出生日期",
        [
            "阳历（公历）：S + 日期，例如 S1990-05-15",
            "中国农历：  C + 日期，例如 C1990-04-21；闰月写 C2023-闰02-15（或 C2023-L02-15）",
            "支持 1900–2100 年。若已知四柱八字、想直接输入，填 P",
        ],
        optional=False,
        parse=parse_birth_or_manual,
    )

    if birth == "P":
        names = ["年柱", "月柱", "日柱", "时柱"]
        vals = []
        for i, n in enumerate(names):
            vals.append(
                ask(
                    f"2.{i + 1}/9",
                    n,
                    ["填六十甲子中的一柱，如 甲子、丙寅、癸亥"],
                    optional=False,
                    parse=validate_pillar,
                )
            )
        p.pillars = tuple(vals)
        p.chart_note = "（手动输入四柱）"
    else:
        t = ask(
            "3/9",
            "出生时间",
            [
                "24 小时制，例如 10:30、23:15；或填时辰，例如 午 / 午时",
                "用于排日柱、时柱；节气交界附近的出生时间也会影响年柱、月柱",
            ],
            optional=True,
            parse=parse_time,
            default_note="按午时 12:00 计。本模型的六门条号只取决于年柱、月柱和六亲，时辰不影响结果",
        )
        hour, minute = (12, 0) if t is SKIP else t
        chart = chart_from_birth(birth, hour, minute)
        p.pillars = (chart.year, chart.month, chart.day, chart.hour)
        print()
        print("   " + c("✓ 排盘结果", "1;32"))
        print(f"     输入：{birth.label()}  {hour:02d}:{minute:02d}")
        print(f"     阳历：{chart.solar}    农历：{chart.lunar}")
        print(f"     四柱：{c(chart.pillars, '1')}    生肖（以立春为界）：{chart.zodiac}")

    zodiac_lines = ["可填：" + ZODIAC_HELP]
    ke_note = "将无法唯一确定真刻，结果改为列出八刻候选"
    for step, key, title in (("4/9", "father", "父亲生肖"), ("5/9", "mother", "母亲生肖")):
        v = ask(step, title, zodiac_lines, optional=True, parse=parse_zodiac, default_note=ke_note)
        if v is not SKIP:
            p.qin[key] = v
    v = ask(
        "6/9",
        "兄弟人数",
        ["不含自己，填 0–5 的整数（模型训练范围为 0–5）"],
        optional=True,
        parse=int_parser(0, 5, "兄弟人数"),
        default_note=ke_note,
    )
    if v is not SKIP:
        p.qin["brothers"] = v
    v = ask(
        "7/9",
        "配偶生肖",
        ["可填：" + ZODIAC_HELP, "未婚或不确定可跳过"],
        optional=True,
        parse=parse_zodiac,
        default_note=ke_note,
    )
    if v is not SKIP:
        p.qin["spouse"] = v
    for step, key, title in (("8/9", "sons", "儿子人数"), ("9/9", "daughters", "女儿人数")):
        v = ask(
            step,
            title,
            ["填 0–3 的整数（模型训练范围为 0–3）"],
            optional=True,
            parse=int_parser(0, 3, title),
            default_note="按 0 计，不影响六门条号",
        )
        if v is not SKIP:
            p.qin[key] = v
    return p


# ---------------------------------------------------------------- model
def resolve_checkpoint(arg: str) -> Path:
    for cand in (arg, os.environ.get("SHENSHU_CHECKPOINT", "")):
        if cand:
            if not Path(cand).is_file():
                raise SystemExit(f"找不到权重文件：{cand}")
            return Path(cand)
    for variant in ("fp16", "fp32"):
        cached = cache_dir() / VARIANTS[variant]
        if cached.is_file():
            return cached
    print()
    print(c("未找到模型权重。", "33") + f" 将从 GitHub Release {RELEASE_TAG} 下载 fp16 权重（约 40 MB，校验 SHA256）。")
    ans = input(c("   是否下载？[Y/n] > ", "1;33")).strip().lower()
    if ans not in ("", "y", "yes", "是"):
        raise SystemExit("已取消。也可用 --checkpoint 指定本地权重文件。")
    return download("fp16")


def resolve_lut(arg: str) -> dict[int, str]:
    path = arg or os.environ.get("SHENSHU_LUT", "")
    if not path:
        v = ask(
            "条文库",
            "条文库文件路径",
            [
                "条文库不随本项目发布（版权原因），需自备 List.csv（列：条文数, 吉凶断词）",
                "或 {条文数: 断语} 形式的 JSON。也可设置环境变量 SHENSHU_LUT 免去每次输入",
            ],
            optional=True,
            parse=_check_path,
            default_note="只显示条文号，不显示断语",
        )
        path = "" if v is SKIP else v
    return load_lut(path) if path else {}


def _check_path(raw: str) -> str:
    p = raw.strip().strip('"').strip("'")
    if not Path(p).is_file():
        raise BirthError(f"找不到文件：{p}")
    return p


def make_batch(p: Profile) -> dict[str, torch.Tensor]:
    def t(x: int) -> torch.Tensor:
        return torch.tensor([x], dtype=torch.long)

    gz = [parse_pillar(x) for x in p.pillars]
    q = p.qin
    return {
        "sex": t(1 if p.sex == "M" else 2),
        "year_gan": t(gz[0][0]), "year_zhi": t(gz[0][1]),
        "month_gan": t(gz[1][0]), "month_zhi": t(gz[1][1]),
        "day_gan": t(gz[2][0]), "day_zhi": t(gz[2][1]),
        "hour_gan": t(gz[3][0]), "hour_zhi": t(gz[3][1]),
        # skipped zodiac → placeholder 鼠; it only matters through the 真刻, which is then enumerated
        "father_zodiac": t(q.get("father", 1)),
        "mother_zodiac": t(q.get("mother", 1)),
        "spouse_zodiac": t(q.get("spouse", 1)),
        "brothers": t(q.get("brothers", 0)),
        "sons": t(q.get("sons", 0)),
        "daughters": t(q.get("daughters", 0)),
    }


@torch.no_grad()
def predict(model, p: Profile) -> tuple[int | None, list[list[tuple[int, float]]]]:
    """Returns (true_ke or None, rows). rows[k] = six (条文数, prob) for 刻 k."""
    batch = make_batch(p)
    h = model.encode_hidden(batch)
    known = all(k in p.qin for k in ("father", "mother", "brothers", "spouse"))
    if known:
        ke = int(model.qin_true_ke(model.qin_features(batch)).argmax(-1)[0])
        kes = [ke]
    else:
        ke, kes = None, list(range(8))
    rows = []
    for k in kes:
        probs = model.bar_logits_from_hidden(h, torch.tensor([k])).softmax(-1)[0]
        conf, ids = probs.max(-1)
        rows.append([(int(i) + VERSE_OFFSET, float(cf)) for i, cf in zip(ids, conf)])
    return ke, rows


def describe(p: Profile) -> str:
    q = p.qin

    def z(key: str) -> str:
        return ZODIAC_CN[q[key] - 1] if key in q else "?"

    def n(key: str) -> str:
        return str(q[key]) if key in q else "?"

    return (
        f"{'男' if p.sex == 'M' else '女'} · {' '.join(p.pillars)} · "
        f"父{z('father')} 母{z('mother')} 兄弟{n('brothers')} 配偶{z('spouse')} "
        f"子{q.get('sons', 0)} 女{q.get('daughters', 0)}"
    )


def print_book(rows: list[tuple[int, float]], lut: dict[int, str], indent: str = "  ") -> None:
    for sec, (vid, prob) in zip(SECTIONS, rows):
        text = lut.get(vid, "") if lut else ""
        head = f"{indent}【{sec}】" + " " * (4 - len(sec)) * 2
        print(f"{head}#{vid:<6} {text}" + c(f"  (p={prob:.4f})", "2"))


def show_result(p: Profile, ke: int | None, rows, lut: dict[int, str]) -> None:
    print()
    print(c("═" * 64, "36"))
    print(c(" 命书", "1;36") + f"   {describe(p)}")
    print(c("═" * 64, "36"))
    if ke is not None:
        print(f"  真刻：{KE_LABELS[ke]}（序号 {ke}，由父、母、兄弟、配偶推定）")
        print_book(rows[0], lut)
    else:
        missing = [n for k, n in (("father", "父亲生肖"), ("mother", "母亲生肖"), ("brothers", "兄弟人数"), ("spouse", "配偶生肖")) if k not in p.qin]
        print(c(f"  未提供：{'、'.join(missing)} → 无法唯一确定真刻。", "33"))
        print("  以下列出八刻候选，传统做法是对照父母、婚姻等已知事实「考刻」，取最相符的一刻：")
        for k, r in enumerate(rows):
            print()
            print(c(f"  ── {KE_LABELS[k]}（序号 {k}）──", "1"))
            print_book(r, lut, indent="    ")
    if not lut:
        print()
        print(c("  提示：未加载条文库，只显示条文号。下次可输入 List.csv 路径，或设置 SHENSHU_LUT。", "2"))
    print(c("═" * 64, "36"))


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    global COLOR
    ap = argparse.ArgumentParser(prog="shenshu", description="铁板神数 Transformer 交互式命令行")
    ap.add_argument("--checkpoint", default="", help="权重文件（.safetensors / .pt）；默认用已下载的 Release 权重")
    ap.add_argument("--lut", default="", help="条文库 List.csv 或 JSON；也可设置环境变量 SHENSHU_LUT")
    ap.add_argument("--no-color", action="store_true", help="关闭彩色输出")
    ap.add_argument("--version", action="version", version=f"shenshu {__version__}")
    args = ap.parse_args(argv)

    COLOR = _setup_console() and not args.no_color
    print(c(BANNER, "1;36"))
    print(c(f"  铁板神数 Transformer · Shenshu-42M-v1 · v{__version__}", "1"))
    print("  按提示逐项输入；标「可跳过」的项直接回车即可。随时输入 q 退出。")
    print(c("  仅供传统文化与机器学习研究，不构成任何现实决策建议。", "2"))

    try:
        ckpt = resolve_checkpoint(args.checkpoint)
        model, info = load_model(ckpt, "cpu")
        print(c(f"\n  ✓ 已加载权重 {ckpt.name}（step {info.get('step')}）", "32"))
        lut = resolve_lut(args.lut)
        if lut:
            print(c(f"  ✓ 已加载条文库（{len(lut)} 条）", "32"))
        while True:
            p = collect_profile()
            ke, rows = predict(model, p)
            show_result(p, ke, rows, lut)
            again = input(c("\n再算一次？[y/N] > ", "1;33")).strip().lower()
            if again not in ("y", "yes", "是"):
                break
    except (Quit, KeyboardInterrupt, EOFError):
        print()
    print("再见。")


if __name__ == "__main__":
    main()
