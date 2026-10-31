"""CLI demo: 四柱 + 六亲 → 命书.

Shenshu-42M-v1 predicts six 条文号 and looks the verses up in a LUT
(``--lut``: List.csv with 条文数/吉凶断词 columns, or a ``{"1001": "..."}`` JSON).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from transformer_shenshu.data.dataset import ZODIAC, parse_pillar, _idx
from transformer_shenshu.model import ModelConfig, ShenShuModel

SECTIONS = ["总述", "父母", "婚姻", "子女", "功名财帛", "健康"]
ZODIAC_CN = list("鼠牛虎兔龙蛇马羊猴鸡狗猪")
VERSE_OFFSET = 1001  # class id 0 ↔ 条文数 1001


def zodiac_index(value: str) -> int:
    """'rat' / '鼠' / '属鼠' → 1..12 (0 = unknown)."""
    v = (value or "").strip().lower().removeprefix("属")
    if v in ZODIAC_CN:
        return ZODIAC_CN.index(v) + 1
    return _idx(ZODIAC, v)


def parse_qin(text: str) -> dict[str, str]:
    qin: dict[str, str] = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            qin[k.strip()] = v.strip()
    return qin


def build_batch_from_args(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    yg, yz = parse_pillar(args.year)
    mg, mz = parse_pillar(args.month)
    dg, dz = parse_pillar(args.day)
    hg, hz = parse_pillar(args.hour)
    qin = parse_qin(args.qin)

    def t(x: int) -> torch.Tensor:
        return torch.tensor([x], dtype=torch.long, device=device)

    return {
        "sex": t(1 if args.sex == "M" else 2),
        "year_gan": t(yg),
        "year_zhi": t(yz),
        "month_gan": t(mg),
        "month_zhi": t(mz),
        "day_gan": t(dg),
        "day_zhi": t(dz),
        "hour_gan": t(hg),
        "hour_zhi": t(hz),
        "father_zodiac": t(zodiac_index(qin.get("father", qin.get("father_zodiac", "")))),
        "mother_zodiac": t(zodiac_index(qin.get("mother", qin.get("mother_zodiac", "")))),
        "brothers": t(min(int(qin.get("brothers", 0)), 15)),
        "spouse_zodiac": t(zodiac_index(qin.get("spouse", qin.get("spouse_zodiac", "")))),
        "sons": t(min(int(qin.get("sons", 0)), 15)),
        "daughters": t(min(int(qin.get("daughters", 0)), 15)),
    }


def load_lut(path: str) -> dict[int, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"LUT not found: {p}")
    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}
    lut: dict[int, str] = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lut[int(row["条文数"])] = (row.get("吉凶断词") or "").strip()
    return lut


def describe_input(args: argparse.Namespace) -> str:
    qin = parse_qin(args.qin)

    def z(key: str) -> str:
        i = zodiac_index(qin.get(key, ""))
        return ZODIAC_CN[i - 1] if i else "?"

    return (
        f"{'男' if args.sex == 'M' else '女'} · {args.year} {args.month} {args.day} {args.hour} · "
        f"父{z('father')} 母{z('mother')} 兄弟{qin.get('brothers', 0)} 配偶{z('spouse')} "
        f"子{qin.get('sons', 0)} 女{qin.get('daughters', 0)}"
    )


@torch.no_grad()
def predict_book(model: ShenShuModel, batch: dict, lut: dict[int, str]) -> tuple[int, list[str]]:
    model.eval()
    out = model(batch)
    probs = out["bar_logits"].softmax(-1)
    conf, ids = probs.max(-1)
    ke = int(out["true_ke_logits"].argmax(-1)[0])
    lines = []
    for sec, cid, p in zip(SECTIONS, ids[0].tolist(), conf[0].tolist()):
        vid = int(cid) + VERSE_OFFSET
        text = lut.get(vid, "(未提供 --lut，仅显示条号)") if lut else "(未提供 --lut，仅显示条号)"
        lines.append(f"【{sec}】#{vid}  {text}  (p={p:.4f})")
    return ke, lines


def main() -> None:
    p = argparse.ArgumentParser(description="Transformer神数 CLI demo")
    p.add_argument("--sex", choices=["M", "F"], default="M")
    p.add_argument("--year", default="庚午", help="年柱，如 丙寅")
    p.add_argument("--month", default="辛巳", help="月柱")
    p.add_argument("--day", default="丙寅", help="日柱")
    p.add_argument("--hour", default="癸巳", help="时柱")
    p.add_argument(
        "--qin",
        default="father=rat,mother=sheep,brothers=1,spouse=rooster,sons=1,daughters=0",
        help="六亲：father/mother/spouse 生肖（rat 或 鼠），brothers/sons/daughters 人数",
    )
    p.add_argument("--checkpoint", default="")
    p.add_argument("--lut", default="", help="条文库：List.csv（条文数,吉凶断词）或 {条文数: 文本} JSON")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = ModelConfig()
    model = ShenShuModel(cfg)

    if args.checkpoint and Path(args.checkpoint).is_file():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        raw_cfg = ckpt.get("config", {})
        cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in ModelConfig.__dataclass_fields__})
        model = ShenShuModel(cfg)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("step")
        print(f"loaded {args.checkpoint}" + (f" (step {step})" if step is not None else ""))
    else:
        print("no checkpoint; using randomly initialized weights (smoke demo)")

    model.to(device)
    batch = build_batch_from_args(args, device)
    print("输入:", describe_input(args))

    ke, lines = predict_book(model, batch, load_lut(args.lut))
    print("真刻(模型):", ke)
    print("---- 命书 ----")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
