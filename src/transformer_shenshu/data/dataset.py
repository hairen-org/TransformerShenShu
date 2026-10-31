"""Memmap / compact dataset for Shenshu-42M-v1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

GAN = list("甲乙丙丁戊己庚辛壬癸")
ZHI = list("子丑寅卯辰巳午未申酉戌亥")
ZODIAC = [
    "rat", "ox", "tiger", "rabbit", "dragon", "snake",
    "horse", "sheep", "monkey", "rooster", "dog", "pig",
]

ROW_DTYPE = np.dtype([
    ("sex", "u1"),
    ("year_gan", "u1"), ("year_zhi", "u1"),
    ("month_gan", "u1"), ("month_zhi", "u1"),
    ("day_gan", "u1"), ("day_zhi", "u1"),
    ("hour_gan", "u1"), ("hour_zhi", "u1"),
    ("father_zodiac", "u1"), ("mother_zodiac", "u1"), ("spouse_zodiac", "u1"),
    ("brothers", "u1"), ("sons", "u1"), ("daughters", "u1"),
    ("verse_ids", "u2", (6,)),
])


def _idx(table: list[str], value: str, offset: int = 1) -> int:
    if not value:
        return 0
    try:
        return table.index(value) + offset
    except ValueError:
        return 0


def parse_pillar(pillar: str) -> tuple[int, int]:
    if not pillar or len(pillar) < 2:
        return 0, 0
    return _idx(GAN, pillar[0]), _idx(ZHI, pillar[1])


class CompactMemmapDataset(Dataset):
    """Streaming mmap over structured .npy from build_processed_42m.py."""

    FEATURE_KEYS = [
        "sex", "year_gan", "year_zhi", "month_gan", "month_zhi",
        "day_gan", "day_zhi", "hour_gan", "hour_zhi",
        "father_zodiac", "mother_zodiac", "brothers", "spouse_zodiac", "sons", "daughters",
    ]

    def __init__(self, path: str | Path, max_rows: int | None = None, seed: int = 0):
        self.path = Path(path)
        self.data = np.load(str(self.path), mmap_mode="r")
        self.n = len(self.data) if max_rows is None else min(len(self.data), max_rows)
        self.indices = np.arange(self.n)
        if max_rows is not None and max_rows < len(self.data):
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(len(self.data), size=self.n, replace=False)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        r = self.data[int(self.indices[idx])]
        item = {k: int(r[k]) for k in self.FEATURE_KEYS}
        item["verse_ids"] = [int(x) for x in r["verse_ids"]]
        return item


def collate_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    keys = [
        "sex", "year_gan", "year_zhi", "month_gan", "month_zhi",
        "day_gan", "day_zhi", "hour_gan", "hour_zhi",
        "father_zodiac", "mother_zodiac", "brothers", "spouse_zodiac", "sons", "daughters",
    ]
    batch = {k: torch.tensor([s[k] for s in samples], dtype=torch.long) for k in keys}
    if "verse_ids" in samples[0]:
        batch["verse_ids"] = torch.tensor([s["verse_ids"] for s in samples], dtype=torch.long)
    return batch
