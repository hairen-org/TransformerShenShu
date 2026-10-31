"""Load / export / download Shenshu-42M-v1 weights.

Release assets (GitHub Release ``v1.0.0``):

* ``shenshu-42m-v1.safetensors``       fp32 weights (config embedded in metadata)
* ``shenshu-42m-v1-fp16.safetensors``  fp16 weights (cast back to fp32 on load)
* ``config.json``                      ModelConfig + training metrics
* ``SHA256SUMS``
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import torch

from transformer_shenshu.model import ModelConfig, ShenShuModel

REPO = "hairen-org/TransformerShenShu"
RELEASE_TAG = "v1.0.0"
VARIANTS = {
    "fp32": "shenshu-42m-v1.safetensors",
    "fp16": "shenshu-42m-v1-fp16.safetensors",
}
# Filled from the released files; used to verify downloads.
SHA256: dict[str, str] = {
    "shenshu-42m-v1.safetensors": "2286fefec10b94f6a63a4708b54aae2693820bf489245f28d60f060f17f8fec9",
    "shenshu-42m-v1-fp16.safetensors": "975ed9b06ae9578464f5570ae8841183efed2ee7b8ac83b2a29514ffc54206a9",
}


def _filter_cfg(raw: dict) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[ShenShuModel, dict]:
    """Load a ``.safetensors`` export or a ``.pt`` training checkpoint.

    ``.pt`` files are opened with ``weights_only=True`` (no arbitrary code execution).
    Returns ``(model, info)`` where ``info`` holds the config and any stored metrics.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        info = json.loads(meta.get("shenshu", "{}"))
        if not info.get("config"):
            side = path.with_name("config.json")
            if side.is_file():
                info = json.loads(side.read_text(encoding="utf-8"))
        state = load_file(str(path))
    else:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        state = ckpt["model"]
        info = {k: v for k, v in ckpt.items() if k != "model"}
    model = ShenShuModel(_filter_cfg(info.get("config", {})))
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.to(device).eval()
    return model, info


def export_safetensors(ckpt_path: str | Path, out_path: str | Path, dtype: torch.dtype = torch.float32) -> dict:
    """Convert a training ``.pt`` checkpoint to ``.safetensors`` (config in metadata)."""
    from safetensors.torch import save_file

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    info = {
        "name": "Shenshu-42M-v1",
        "config": ckpt["config"],
        "step": ckpt.get("step"),
        "val_bar_acc": ckpt.get("val_bar_acc"),
        "dtype": str(dtype).replace("torch.", ""),
    }
    state = {k: v.to(dtype).contiguous() for k, v in ckpt["model"].items()}
    save_file(state, str(out_path), metadata={"shenshu": json.dumps(info)})
    return info


def sha256sum(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir() -> Path:
    base = os.environ.get("SHENSHU_CACHE") or Path.home() / ".cache" / "transformer_shenshu"
    return Path(base) / RELEASE_TAG


def download(variant: str = "fp32", force: bool = False) -> Path:
    """Download a release asset into the cache and verify its SHA256."""
    name = VARIANTS[variant]
    dest = cache_dir() / name
    expected = SHA256.get(name)
    if dest.is_file() and not force and (not expected or sha256sum(dest) == expected):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    got = sha256sum(tmp)
    if expected and got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for {name}: {got} != {expected}")
    tmp.replace(dest)
    return dest
