"""Training entry for Shenshu-42M-v1 (六门条文号分类)."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer_shenshu.data import CompactMemmapDataset, collate_batch
from transformer_shenshu.model import ModelConfig, ShenShuModel


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model_config(cfg_yaml: dict) -> ModelConfig:
    raw = cfg_yaml.get("model") or {}
    fields = ModelConfig.__dataclass_fields__
    return ModelConfig(**{k: v for k, v in raw.items() if k in fields})


@torch.no_grad()
def evaluate(model: ShenShuModel, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict:
    model.eval()
    total_correct = 0.0
    total_sec = torch.zeros(6, device=device)
    total_n = 0
    total_loss = 0.0
    total_ke = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        B = batch["verse_ids"].size(0)
        total_loss += float(out["loss_bar"].detach()) * B
        preds = out["bar_logits"].argmax(dim=-1)
        correct = (preds == batch["verse_ids"]).float()
        total_correct += float(correct.mean().detach()) * B
        total_sec += correct.mean(dim=0) * B
        if "true_ke_acc" in out:
            total_ke += float(out["true_ke_acc"].detach()) * B
        total_n += B
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train()
    if total_n == 0:
        return {"loss": 0.0, "bar_acc": 0.0, "bar_acc_per_sec": [0.0] * 6}
    res = {
        "loss": total_loss / total_n,
        "bar_acc": total_correct / total_n,
        "bar_acc_per_sec": (total_sec / total_n).detach().cpu().tolist(),
    }
    if total_ke > 0:
        res["true_ke_acc"] = total_ke / total_n
    return res


def cosine_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_loop(args: argparse.Namespace) -> None:
    cfg_yaml = load_config(args.config)
    tcfg = cfg_yaml.get("train") or {}
    data_cfg = cfg_yaml.get("data") or {}

    batch_size = args.batch_size or int(tcfg.get("batch_size", 32))
    lr = args.lr if args.lr is not None else float(tcfg.get("lr", 3e-4))
    max_steps = args.max_steps or int(tcfg.get("max_steps", 300_000))
    grad_accum = args.grad_accum or int(tcfg.get("grad_accum", 2))
    warmup = int(tcfg.get("warmup_steps", 2000))
    log_every = int(tcfg.get("log_every", 50))
    val_every = int(tcfg.get("val_every", 5000))
    val_subset = int(tcfg.get("val_subset", 10000))
    early_acc = float(tcfg.get("early_stop_bar_acc", 0.95))
    early_patience = int(tcfg.get("early_stop_patience", 2))
    seed = int(tcfg.get("seed", 20261030))
    use_fp16 = str(tcfg.get("precision", "fp16")).lower() in ("fp16", "float16", "amp")

    train_path = Path(args.train_data or data_cfg.get("train", "private/out/shenshu_42m_v1/train.npy"))
    val_path = Path(args.val_data or data_cfg.get("val", "private/out/shenshu_42m_v1/val.npy"))
    out_dir = Path(args.out_dir or tcfg.get("out_dir", "checkpoints/Shenshu-42M-v1"))
    art_dir = Path(args.artifacts or tcfg.get("artifacts", "artifacts/Shenshu-42M-v1"))
    log_dir = Path(args.log_dir or "logs/Shenshu-42M-v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mcfg = build_model_config(cfg_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ShenShuModel(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tcfg.get("weight_decay", 0.01)))
    scaler = GradScaler("cuda", enabled=use_fp16 and device.type == "cuda")

    train_ds = CompactMemmapDataset(train_path)
    val_ds_full = CompactMemmapDataset(val_path)
    val_ds_sub = CompactMemmapDataset(val_path, max_rows=val_subset, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader_sub = DataLoader(
        val_ds_sub, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_batch,
    )
    val_loader_full = DataLoader(
        val_ds_full, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_batch,
    )

    print(
        f"device={device} params={model.count_parameters():,} "
        f"train={len(train_ds):,} val={len(val_ds_full):,} "
        f"batch={batch_size} accum={grad_accum} fp16={use_fp16}",
        flush=True,
    )

    if args.dry_run:
        max_steps = min(max_steps, 20)
        print("DRY RUN: capping steps", max_steps, flush=True)

    history_path = art_dir / "history.csv"
    hist_file = history_path.open("w", encoding="utf-8", newline="")
    hist_writer = csv.DictWriter(
        hist_file,
        fieldnames=[
            "step", "loss", "bar_acc_train", "lr",
            "val_loss", "val_bar_acc",
            "val_s0", "val_s1", "val_s2", "val_s3", "val_s4", "val_s5",
            "wall_sec",
        ],
    )
    hist_writer.writeheader()

    step = 0
    best_acc = -1.0
    high_acc_streak = 0
    t0 = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, desc="train-42m")
    stop = False

    while step < max_steps and not stop:
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            cur_lr = cosine_lr(step, warmup, max_steps, lr)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with autocast("cuda", enabled=use_fp16 and device.type == "cuda"):
                out = model(batch)
                loss = out.get("loss", out["loss_bar"]) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            step += 1
            train_acc = float(out["bar_acc"].detach())
            train_loss = float(out["loss_bar"].detach())
            ke_acc = float(out["true_ke_acc"].detach()) if "true_ke_acc" in out else -1.0
            pbar.set_postfix(loss=train_loss, acc=f"{train_acc:.3f}", ke=f"{ke_acc:.3f}", lr=f"{cur_lr:.2e}")
            pbar.update(1)

            row = {
                "step": step,
                "loss": train_loss,
                "bar_acc_train": train_acc,
                "lr": cur_lr,
                "val_loss": "",
                "val_bar_acc": "",
                "val_s0": "", "val_s1": "", "val_s2": "", "val_s3": "", "val_s4": "", "val_s5": "",
                "wall_sec": round(time.time() - t0, 1),
            }

            do_val = (step % val_every == 0) or (step == max_steps)
            if do_val:
                metrics = evaluate(model, val_loader_sub, device)
                row["val_loss"] = metrics["loss"]
                row["val_bar_acc"] = metrics["bar_acc"]
                for i, a in enumerate(metrics["bar_acc_per_sec"]):
                    row[f"val_s{i}"] = a
                ke_s = f" true_ke_acc={metrics['true_ke_acc']:.4f}" if "true_ke_acc" in metrics else ""
                print(
                    f"\n[val@{step}] loss={metrics['loss']:.4f} bar_acc={metrics['bar_acc']:.4f}"
                    f"{ke_s} per_sec={[round(x, 4) for x in metrics['bar_acc_per_sec']]}",
                    flush=True,
                )
                if metrics["bar_acc"] > best_acc:
                    best_acc = metrics["bar_acc"]
                    ckpt = {
                        "model": model.state_dict(),
                        "config": mcfg.__dict__,
                        "step": step,
                        "val_bar_acc": best_acc,
                        "val_metrics": metrics,
                    }
                    torch.save(ckpt, out_dir / "best_bar_acc.pt")
                    print(f"saved best_bar_acc.pt acc={best_acc:.4f}", flush=True)
                if metrics["bar_acc"] >= early_acc:
                    high_acc_streak += 1
                else:
                    high_acc_streak = 0
                if high_acc_streak >= early_patience:
                    print(f"early stop: val bar-acc>={early_acc} for {early_patience} vals", flush=True)
                    stop = True

            if step % log_every == 0 or do_val:
                hist_writer.writerow(row)
                hist_file.flush()

            if step % int(tcfg.get("save_every", 10000)) == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "config": mcfg.__dict__,
                    "step": step,
                    "val_bar_acc": best_acc,
                }
                torch.save(ckpt, out_dir / "last.pt")

            if step >= max_steps or stop:
                break

    pbar.close()

    # final full val
    print("running full validation...", flush=True)
    full_metrics = evaluate(model, val_loader_full, device)
    print(
        f"[full val] loss={full_metrics['loss']:.4f} bar_acc={full_metrics['bar_acc']:.4f} "
        f"per_sec={[round(x, 4) for x in full_metrics['bar_acc_per_sec']]}",
        flush=True,
    )
    ckpt = {
        "model": model.state_dict(),
        "config": mcfg.__dict__,
        "step": step,
        "val_bar_acc": full_metrics["bar_acc"],
        "val_metrics": full_metrics,
        "wall_sec": time.time() - t0,
    }
    torch.save(ckpt, out_dir / "last.pt")
    if full_metrics["bar_acc"] > best_acc:
        torch.save(ckpt, out_dir / "best_bar_acc.pt")
        best_acc = full_metrics["bar_acc"]

    summary = {
        "steps": step,
        "best_bar_acc": best_acc,
        "final_full_val_bar_acc": full_metrics["bar_acc"],
        "final_full_val_loss": full_metrics["loss"],
        "final_per_sec": full_metrics["bar_acc_per_sec"],
        "wall_sec": round(time.time() - t0, 1),
        "params": model.count_parameters(),
    }
    (art_dir / "train_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    hist_file.close()
    print("TRAIN DONE", summary, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Shenshu-42M-v1 trainer")
    p.add_argument("--config", default="configs/shenshu_42m_v1.yaml")
    p.add_argument("--train-data", default="")
    p.add_argument("--val-data", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--artifacts", default="")
    p.add_argument("--log-dir", default="")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
