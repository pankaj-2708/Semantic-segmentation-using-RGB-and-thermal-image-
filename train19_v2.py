"""
train19_v2.py -- training / evaluation pipeline for RTFNet18V2 (model19_v2.py) on the prepared
MFNet RGB-T dataset (dataset19_v2.py).  Never imports the teammate's files or the V1 files.

Typical use (run from the repository root, in this order):
    python dataset19_v2.py --diagnose                  # 1. read-only label / alignment diagnostic
    python train19_v2.py --sanity                      # 2. real-data sanity check, writes nothing permanent
    python train19_v2.py --epochs 100                  # 3. full run (class-weighted CE, see defaults)
    python train19_v2.py --epochs 100 --resume         # continue from checkpoints/model19_v2/last.pth
    python train19_v2.py --eval-only --eval-split val  # evaluate best.pth (no training)

Outputs (V1 folders checkpoints/model19 and plots/model19 are never touched):
    checkpoints/model19_v2/{best.pth,last.pth,history.json,history.csv,test_results.json,train_log.txt,...}
    plots/model19_v2/{training_curves.png,val_predictions_epochNNN.png,val_pred_distribution_epochNNN.png,
                      test_predictions.png,test_confusion_matrix.png,test_pred_distribution.png}

POLICIES (printed at start-up; they are separate switches)
---------------------------------------------------------
* Loss class-0 policy  --class0-loss include|ignore
    include (default): class 0 is an ordinary class in the loss, exactly like the official RTFNet
                       training (plain CE, no ignore_index).
    ignore           : class-0 pixels are excluded from CE and Dice.  The network is then never
                       taught to say "class 0"; it is rewarded only on the 8 labelled classes.
                       all-9 mIoU is then expected to be poor on class 0, so select on fg8.
* Evaluation (NEVER changed by the loss policy): one confusion matrix over the whole set, all 9
  classes (class 0 included, as official RTFNet util.compute_results) -> ``mIoU (9)``; and mIoU over
  classes 1..8 -> ``mIoU (1-8)``; both are always reported.  Pixels labelled 255 (if any) are ignored.
* Best checkpoint: --select-metric auto = mIoU(9) if class 0 is in the loss, else mIoU(1-8).
* Class weights --class-weights none|inv_sqrt|median_freq|enet, computed from TRAIN label pixel
  frequencies only, clipped so max/min weight <= --max-weight-ratio (default 20), normalised to mean
  1 over the classes in the loss, printed at start-up and stored in every checkpoint.  The same
  weights are used for training and validation/test loss.  They are a hypothesis against the 92 %
  class-0 imbalance, not a guaranteed fix; dice is not applied automatically.

TIMING: every epoch prints train / validation / other / total durations (time.perf_counter, CUDA
synchronised), mean epoch time, mean of the last 5 epochs, elapsed, ETA (recent mean x epochs left)
and the expected finish time.  Nothing is estimated before it is measured.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import dataset19_v2 as ds2
import model19_v2 as m2

NUM_CLASSES = ds2.NUM_CLASSES
IGNORE_INDEX = ds2.IGNORE_INDEX
SCRIPT_DIR = Path(__file__).resolve().parent
RESERVED_RUN_NAMES = {"model19", "model", "model1", "model2", "model3"}   # V1 / teammates' folders
CKPT_ARG_KEYS = ("head", "thermal_norm", "img_size", "class0_loss", "class_weights", "max_weight_ratio",
                 "ce_weight", "dice_weight", "select_metric", "thermal_init")


# =========================================================================== #
# Utilities
# =========================================================================== #
class Logger:
    def __init__(self) -> None:
        self.path: Optional[Path] = None

    def attach(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


log = Logger()


def fmt_time(seconds: float) -> str:
    """HH:MM:SS (hours not wrapped at 24)."""
    s = int(round(max(seconds, 0)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_clock(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%d %I:%M %p")


def estimate_eta(epoch_totals: Sequence[float], epochs_left: int, window: int = 5) -> Tuple[float, float, float]:
    """(eta_seconds, mean_all, mean_recent) from MEASURED epoch durations (ETA uses the recent mean)."""
    if not epoch_totals:
        return float("nan"), float("nan"), float("nan")
    mean_all = float(np.mean(epoch_totals))
    mean_recent = float(np.mean(epoch_totals[-window:]))
    return mean_recent * max(epochs_left, 0), mean_all, mean_recent


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_tag(device: torch.device) -> str:
    """Short label used in live progress bars so CPU/GPU use is always visible."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "CPU"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx).replace("NVIDIA GeForce ", "")
    return f"GPU:{name}"


def live_device_status(device: torch.device) -> str:
    """Low-overhead live device indicator; reports current allocated/reserved VRAM on CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "CPU active"
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        allocated = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        total = props.total_memory / (1024 ** 3)
        return f"GPU VRAM {allocated:.1f}/{total:.1f}G alloc; {reserved:.1f}G reserved"
    except Exception:
        return f"{device_tag(device)} active"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def progress(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True, mininterval=1.0)
    except ImportError:
        return iterable


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False "
                           f"(torch {torch.__version__}, CUDA build {torch.version.cuda}).")
    return torch.device(arg)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)   # our own trusted file
    except TypeError:
        return torch.load(path, map_location=device)


def even_subset(ds, n: Optional[int]):
    if n is None or n >= len(ds):
        return ds
    idx = np.unique(np.linspace(0, len(ds) - 1, num=n).round().astype(int)).tolist()
    return Subset(ds, idx)


def is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_cls is not None and isinstance(exc, oom_cls)) or "out of memory" in str(exc).lower()


OOM_HELP = """
CUDA ran out of memory.  Nothing was changed silently.  Options (pick one and re-run):
  1. Smaller batch:  --batch-size 1 --accum-steps 8   (same effective batch of 8).
  2. Lower resolution, e.g. --img-size 384 512  (NOT directly comparable with teammates' 480x640 runs).
  3. Close other GPU programs (browsers with hardware acceleration, games, other Python sessions).
  4. Keep AMP on (default on CUDA).  Gradient accumulation does not reduce activation memory.
"""


# =========================================================================== #
# Class weights and loss
# =========================================================================== #
def compute_class_weights(counts9: np.ndarray, mode: str, max_ratio: float, class0_loss: str) -> Optional[np.ndarray]:
    """Per-class loss weights from TRAIN pixel counts (float64[9]); None for mode 'none'.

    inv_sqrt: f^-0.5 | median_freq: median(f)/f | enet: 1/ln(1.02+f).  Then clipped to
    max/min <= max_ratio over the classes in the loss and normalised to mean 1 over those classes.
    Classes not in the loss (class 0 under --class0-loss ignore) get weight 1.0 (never used)."""
    if mode == "none":
        return None
    active = list(range(NUM_CLASSES)) if class0_loss == "include" else list(range(1, NUM_CLASSES))
    c = counts9.astype(np.float64)
    f = np.maximum(c[active] / max(c[active].sum(), 1.0), 1e-6)
    if mode == "inv_sqrt":
        w = f ** -0.5
    elif mode == "median_freq":
        w = np.median(f) / f
    elif mode == "enet":
        w = 1.0 / np.log(1.02 + f)
    else:
        raise ValueError(f"unknown class-weight mode {mode!r}")
    w = np.minimum(w, w.min() * max_ratio)
    w = w / w.mean()
    full = np.ones(NUM_CLASSES, dtype=np.float64)
    full[active] = w
    return full


class SegmentationLoss(nn.Module):
    """total = ce_weight * CE(weighted) + dice_weight * SoftDice.   Targets: int64 [B,H,W].

    class0_loss='ignore' maps class-0 pixels to ignore_index for BOTH terms (Dice is then averaged over
    classes 1..8).  Dice is the batch-level soft Dice per class (smooth 1).  Always computed in fp32."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 0.0,
                 class_weights: Optional[Sequence[float]] = None, class0_loss: str = "include",
                 num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight <= 0:
            raise ValueError("loss weights must be >= 0 and not both zero")
        if class0_loss not in ("include", "ignore"):
            raise ValueError("class0_loss must be 'include' or 'ignore'")
        self.ce_weight, self.dice_weight = float(ce_weight), float(dice_weight)
        self.class0_loss, self.num_classes, self.ignore_index = class0_loss, num_classes, ignore_index
        self.dice_classes = list(range(num_classes)) if class0_loss == "include" else list(range(1, num_classes))
        w = None if class_weights is None else torch.as_tensor(list(class_weights), dtype=torch.float32)
        self.register_buffer("weight", w, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logits = logits.float()
        tgt = target
        if self.class0_loss == "ignore":
            tgt = torch.where(target == 0, torch.full_like(target, self.ignore_index), target)
        valid = tgt != self.ignore_index
        parts: Dict[str, torch.Tensor] = {}
        total = logits.sum() * 0.0
        if self.ce_weight > 0:
            w = self.weight.to(logits.device) if self.weight is not None else None
            ce_sum = F.cross_entropy(logits, tgt, weight=w, ignore_index=self.ignore_index, reduction="sum")
            denom = valid.sum().float() if w is None else w[tgt[valid]].sum()
            ce = ce_sum / denom.clamp_min(1e-8)          # 0 (not NaN) if every pixel is ignored
            parts["ce"] = ce.detach()
            total = total + self.ce_weight * ce
        if self.dice_weight > 0:
            tgt0 = torch.where(valid, tgt, torch.zeros_like(tgt))
            vf = valid.unsqueeze(1).float()
            probs = logits.softmax(1) * vf
            one_hot = F.one_hot(tgt0, self.num_classes).permute(0, 3, 1, 2).float() * vf
            inter = (probs * one_hot).sum((0, 2, 3))
            denom = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
            dice = (2 * inter + 1.0) / (denom + 1.0)
            dloss = 1.0 - dice[self.dice_classes].mean()
            parts["dice"] = dloss.detach()
            total = total + self.dice_weight * dloss
        return total, parts


def describe_policies(args, weights: Optional[np.ndarray], counts9: Optional[np.ndarray], select_metric: str) -> None:
    log("-" * 72)
    log("POLICIES")
    log(f"  loss             : {args.ce_weight} x CE" + (f" + {args.dice_weight} x SoftDice" if args.dice_weight > 0 else " (no Dice)"))
    log(f"  class-0 in loss  : {args.class0_loss.upper()}  "
        + ("(official RTFNet: class 0 is an ordinary class in the loss)" if args.class0_loss == "include"
           else "(class-0 pixels excluded from CE/Dice; DEVIATES from the official loss)"))
    log(f"  class weights    : {args.class_weights}" + (f" (max/min <= {args.max_weight_ratio:g}, mean 1)" if weights is not None else ""))
    if weights is not None and counts9 is not None:
        share = counts9 / max(counts9.sum(), 1)
        for i in range(NUM_CLASSES):
            used = args.class0_loss == "include" or i > 0
            log(f"      {i} {ds2.CLASS_NAMES[i]:<22} train px {100 * share[i]:7.3f}%  weight {weights[i]:6.3f}" + ("" if used else "  (ignored)"))
    log("  evaluation       : whole-set confusion matrix; mIoU over all 9 classes (class 0 INCLUDED, official "
        "RTFNet) and mIoU over classes 1-8 are both reported every epoch; pixel label 255 ignored")
    log(f"  best checkpoint  : selected on validation {select_metric} ({'mIoU over 9 classes' if select_metric == 'miou_all9' else 'mIoU over classes 1-8'}); test split used only once at the end")
    if args.class0_loss == "ignore" and select_metric == "miou_all9":
        log("  WARNING: class 0 is ignored in the loss but mIoU(9) is the selection metric; class 0 will score "
            "poorly by construction.  Consider --select-metric miou_fg8.")
    log("-" * 72)


# =========================================================================== #
# Metrics (whole-set confusion matrix)
# =========================================================================== #
def update_confusion(conf: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                     n: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> None:
    valid = (target != ignore_index) & (target >= 0) & (target < n)
    idx = target[valid] * n + pred[valid]
    conf += torch.bincount(idx, minlength=n * n).reshape(n, n)


def compute_metrics(conf: np.ndarray) -> Dict[str, object]:
    conf = conf.astype(np.float64)
    tp, gt, pr = np.diag(conf), conf.sum(1), conf.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(gt + pr - tp > 0, tp / (gt + pr - tp), np.nan)
        dice = np.where(gt + pr > 0, 2 * tp / (gt + pr), np.nan)
        recall = np.where(gt > 0, tp / gt, np.nan)
        precision = np.where(pr > 0, tp / pr, np.nan)
    total = conf.sum()
    nm = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    return {
        "pixel_acc": float(tp.sum() / total) if total > 0 else float("nan"),
        "miou_all9": nm(iou), "miou_fg8": nm(iou[1:]),
        "mean_dice_all9": nm(dice), "mean_dice_fg8": nm(dice[1:]),
        "iou_per_class": iou.tolist(), "dice_per_class": dice.tolist(),
        "recall_per_class": recall.tolist(), "precision_per_class": precision.tolist(),
        "gt_pixels_per_class": gt.astype(np.int64).tolist(), "pred_pixels_per_class": pr.astype(np.int64).tolist(),
        "gt_share": (gt / total).tolist() if total > 0 else [float("nan")] * len(gt),
        "pred_share": (pr / total).tolist() if total > 0 else [float("nan")] * len(pr),
        "total_pixels": int(total),
    }


def metric_value(metrics: Dict[str, object], key: str) -> float:
    return float(metrics[key])


def _pct(v) -> str:
    return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:5.1f}"


def metrics_table(m: Dict[str, object]) -> str:
    f = lambda v: "   n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:6.2f}"
    lines = [f"{'class':<24}{'IoU %':>8}{'Dice %':>8}{'Recall %':>10}{'Prec %':>8}{'GT share %':>12}{'Pred share %':>14}"]
    for i, name in enumerate(ds2.CLASS_NAMES):
        lines.append(f"{i} {name:<22}{f(m['iou_per_class'][i]):>8}{f(m['dice_per_class'][i]):>8}"
                     f"{f(m['recall_per_class'][i]):>10}{f(m['precision_per_class'][i]):>8}"
                     f"{100 * m['gt_share'][i]:>12.3f}{100 * m['pred_share'][i]:>14.3f}")
    lines.append(f"{'mIoU (9, incl. class 0)':<32}{100 * m['miou_all9']:6.2f}")
    lines.append(f"{'mIoU (classes 1-8)':<32}{100 * m['miou_fg8']:6.2f}")
    lines.append(f"{'mean Dice (9 / 1-8)':<32}{100 * m['mean_dice_all9']:6.2f} / {100 * m['mean_dice_fg8']:6.2f}")
    lines.append(f"{'pixel accuracy':<32}{100 * m['pixel_acc']:6.2f}   (misleading: always predicting class 0 already scores ~ its pixel share)")
    return "\n".join(lines)


def nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    return obj


# =========================================================================== #
# Train / evaluate
# =========================================================================== #
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, accum_steps,
                    grad_clip, epoch, epochs, log_interval) -> float:
    model.set_train_mode()
    n_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    n_seen, t0 = 0, time.perf_counter()
    bar = progress(loader, total=n_batches, desc=f"train {epoch}/{epochs} [{device_tag(device)}]")
    plain = not hasattr(bar, "set_postfix")
    for it, (rgb, thermal, target) in enumerate(bar):
        rgb = rgb.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(device, use_amp):
            logits = model(rgb, thermal)
        loss, _ = criterion(logits, target)
        group_start = (it // accum_steps) * accum_steps
        group_size = min(accum_steps, n_batches - group_start)
        scaler.scale(loss / group_size).backward()
        if it + 1 == group_start + group_size:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        bs = rgb.shape[0]
        loss_sum += loss.detach() * bs
        n_seen += bs
        if (it + 1) % 20 == 0 or it + 1 == n_batches:       # occasional GPU sync for NaN check / display
            cur = float(loss_sum) / n_seen
            if not math.isfinite(cur):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}, iteration {it}. "
                                         "Try a lower --lr, --grad-clip 1.0 or --amp off.")
            if not plain:
                bar.set_postfix(loss=f"{cur:.4f}", device=live_device_status(device))
            elif (it + 1) % log_interval == 0 or it + 1 == n_batches:
                print(f"  train {epoch}/{epochs} [{it + 1}/{n_batches}] loss {cur:.4f} "
                      f"device={live_device_status(device)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if hasattr(bar, "close"):
        bar.close()
    return float(loss_sum) / max(n_seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, desc="val") -> Dict[str, object]:
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    loss_sum, n_seen = torch.zeros((), device=device), 0
    bar = progress(loader, total=len(loader), desc=f"{desc} [{device_tag(device)}]")
    for batch_idx, (rgb, thermal, target) in enumerate(bar):
        rgb = rgb.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(device, use_amp):
            logits = model(rgb, thermal)
        loss, _ = criterion(logits, target)
        update_confusion(conf, logits.argmax(1), target)
        loss_sum += loss.detach() * rgb.shape[0]
        n_seen += rgb.shape[0]
        if hasattr(bar, "set_postfix") and ((batch_idx + 1) % 20 == 0 or batch_idx + 1 == len(loader)):
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}", device=live_device_status(device))
    if hasattr(bar, "close"):
        bar.close()
    conf_np = conf.cpu().numpy()
    return {"loss": float(loss_sum) / max(n_seen, 1), "metrics": compute_metrics(conf_np), "conf": conf_np}


# =========================================================================== #
# Checkpoints / optimiser
# =========================================================================== #
def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if args.scheduler == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


# =========================================================================== #
# Plots (optional matplotlib; plotting never aborts training)
# =========================================================================== #
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        if not getattr(_plt, "warned", False):
            log("WARNING: matplotlib not installed -> plots skipped (pip install matplotlib)")
            _plt.warned = True
        return None


def plot_curves(history: Dict[str, list], path: Path, select_metric: str) -> None:
    plt = _plt()
    if plt is None or not history["epoch"]:
        return
    try:
        ep = history["epoch"]
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        ax[0, 0].plot(ep, history["train_loss"], label="train"); ax[0, 0].plot(ep, history["val_loss"], label="val")
        ax[0, 0].set_title("Loss"); ax[0, 0].legend()
        ax[0, 1].plot(ep, [100 * v for v in history["val_miou_all9"]], label="mIoU (9)")
        ax[0, 1].plot(ep, [100 * v for v in history["val_miou_fg8"]], label="mIoU (1-8)")
        key = "val_" + select_metric.replace("miou_", "miou_")
        vals = np.array([np.nan if v is None else v for v in history[key]], dtype=float)
        if np.isfinite(vals).any():
            b = int(np.nanargmax(vals))
            ax[0, 1].scatter([ep[b]], [100 * vals[b]], c="r", zorder=3, label=f"best ({select_metric}) @ {ep[b]}")
        ax[0, 1].set_title("Validation mIoU (%)"); ax[0, 1].legend()
        ax[0, 2].plot(ep, [100 * v for v in history["val_pixel_acc"]], label="pixel acc")
        ax[0, 2].plot(ep, [100 * v for v in history["val_mean_dice_all9"]], label="mean Dice (9)")
        ax[0, 2].plot(ep, [100 * v for v in history["val_mean_dice_fg8"]], label="mean Dice (1-8)")
        ax[0, 2].set_title("Validation pixel acc / Dice (%)"); ax[0, 2].legend()
        ax[1, 0].plot(ep, [100 * s[0] for s in history["val_pred_share"]], label="predicted class-0 share")
        ax[1, 0].axhline(100 * history["val_gt_share0"][-1], color="k", ls="--", label="true class-0 share")
        ax[1, 0].set_title("Class-0 share of predictions (%) - collapse monitor"); ax[1, 0].legend()
        ax[1, 1].plot(ep, history["lr"]); ax[1, 1].set_title("Learning rate")
        ax[1, 2].plot(ep, history["epoch_seconds"], label="epoch total"); ax[1, 2].plot(ep, history["train_seconds"], label="train")
        ax[1, 2].plot(ep, history["val_seconds"], label="validation"); ax[1, 2].set_title("Seconds per epoch"); ax[1, 2].legend()
        for a in ax.ravel():
            a.set_xlabel("epoch"); a.grid(alpha=0.3)
        fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=105); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not save training curves: {exc}")


def plot_iou_history(history: Dict[str, list], path: Path) -> None:
    """Heatmap of validation IoU for every class across epochs."""
    plt = _plt()
    if plt is None or not history.get("val_iou_per_class"):
        return
    try:
        values = np.asarray([
            [np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in row]
            for row in history["val_iou_per_class"]
        ], dtype=float).T
        fig_h = max(4.5, 0.48 * NUM_CLASSES + 1.4)
        fig, ax = plt.subplots(figsize=(max(10, min(20, len(history["epoch"]) * 0.16)), fig_h))
        im = ax.imshow(values * 100.0, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=100)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels([f"{i} {n}" for i, n in enumerate(ds2.CLASS_NAMES)])
        epochs = history["epoch"]
        tick_idx = np.unique(np.linspace(0, len(epochs) - 1, num=min(12, len(epochs))).round().astype(int))
        ax.set_xticks(tick_idx); ax.set_xticklabels([epochs[i] for i in tick_idx])
        ax.set_xlabel("Epoch"); ax.set_title("Validation IoU by class across training (%)")
        fig.colorbar(im, ax=ax, label="IoU %", fraction=0.025, pad=0.02)
        fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=125); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not save per-class IoU history: {exc}")


def plot_confusion_counts(conf: np.ndarray, path: Path, title: str) -> None:
    """Raw confusion counts displayed on log scale, complementary to row-normalised confusion."""
    plt = _plt()
    if plt is None:
        return
    try:
        counts = np.asarray(conf, dtype=np.float64)
        values = np.log10(counts + 1.0)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(values, cmap="magma", interpolation="nearest")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels([f"{i} {n}" for i, n in enumerate(ds2.CLASS_NAMES)], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([f"{i} {n}" for i, n in enumerate(ds2.CLASS_NAMES)], fontsize=8)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class"); ax.set_title(title)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = int(counts[i, j])
                if val:
                    ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=6,
                            color="white" if values[i, j] > values.max() * 0.55 else "black")
        fig.colorbar(im, ax=ax, label="log10(count + 1)", fraction=0.046)
        fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=125); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not save raw confusion counts: {exc}")


def plot_classwise_metrics(m: Dict[str, object], path: Path, title: str) -> None:
    """Readable 2x2 per-class IoU/Dice/recall/precision dashboard."""
    plt = _plt()
    if plt is None:
        return
    try:
        metrics = [("iou_per_class", "IoU (%)"), ("dice_per_class", "Dice (%)"),
                   ("recall_per_class", "Recall (%)"), ("precision_per_class", "Precision (%)")]
        x = np.arange(NUM_CLASSES)
        colors = [ds2.PALETTE[i] / 255.0 for i in range(NUM_CLASSES)]
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for ax, (key, subtitle) in zip(axes.flat, metrics):
            vals = np.array([0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else 100 * float(v)
                             for v in m[key]])
            bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.35)
            ax.set_xticks(x); ax.set_xticklabels([f"{i}\n{ds2.CLASS_NAMES[i].split('/')[0]}" for i in x], fontsize=8)
            ax.set_ylim(0, 100); ax.set_ylabel(subtitle); ax.set_title(subtitle)
            ax.grid(axis="y", alpha=0.25)
            for bar, value in zip(bars, vals):
                if value >= 3:
                    ax.text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.0f}", ha="center", va="bottom", fontsize=7)
        fig.suptitle(title, fontsize=14); fig.tight_layout(rect=(0, 0, 1, 0.96))
        path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=125); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not save per-class metric dashboard: {exc}")


def plot_pred_distribution(m: Dict[str, object], path: Path, title: str) -> None:
    """Ground-truth vs predicted pixel share per class (log y) + per-class IoU."""
    plt = _plt()
    if plt is None:
        return
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
        x, w = np.arange(NUM_CLASSES), 0.38
        ax[0].bar(x - w / 2, [100 * v for v in m["gt_share"]], w, label="ground truth")
        ax[0].bar(x + w / 2, [100 * v for v in m["pred_share"]], w, label="prediction")
        ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_ylim(1e-4, 200)
        ax[0].set_xticklabels([f"{i}\n{n.split('/')[0]}" for i, n in enumerate(ds2.CLASS_NAMES)], fontsize=8)
        ax[0].set_ylabel("% of pixels (log)"); ax[0].set_title("Pixel share per class"); ax[0].legend()
        iou = [0.0 if (v is None or math.isnan(v)) else 100 * v for v in m["iou_per_class"]]
        ax[1].bar(x, iou, color=[ds2.PALETTE[i] / 255.0 if i else (0.3, 0.3, 0.3) for i in range(NUM_CLASSES)], edgecolor="k")
        ax[1].set_xticks(x); ax[1].set_ylim(0, 100); ax[1].set_ylabel("IoU %"); ax[1].set_title("Per-class IoU")
        ax[1].set_xticklabels([str(i) for i in range(NUM_CLASSES)])
        fig.suptitle(title); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=105); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not save prediction distribution: {exc}")


@torch.no_grad()
def save_prediction_grid(model, dataset, indices: Sequence[int], path: Path, device, use_amp, title: str,
                         thermal_mean: float, thermal_std: float) -> None:
    """Save detailed, easy-to-read RGB-T segmentation results with overlays and confidence."""
    plt = _plt()
    if plt is None:
        return
    try:
        model.eval()
        n = len(indices)
        if n == 0:
            return
        panel_names = ["RGB", "Thermal", "Ground truth", "Prediction", "Prediction overlay",
                       "Error map", "Max-class confidence"]
        fig, axes = plt.subplots(n, len(panel_names), figsize=(25, 3.7 * n + 1.2), squeeze=False)
        for r, idx in enumerate(indices):
            rgb, th, y = dataset[idx]
            with autocast(device, use_amp):
                logits = model(rgb[None].to(device), th[None].to(device)).float()
                probs = logits.softmax(1)[0]
                confidence, pred = probs.max(0)
                pred = pred.cpu()
                confidence = confidence.cpu().numpy()
            rgb_u8 = ds2.denormalize_rgb(rgb)
            th_u8 = ds2.denormalize_thermal(th, thermal_mean, thermal_std)
            gt_np = y.detach().cpu().numpy().astype(np.int64)
            pred_np = pred.numpy().astype(np.int64)
            gt_rgb = ds2.colorize_mask(y)
            pred_rgb = ds2.colorize_mask(pred)
            overlay = rgb_u8.copy()
            fg_pred = (pred_np > 0) & (pred_np < NUM_CLASSES)
            if np.any(fg_pred):
                overlay[fg_pred] = (0.52 * rgb_u8[fg_pred] + 0.48 * pred_rgb[fg_pred]).astype(np.uint8)
            # Green = correctly predicted labelled foreground; red = wrong/missed foreground;
            # blue = non-zero prediction where the target says class 0 (unlabeled/background).
            err = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
            err[gt_np == 0] = (36, 36, 36)
            foreground = (gt_np > 0) & (gt_np < NUM_CLASSES)
            correct_fg = foreground & (pred_np == gt_np)
            missed_fg = foreground & (pred_np == 0)
            wrong_fg = foreground & (pred_np > 0) & (pred_np != gt_np)
            bg_nonzero = (gt_np == 0) & (pred_np > 0)
            err[correct_fg] = (40, 210, 95)
            err[missed_fg] = (255, 160, 0)
            err[wrong_fg] = (235, 55, 65)
            err[bg_nonzero] = (40, 155, 255)
            err[gt_np == IGNORE_INDEX] = (255, 255, 255)
            panels = [(rgb_u8, None), (th_u8, "gray"), (gt_rgb, None), (pred_rgb, None),
                      (overlay, None), (err, None), (confidence, "magma")]
            for c, ((img, cm), head) in enumerate(zip(panels, panel_names)):
                axes[r, c].imshow(img, cmap=cm, vmin=0 if cm == "magma" else None,
                                  vmax=1 if cm == "magma" else None)
                axes[r, c].axis("off")
                axes[r, c].set_title(head + (f" [{dataset.sample_name(idx)}]" if c == 0 else ""), fontsize=8)
        fig.legend(handles=ds2.legend_handles(), loc="lower center", ncol=5, fontsize=8, frameon=False)
        fig.suptitle(title + " | error: green=correct FG, orange=missed FG, red=wrong FG, blue=class-0 disagreement",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94)); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not save prediction grid: {exc}")


def plot_confusion(conf: np.ndarray, path: Path, title: str) -> None:
    plt = _plt()
    if plt is None:
        return
    try:
        norm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(range(NUM_CLASSES)); ax.set_yticklabels(ds2.CLASS_NAMES)
        ax.set_xlabel("predicted class id"); ax.set_ylabel("ground truth"); ax.set_title(title)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised (recall)")
        fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=105); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not save confusion matrix: {exc}")


# =========================================================================== #
# Arguments
# =========================================================================== #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate RTFNet18V2 on MFNet RGB-T",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data-root", default=None, help="default: auto-detect sibling ../MFNet_dataset (Kaggle raw) first, then ../MFNet_CMX (prepared)")
    g.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), default=[480, 640])
    g.add_argument("--batch-size", type=int, default=2, help="official RTFNet uses 2; on a 4 GB GPU use 1 if OOM")
    g.add_argument("--accum-steps", type=int, default=4, help="effective batch = batch-size x accum-steps")
    g.add_argument("--eval-batch-size", type=int, default=2)
    g.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows; try 2 for speed")
    g.add_argument("--no-augment", action="store_true")
    g.add_argument("--flip-prob", type=float, default=0.5)
    g.add_argument("--crop-rate", type=float, default=0.1)
    g.add_argument("--thermal-norm", choices=ds2.THERMAL_NORMS, default="dataset",
                   help="dataset: train-split thermal mean/std; imagenet_gray: fixed 0.449/0.226 (V1)")
    g.add_argument("--skip-label-scan", action="store_true", help="skip decoding all masks at start-up")
    g.add_argument("--deep-validate", action="store_true", help="also decode every RGB/thermal PNG")
    g = p.add_argument_group("model")
    g.add_argument("--head", choices=m2.HEADS, default="conv1x1",
                   help="conv1x1: plain 1x1 classifier (V2 default); official: BN+residual on class scores (V1)")
    g.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--require-pretrained", action="store_true", help="abort if ImageNet weights cannot be loaded")
    g.add_argument("--weights-path", default=None, help="local torchvision resnet18 .pth (avoids a download)")
    g.add_argument("--thermal-init", choices=["mean", "sum"], default="mean")
    g.add_argument("--bn-mode", choices=["train", "freeze-encoder"], default="train",
                   help="freeze-encoder keeps encoder BatchNorm in eval mode (stabler with tiny batches)")
    g.add_argument("--prior-bias", action="store_true", help="init classifier bias to log(train class share) (conv1x1 only)")
    g = p.add_argument_group("loss")
    g.add_argument("--ce-weight", type=float, default=1.0)
    g.add_argument("--dice-weight", type=float, default=0.0)
    g.add_argument("--class-weights", choices=["none", "inv_sqrt", "median_freq", "enet"], default="inv_sqrt")
    g.add_argument("--max-weight-ratio", type=float, default=20.0, help="max/min class-weight ratio after clipping")
    g.add_argument("--class0-loss", choices=["include", "ignore"], default="include")
    g.add_argument("--select-metric", choices=["auto", "miou_all9", "miou_fg8"], default="auto")
    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=0.01, help="official RTFNet SGD value; ~1e-3..1e-4 for adamw")
    g.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--scheduler", choices=["exp", "cosine", "none"], default="exp",
                   help="exp = lr * lr-decay^epoch (official RTFNet, gamma 0.95)")
    g.add_argument("--lr-decay", type=float, default=0.95)
    g.add_argument("--grad-clip", type=float, default=0.0, help="max grad-norm; 0 disables")
    g = p.add_argument_group("system / output")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="auto")
    g.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    g.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True,
                   help="faster convs for fixed input size; slightly non-deterministic")
    g.add_argument("--output-root", default=str(SCRIPT_DIR))
    g.add_argument("--run-name", default="model19_v2")
    g.add_argument("--resume", nargs="?", const="last", default=None, help="PATH, or last.pth if no value")
    g.add_argument("--force-resume", action="store_true", help="resume even if loss/model settings differ")
    g.add_argument("--overwrite", action="store_true", help="allow starting over in a non-empty run folder")
    g.add_argument("--eval-only", action="store_true")
    g.add_argument("--checkpoint", default=None, help="for --eval-only (default: <ckpt-dir>/best.pth)")
    g.add_argument("--eval-split", choices=["val", "test"], default="test")
    g.add_argument("--skip-test", action="store_true")
    g.add_argument("--vis-every", type=int, default=5)
    g.add_argument("--vis-samples", type=int, default=4)
    g.add_argument("--log-interval", type=int, default=100, help="iterations between prints when tqdm is missing")
    g.add_argument("--max-train-samples", type=int, default=None, help="debug: even subset")
    g.add_argument("--max-eval-samples", type=int, default=None, help="debug: even subset of val/test")
    g.add_argument("--sanity", action="store_true", help="run the real-data sanity checks and exit")
    g.add_argument("--sanity-steps", type=int, default=2)
    a = p.parse_args(argv)
    if min(a.batch_size, a.accum_steps, a.epochs, a.eval_batch_size) < 1:
        p.error("--batch-size, --accum-steps, --eval-batch-size and --epochs must be >= 1")
    if a.run_name in RESERVED_RUN_NAMES:
        p.error(f"--run-name {a.run_name!r} is reserved for V1 / teammates' files; pick another name")
    a.select_metric_resolved = ("miou_all9" if a.class0_loss == "include" else "miou_fg8") \
        if a.select_metric == "auto" else a.select_metric
    return a


def describe_environment(device: torch.device) -> None:
    log(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda available: {torch.cuda.is_available()}")
    try:
        import torchvision
        log(f"torchvision {torchvision.__version__}")
    except Exception:  # noqa: BLE001
        pass
    if device.type == "cuda":
        pr = torch.cuda.get_device_properties(device)
        log(f"GPU: {pr.name} | {pr.total_memory / 2 ** 30:.1f} GiB | compute capability {pr.major}.{pr.minor}")
    else:
        log("Running on CPU (no CUDA device) -- expect very slow training")


# =========================================================================== #
# Data / model construction
# =========================================================================== #
def make_data(args, device, splits, cache_dir: Optional[Path], thermal_override=None):
    return ds2.build_dataloaders(
        data_root=args.data_root, img_size=tuple(args.img_size), batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, num_workers=args.num_workers, seed=args.seed,
        augment_train=not args.no_augment, flip_prob=args.flip_prob, crop_rate=args.crop_rate,
        splits=splits, validate=True, scan_labels=not args.skip_label_scan, deep=args.deep_validate,
        pin_memory=(device.type == "cuda"), thermal_norm=args.thermal_norm,
        thermal_cache=(cache_dir / "thermal_stats.json") if cache_dir else None,
        thermal_override=thermal_override)


def train_label_counts(info, cache_dir: Optional[Path]) -> np.ndarray:
    """Train-split label pixel counts (int64[9]) from the validation scan, or decoded now."""
    rep, root = info["report"], info["root"]
    if rep is not None and "train" in rep.label_pixels:
        return rep.label_pixels["train"][:NUM_CLASSES].astype(np.int64)
    cache = (cache_dir / "train_label_counts.json") if cache_dir else None
    names = ds2.read_split_file(root, "train")[0]
    key = {"root": str(root), "n": len(names)}
    if cache is not None and cache.is_file():
        try:
            c = json.loads(cache.read_text(encoding="utf-8"))
            if c.get("key") == key:
                return np.array(c["counts"], dtype=np.int64)
        except Exception:  # noqa: BLE001
            pass
    counts = ds2.compute_label_counts(root, names)[:NUM_CLASSES]
    if cache is not None:
        cache.write_text(json.dumps({"key": key, "counts": counts.tolist()}), encoding="utf-8")
    return counts


def rebuild_loader(loader: DataLoader, subset_ds, batch_size, shuffle, args):
    return DataLoader(subset_ds, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      pin_memory=loader.pin_memory, drop_last=False, persistent_workers=args.num_workers > 0)


def build_for_args(args, device, pretrained: bool, prior=None):
    model = m2.build_model(NUM_CLASSES, pretrained=pretrained, head=args.head, thermal_init=args.thermal_init,
                           weights_path=args.weights_path, prior_probs=prior,
                           freeze_encoder_bn=(args.bn_mode == "freeze-encoder"), device=device)
    info = model.pretrained_info
    if pretrained and args.require_pretrained and not info["loaded"]:
        raise RuntimeError("--require-pretrained was set but ImageNet weights could not be loaded: " + info["message"])
    log(m2.describe_model(model) + f" | encoder BN: {'frozen (eval)' if model.freeze_encoder_bn else 'train mode'}")
    return model


# =========================================================================== #
# Final / standalone evaluation
# =========================================================================== #
def evaluate_checkpoint(args, ckpt_path: Path, split: str, device, use_amp, datasets, loaders,
                        plots_dir: Path, ckpt_dir: Path, thermal: Tuple[float, float]) -> Dict[str, object]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch_load(ckpt_path, device)
    model = build_for_args(args, device, pretrained=False)
    model.load_state_dict(ckpt["model"])
    criterion = SegmentationLoss(args.ce_weight, args.dice_weight, ckpt.get("class_weights"), args.class0_loss).to(device)
    loader = loaders[split]
    if args.max_eval_samples:
        loader = rebuild_loader(loader, even_subset(datasets[split], args.max_eval_samples), args.eval_batch_size, False, args)
    log(f"Evaluating {ckpt_path.name} (epoch {ckpt.get('epoch', '?')}) on the {split.upper()} split ({len(loader.dataset)} samples)")
    sync(device); t0 = time.perf_counter()
    res = evaluate(model, loader, criterion, device, use_amp, desc=split)
    sync(device); sec = time.perf_counter() - t0
    m = res["metrics"]
    log(f"{split} loss {res['loss']:.4f}   (evaluation time {fmt_time(sec)})")
    log(metrics_table(m))
    out = {"split": split, "checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt.get("epoch"),
           "selected_on": f"validation {ckpt.get('select_metric', args.select_metric_resolved)}",
           "img_size": list(args.img_size), "loss": res["loss"], "eval_seconds": sec,
           "class_names": ds2.CLASS_NAMES, "metrics": m, "confusion_matrix": res["conf"].tolist(),
           "class_weights": ckpt.get("class_weights"), "class0_loss": args.class0_loss, "head": args.head,
           "model_info": ckpt.get("model_info"), "args": {k: v for k, v in vars(args).items()}}
    with open(ckpt_dir / f"{split}_results.json", "w", encoding="utf-8") as f:
        json.dump(nan_to_none(out), f, indent=2, default=str)
    plot_confusion(res["conf"], plots_dir / f"{split}_confusion_matrix.png", f"{split} row-normalised confusion ({ckpt_path.name})")
    plot_confusion_counts(res["conf"], plots_dir / f"{split}_confusion_counts.png", f"{split} raw confusion counts ({ckpt_path.name})")
    plot_pred_distribution(m, plots_dir / f"{split}_pred_distribution.png", f"{split}: ground truth vs predicted pixel share ({ckpt_path.name})")
    plot_classwise_metrics(m, plots_dir / f"{split}_classwise_metrics.png", f"{split}: per-class segmentation metrics")
    # A tidy per-class table is useful for the report and model-to-model comparison.
    class_csv = ckpt_dir / f"{split}_class_metrics.csv"
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "IoU", "Dice", "Recall", "Precision", "GT_share", "Pred_share"])
        for cid, cname in enumerate(ds2.CLASS_NAMES):
            writer.writerow([cid, cname, m["iou_per_class"][cid], m["dice_per_class"][cid],
                             m["recall_per_class"][cid], m["precision_per_class"][cid],
                             m["gt_share"][cid], m["pred_share"][cid]])
    ds = datasets[split]
    idx = np.unique(np.linspace(0, len(ds) - 1, num=args.vis_samples).round().astype(int)).tolist()
    save_prediction_grid(model, ds, idx, plots_dir / f"{split}_predictions.png", device, use_amp,
                         f"{split} predictions ({ckpt_path.name})", *thermal)
    log(f"Saved {ckpt_dir / (split + '_results.json')}, {class_csv}, and visualizations in {plots_dir}")
    out["_seconds"] = sec
    return out


# =========================================================================== #
# Sanity mode
# =========================================================================== #
def run_sanity(args, device, use_amp) -> int:
    results: List[Tuple[str, str]] = []
    state: Dict[str, object] = {}
    H, W = args.img_size

    def check(name, fn):
        try:
            detail = fn() or ""
            results.append((name, "PASS")); log(f"[PASS] {name}: {detail}")
            return True
        except Exception as exc:  # noqa: BLE001
            results.append((name, "FAIL")); log(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return False

    def c_data():
        datasets, loaders, info = make_data(args, device, ds2.SPLITS, None)
        state.update(datasets=datasets, loaders=loaders, info=info)
        return "counts " + ", ".join(f"{s}={len(d)}" for s, d in datasets.items())

    def c_sample():
        for split in ("train", "val"):
            rgb, th, y = state["datasets"][split][0]
            assert tuple(rgb.shape) == (3, H, W) and rgb.dtype == torch.float32, (rgb.shape, rgb.dtype)
            assert tuple(th.shape) == (1, H, W) and th.dtype == torch.float32, (th.shape, th.dtype)
            assert tuple(y.shape) == (H, W) and y.dtype == torch.int64
            ids = torch.unique(y).tolist()
            assert all(i in range(NUM_CLASSES) or i == IGNORE_INDEX for i in ids), ids
            assert torch.isfinite(rgb).all() and torch.isfinite(th).all()
        return (f"rgb/thermal/mask aligned, ids valid; thermal tensor mean {float(th.mean()):+.2f} std {float(th.std()):.2f} "
                "(expect roughly 0 / 1 with --thermal-norm dataset)")

    def c_policy():
        counts = train_label_counts(state["info"], None)
        w = compute_class_weights(counts, args.class_weights, args.max_weight_ratio, args.class0_loss)
        state.update(counts=counts, weights=w)
        describe_policies(args, w, counts, args.select_metric_resolved)
        assert w is None or (np.all(np.isfinite(w)) and w.max() / w.min() <= args.max_weight_ratio * 1.001 + 1e-9 or args.class0_loss == "ignore")
        return f"train class-0 share {100 * counts[0] / counts.sum():.2f}%"

    def c_model():
        prior = (counts_to_prior(state["counts"]) if args.prior_bias else None)
        state["model"] = build_for_args(args, device, args.pretrained, prior)
        return state["model"].pretrained_info["message"]

    def c_forward():
        rgb, th, y = next(iter(state["loaders"]["train"]))
        state["batch"] = (rgb.to(device), th.to(device), y.to(device))
        model = state["model"].set_train_mode()
        with autocast(device, use_amp):
            out = model(state["batch"][0], state["batch"][1])
        assert tuple(out.shape) == (rgb.shape[0], NUM_CLASSES, H, W), tuple(out.shape)
        state["logits"] = out
        return f"logits {tuple(out.shape)} dtype {out.dtype}"

    def c_loss():
        det = []
        for ce, dice, c0 in [(1, 0, "include"), (1, 1, "include"), (1, 1, "ignore")]:
            crit = SegmentationLoss(ce, dice, state["weights"], c0).to(device)
            loss, parts = crit(state["logits"].detach(), state["batch"][2])
            assert torch.isfinite(loss), f"non-finite loss for ce={ce} dice={dice} class0={c0}"
            det.append(f"[ce={ce},dice={dice},class0={c0}] {float(loss):.4f}")
        state["crit"] = SegmentationLoss(args.ce_weight, args.dice_weight, state["weights"], args.class0_loss).to(device)
        # an all-class-0 target under 'ignore' must give a finite zero loss, not NaN
        crit = SegmentationLoss(1, 1, None, "ignore").to(device)
        z, _ = crit(state["logits"].detach(), torch.zeros_like(state["batch"][2]))
        assert torch.isfinite(z), "all-ignored batch produced a non-finite loss"
        return " ".join(det)

    def c_backward():
        res = m2.loss_backward_test(state["model"], state["batch"][0], state["batch"][1], state["batch"][2], state["crit"])
        return f"loss {res['loss']:.4f}, non-zero grad tensors {res['grad_tensors_nonzero']}"

    def c_mini_train():
        n_acc = min(args.accum_steps, 2)
        ds = even_subset(state["datasets"]["train"], args.sanity_steps * n_acc * args.batch_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        model = state["model"]
        before = {k: v.detach().clone() for k, v in list(model.state_dict().items())[:4]}
        opt, scaler = build_optimizer(model, args), make_scaler(use_amp)
        sync(device); t0 = time.perf_counter()
        loss = train_one_epoch(model, loader, state["crit"], opt, scaler, device, use_amp, n_acc, args.grad_clip, 1, 1, 1)
        assert math.isfinite(loss)
        assert any(not torch.equal(before[k], v) for k, v in list(model.state_dict().items())[:4]), "weights did not change"
        val_ds = even_subset(state["datasets"]["val"], 4)
        res = evaluate(model, DataLoader(val_ds, batch_size=args.eval_batch_size), state["crit"], device, use_amp)
        assert res["conf"].sum() == len(val_ds) * H * W, "confusion matrix pixel count mismatch"
        sync(device)
        mem = f", peak GPU mem {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GiB" if device.type == "cuda" else ""
        state["res"] = res
        return (f"{len(ds)} samples, loss {loss:.4f}; val(4) loss {res['loss']:.4f} mIoU9 {100 * res['metrics']['miou_all9']:.2f}%{mem} "
                f"({time.perf_counter() - t0:.1f}s)")

    def c_speed():
        model, crit = state["model"], state["crit"]
        opt, scaler = build_optimizer(model, args), make_scaler(use_amp)
        ds = even_subset(state["datasets"]["train"], args.batch_size * 5)
        batches = list(DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0))
        model.set_train_mode(); times, n = [], 0
        for k, (rgb, th, y) in enumerate(batches):
            sync(device); t0 = time.perf_counter()
            rgb, th, y = rgb.to(device), th.to(device), y.to(device)
            with autocast(device, use_amp):
                out = model(rgb, th)
            loss, _ = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sync(device)
            if k > 0:                                    # first iteration = warm-up / cudnn autotune
                times.append((time.perf_counter() - t0) / rgb.shape[0])
        per_sample = float(np.mean(times))
        n_train = len(state["datasets"]["train"])
        return (f"{per_sample:.3f} s/sample measured over {len(times)} steps (warm-up excluded) -> a train epoch of {n_train} samples "
                f"would take roughly {fmt_time(per_sample * n_train)} (rough projection from a few steps, excludes data-loading "
                "hiccups, validation and checkpointing; the real per-epoch timing is printed during training)")

    def c_ckpt():
        model = state["model"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "last.pth"
            save_checkpoint(path, {"epoch": 1, "model": model.state_dict()})
            ck = torch_load(path, device)
            clone = m2.build_model(NUM_CLASSES, pretrained=False, head=args.head, device=device)
            clone.load_state_dict(ck["model"])
            rgb, th, _ = state["batch"]
            model.eval(); clone.eval()
            with torch.no_grad():
                d = (model(rgb, th) - clone(rgb, th)).abs().max().item()
        assert d < 1e-3, f"reloaded model differs by {d}"
        return f"save/load round-trip max |diff| = {d:.2e}"

    def c_vis():
        if _plt() is None:
            return "skipped (matplotlib missing)"
        info = state["info"]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "grid.png"
            vis_ds = state["datasets"]["val"]
            vis_indices = list(range(min(2, len(vis_ds))))
            save_prediction_grid(state["model"], vis_ds, vis_indices, out, device, use_amp, "sanity",
                                 info["thermal_mean"], info["thermal_std"])
            assert out.is_file() and out.stat().st_size > 1000, "prediction grid not written"
            plot_pred_distribution(state["res"]["metrics"], Path(td) / "dist.png", "sanity")
            plot_confusion(state["res"]["conf"], Path(td) / "conf.png", "sanity")
            assert (Path(td) / "dist.png").is_file() and (Path(td) / "conf.png").is_file()
        return "prediction grid, distribution and confusion plots rendered"

    def c_timing():
        eta, mean_all, recent = estimate_eta([100.0, 110.0, 90.0, 120.0, 100.0, 100.0], 94)
        assert abs(mean_all - 103.3333) < 1e-3 and abs(recent - 104.0) < 1e-9 and abs(eta - 104.0 * 94) < 1e-6
        assert fmt_time(3 * 3600 + 25 * 60 + 9) == "03:25:09" and fmt_time(100 * 3600) == "100:00:00"
        return "ETA/duration formatting self-test on fixed numbers (not a measurement)"

    ok = True
    for name, fn in [("dataset validation + loaders", c_data), ("train/val sample shapes, ids, normalisation", c_sample),
                     ("loss policy + class weights", c_policy), ("build model", c_model),
                     ("forward pass on real batch", c_forward), ("finite loss on real labels (3 policies)", c_loss),
                     ("backward + gradients through both encoders and decoder", c_backward),
                     ("mini training + validation metrics", c_mini_train), ("speed probe", c_speed),
                     ("checkpoint save/load round-trip", c_ckpt), ("visualisation", c_vis), ("timing helpers", c_timing)]:
        ok &= check(name, fn)
        if not ok and name in ("dataset validation + loaders", "loss policy + class weights", "build model"):
            log("Cannot continue after this failure.")
            break
    log("-" * 72)
    for n, s in results:
        log(f"{s:<5} {n}")
    log("SANITY " + ("PASSED" if ok and all(s == "PASS" for _, s in results) else "FAILED"))
    return 0 if ok else 1


def counts_to_prior(counts9: np.ndarray) -> List[float]:
    c = counts9.astype(np.float64)
    return (c / c.sum()).tolist()


# =========================================================================== #
# Run bookkeeping (for the final / interrupted summary)
# =========================================================================== #
@dataclass
class RunState:
    t0: float = field(default_factory=time.perf_counter)
    start_dt: datetime.datetime = field(default_factory=datetime.datetime.now)
    ckpt_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None
    epochs_done: int = 0
    best_epoch: int = -1
    best_score: float = float("nan")
    select_metric: str = ""
    train_sec: float = 0.0
    val_sec: float = 0.0
    other_sec: float = 0.0
    test_sec: float = 0.0
    epoch_totals: List[float] = field(default_factory=list)
    test_out: Optional[Dict[str, object]] = None
    finished_training: bool = False


def checkpoint_status(run: RunState) -> str:
    if run.ckpt_dir is None:
        return "no checkpoint folder (sanity / eval-only)"
    parts = []
    for n in ("last.pth", "best.pth"):
        p = run.ckpt_dir / n
        parts.append(f"{n}: " + (f"present ({datetime.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S})" if p.is_file() else "missing"))
    return " | ".join(parts)


def print_final_summary(run: RunState, args, status: str) -> None:
    total = time.perf_counter() - run.t0
    n = max(run.epochs_done, 1)
    log("=" * 72)
    log(f"RUN SUMMARY ({status})")
    log(f"  started               : {fmt_clock(run.start_dt)}   ended: {fmt_clock(datetime.datetime.now())}")
    if run.epochs_done:
        log(f"  epochs completed      : {run.epochs_done} in this session")
        log(f"  training wall-clock   : {fmt_time(run.train_sec + run.val_sec + run.other_sec)}  "
            f"(train {fmt_time(run.train_sec)} + validation {fmt_time(run.val_sec)} + ckpt/plots {fmt_time(run.other_sec)})")
        log(f"  total validation time : {fmt_time(run.val_sec)}")
        log(f"  mean epoch duration   : {fmt_time(float(np.mean(run.epoch_totals)))}  (min {fmt_time(min(run.epoch_totals))}, max {fmt_time(max(run.epoch_totals))})")
    if run.test_sec:
        log(f"  test evaluation time  : {fmt_time(run.test_sec)}")
    log(f"  end-to-end runtime    : {fmt_time(total)}")
    if run.best_epoch >= 0:
        log(f"  best validation {run.select_metric}: {100 * run.best_score:.2f}% at epoch {run.best_epoch}")
    if run.test_out is not None:
        tm = run.test_out["metrics"]
        log(f"  TEST (best.pth, epoch {run.test_out['checkpoint_epoch']}): mIoU(9) {100 * tm['miou_all9']:.2f}% | mIoU(1-8) {100 * tm['miou_fg8']:.2f}% | "
            f"pixel acc {100 * tm['pixel_acc']:.2f}% | Dice(9) {100 * tm['mean_dice_all9']:.2f}%")
    if run.ckpt_dir is not None:
        log(f"  checkpoints           : {checkpoint_status(run)}")
        log(f"  best checkpoint       : {run.ckpt_dir / 'best.pth'}")
        log(f"  log / history         : {run.ckpt_dir / 'train_log.txt'} , {run.ckpt_dir / 'history.csv'}")
        log(f"  plots                 : {run.plots_dir}")
    log("=" * 72)


# =========================================================================== #
# Main
# =========================================================================== #
def override_args_from_ckpt(args, ckpt_path: Path) -> None:
    ck = torch_load(ckpt_path, "cpu")
    ca = ck.get("args", {})
    changed = []
    for k in CKPT_ARG_KEYS:
        if k in ca and getattr(args, k, None) != ca[k]:
            changed.append(f"{k}: {getattr(args, k, None)} -> {ca[k]}")
            setattr(args, k, ca[k])
    if "select_metric" in ck:
        args.select_metric_resolved = ck["select_metric"]
    if changed:
        log("Using settings stored in the checkpoint: " + "; ".join(changed))
    args._ckpt_thermal = (ck.get("thermal_mean"), ck.get("thermal_std"))


def _run(args, run: RunState) -> int:
    out_root = Path(args.output_root).resolve()
    ckpt_dir = out_root / "checkpoints" / args.run_name
    plots_dir = out_root / "plots" / args.run_name
    if not args.sanity:
        if not args.eval_only and not args.resume and not args.overwrite and \
                ((ckpt_dir / "last.pth").is_file() or (ckpt_dir / "best.pth").is_file()):
            raise SystemExit(f"{ckpt_dir} already contains checkpoints.  Use --resume to continue, a new --run-name, "
                             "or --overwrite to start over (V1 checkpoints are in a different folder and are never touched).")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        log.attach(ckpt_dir / "train_log.txt")
        run.ckpt_dir, run.plots_dir = ckpt_dir, plots_dir
    log(f"===== train19_v2 started {run.start_dt:%Y-%m-%d %H:%M:%S} =====")
    set_seed(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and args.amp != "off"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    if args.amp == "on" and device.type != "cuda":
        log("NOTE: --amp on ignored (CPU)")
    describe_environment(device)
    log(f"AMP (fp16 autocast): {use_amp} | seed {args.seed} | cudnn.benchmark {torch.backends.cudnn.benchmark} | "
        f"workers {args.num_workers} | head {args.head} | encoder BN {args.bn_mode} | thermal norm {args.thermal_norm}")

    if args.sanity:
        return run_sanity(args, device, use_amp)

    # ---------------- evaluation only ---------------- #
    if args.eval_only:
        ck_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "best.pth"
        if not ck_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ck_path}")
        override_args_from_ckpt(args, ck_path)
        tm, ts = getattr(args, "_ckpt_thermal", (None, None))
        override = (tm, ts) if tm is not None else None
        datasets, loaders, info = make_data(args, device, ("val", "test"), ckpt_dir, override)
        t0 = time.perf_counter()
        evaluate_checkpoint(args, ck_path, args.eval_split, device, use_amp, datasets, loaders, plots_dir, ckpt_dir,
                            (info["thermal_mean"], info["thermal_std"]))
        log(f"Evaluation finished in {fmt_time(time.perf_counter() - t0)}")
        return 0

    # ---------------- training ---------------- #
    resume_path = None
    ck = None
    if args.resume is not None:
        resume_path = ckpt_dir / "last.pth" if args.resume == "last" else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"cannot resume: {resume_path} not found")
        ck = torch_load(resume_path, device)
        diffs = [f"{k}: checkpoint={ck['args'].get(k)} now={getattr(args, k, None)}" for k in CKPT_ARG_KEYS
                 if k in ck.get("args", {}) and ck["args"][k] != getattr(args, k, None)]
        if diffs and not args.force_resume:
            raise SystemExit("Refusing to resume: these settings differ from the checkpoint (use the same flags, or --force-resume):\n  "
                             + "\n  ".join(diffs))
        if diffs:
            log("WARNING (--force-resume): settings differ from the checkpoint: " + "; ".join(diffs))

    override = (ck["thermal_mean"], ck["thermal_std"]) if ck is not None and ck.get("thermal_mean") is not None else None
    datasets, loaders, info = make_data(args, device, ds2.SPLITS, ckpt_dir, override)
    counts = train_label_counts(info, ckpt_dir)
    weights = compute_class_weights(counts, args.class_weights, args.max_weight_ratio, args.class0_loss)
    if ck is not None and ck.get("class_weights") is not None and weights is not None:
        weights = np.array(ck["class_weights"], dtype=np.float64)       # keep resumed runs identical
    select_metric = args.select_metric_resolved
    run.select_metric = select_metric
    describe_policies(args, weights, counts, select_metric)

    prior = counts_to_prior(counts) if args.prior_bias else None
    model = build_for_args(args, device, pretrained=args.pretrained and ck is None, prior=prior)
    criterion = SegmentationLoss(args.ce_weight, args.dice_weight, weights, args.class0_loss).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = make_scaler(use_amp)

    train_loader = loaders["train"]
    if args.max_train_samples:
        train_loader = rebuild_loader(train_loader, even_subset(datasets["train"], args.max_train_samples), args.batch_size, True, args)
    val_loader = loaders["val"]
    if args.max_eval_samples:
        val_loader = rebuild_loader(val_loader, even_subset(datasets["val"], args.max_eval_samples), args.eval_batch_size, False, args)

    hist_keys = ["epoch", "train_loss", "val_loss", "val_miou_all9", "val_miou_fg8", "val_pixel_acc",
                 "val_mean_dice_all9", "val_mean_dice_fg8", "lr", "epoch_seconds", "train_seconds", "val_seconds",
                 "val_iou_per_class", "val_pred_share", "val_gt_share0"]
    history = {k: [] for k in hist_keys}
    start_epoch, best_score, best_epoch = 0, -float("inf"), -1
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None and use_amp:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, best_score, best_epoch, history = ck["epoch"], ck["best_score"], ck["best_epoch"], ck["history"]
        for k in hist_keys:
            history.setdefault(k, [])
        log(f"Resumed from {resume_path} (completed epochs: {start_epoch}, best {select_metric} {100 * best_score:.2f}% @ epoch {best_epoch}). "
            "Timing/ETA below are for THIS session only.")
        if start_epoch >= args.epochs:
            log(f"Nothing to train: checkpoint already has {start_epoch} >= --epochs {args.epochs}.")
    if best_epoch >= 0:
        run.best_epoch, run.best_score = best_epoch, best_score

    log(f"Training: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples | batch {args.batch_size} x accum "
        f"{args.accum_steps} = effective {args.batch_size * args.accum_steps} | optimizer {args.optimizer} lr {args.lr} "
        f"({args.scheduler}, gamma {args.lr_decay}) | epochs {start_epoch + 1}..{args.epochs}")
    log(f"Run started at {fmt_clock(run.start_dt)}")

    vis_idx = np.unique(np.linspace(0, len(datasets["val"]) - 1, num=args.vis_samples).round().astype(int)).tolist()
    thermal = (info["thermal_mean"], info["thermal_std"])
    for epoch in range(start_epoch + 1, args.epochs + 1):
        sync(device); t_ep = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp,
                                  args.accum_steps, args.grad_clip, epoch, args.epochs, args.log_interval)
        sync(device); t_tr = time.perf_counter()
        val = evaluate(model, val_loader, criterion, device, use_amp, desc=f"val {epoch}/{args.epochs}")
        sync(device); t_va = time.perf_counter()
        scheduler.step()
        m = val["metrics"]
        score = metric_value(m, select_metric)
        is_best = math.isfinite(score) and score > best_score
        if is_best:
            best_score, best_epoch = score, epoch
            run.best_epoch, run.best_score = epoch, score

        train_sec, val_sec = t_tr - t_ep, t_va - t_tr
        for k, v in [("epoch", epoch), ("train_loss", tr_loss), ("val_loss", val["loss"]), ("val_miou_all9", m["miou_all9"]),
                     ("val_miou_fg8", m["miou_fg8"]), ("val_pixel_acc", m["pixel_acc"]),
                     ("val_mean_dice_all9", m["mean_dice_all9"]), ("val_mean_dice_fg8", m["mean_dice_fg8"]), ("lr", lr_now),
                     ("train_seconds", train_sec), ("val_seconds", val_sec), ("val_iou_per_class", m["iou_per_class"]),
                     ("val_pred_share", m["pred_share"]), ("val_gt_share0", m["gt_share"][0])]:
            history[k].append(v)
        history["epoch_seconds"].append(train_sec + val_sec)       # patched below with the true total

        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if use_amp else None,
                 "best_score": best_score, "best_epoch": best_epoch, "history": history, "args": vars(args),
                 "select_metric": select_metric, "class_weights": None if weights is None else weights.tolist(),
                 "thermal_mean": thermal[0], "thermal_std": thermal[1], "model_info": model.pretrained_info,
                 "val_metrics": m}
        save_checkpoint(ckpt_dir / "last.pth", state)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pth", state)
        with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(nan_to_none(history), f, indent=1)
        with open(ckpt_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            cols = [k for k in history if k not in ("val_iou_per_class", "val_pred_share")]
            w = csv.writer(f); w.writerow(cols)
            for i in range(len(history["epoch"])):
                w.writerow([history[c][i] for c in cols])
        plot_curves(history, plots_dir / "training_curves.png", select_metric)
        if args.vis_every > 0 and (epoch % args.vis_every == 0 or epoch == args.epochs or epoch == start_epoch + 1):
            save_prediction_grid(model, datasets["val"], vis_idx, plots_dir / f"val_predictions_epoch{epoch:03d}.png",
                                 device, use_amp, f"Validation predictions - epoch {epoch}", *thermal)
            plot_pred_distribution(m, plots_dir / f"val_pred_distribution_epoch{epoch:03d}.png",
                                   f"Validation, epoch {epoch}: ground-truth vs predicted pixel share")
            plot_iou_history(history, plots_dir / "val_per_class_iou_heatmap.png")
        sync(device)
        t_end = time.perf_counter()
        total_sec, other_sec = t_end - t_ep, t_end - t_va
        history["epoch_seconds"][-1] = total_sec
        run.epoch_totals.append(total_sec)
        run.epochs_done += 1
        run.train_sec += train_sec; run.val_sec += val_sec; run.other_sec += other_sec

        elapsed = t_end - run.t0
        eta, mean_all, mean_recent = estimate_eta(run.epoch_totals, args.epochs - epoch, window=5)
        finish = datetime.datetime.now() + datetime.timedelta(seconds=eta)
        share = lambda key: " ".join(f"{i}:{100 * v:.2f}" for i, v in enumerate(m[key]))
        collapse = m["pred_share"][0] > 0.98 and m["gt_share"][0] < 0.97
        log("\n".join([
            f"Epoch {epoch:03d}/{args.epochs}" + ("   * new best" if is_best else ""),
            f"Compute device:   {live_device_status(device)}",
            f"Train time:       {fmt_time(train_sec)}",
            f"Validation time:  {fmt_time(val_sec)}",
            f"Other (ckpt/plots): {fmt_time(other_sec)}",
            f"Epoch total:      {fmt_time(total_sec)}",
            f"Average epoch:    {fmt_time(mean_all)}   (mean of last 5: {fmt_time(mean_recent)})",
            f"Total elapsed:    {fmt_time(elapsed)}",
            f"ETA remaining:    {fmt_time(eta)}   (recent mean x {args.epochs - epoch} epochs left; test evaluation not included)",
            f"Expected finish:  {fmt_clock(finish)}",
            f"Train loss:       {tr_loss:.4f}",
            f"Validation loss:  {val['loss']:.4f}",
            f"mIoU (9):         {100 * m['miou_all9']:.2f}%",
            f"mIoU (1-8):       {100 * m['miou_fg8']:.2f}%",
            f"Pixel acc / Dice(9) / Dice(1-8): {100 * m['pixel_acc']:.2f}% / {100 * m['mean_dice_all9']:.2f}% / {100 * m['mean_dice_fg8']:.2f}%",
            f"Learning rate:    {lr_now:.3e}",
            f"Per-class IoU %:  " + " ".join(f"{i}:{_pct(v).strip()}" for i, v in enumerate(m["iou_per_class"])),
            f"GT share %:       {share('gt_share')}",
            f"Pred share %:     {share('pred_share')}" + ("   <-- WARNING: predictions collapsed to class 0" if collapse else ""),
            "-" * 72]))

    run.finished_training = True
    if best_epoch < 0:
        raise RuntimeError("no valid best checkpoint was produced (validation metric was never finite)")
    last_iou = history["val_iou_per_class"][-1]
    log("Per-class validation IoU at the last epoch: " + ", ".join(
        f"{ds2.CLASS_NAMES[i]} {'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:.1f}'}" for i, v in enumerate(last_iou)))
    if not args.skip_test:
        log("Final evaluation of best.pth on the TEST split (not used for model selection or tuning)")
        out = evaluate_checkpoint(args, ckpt_dir / "best.pth", "test", device, use_amp, datasets, loaders, plots_dir, ckpt_dir, thermal)
        run.test_out, run.test_sec = out, out["_seconds"]
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    run = RunState()
    status, rc = "completed", 0
    try:
        rc = _run(args, run)
        if rc != 0:
            status = f"exit code {rc}"
    except KeyboardInterrupt:
        status, rc = "INTERRUPTED by user", 130
        log(f"Interrupted by user after {fmt_time(time.perf_counter() - run.t0)}. Resume with --resume (uses last.pth).")
    except SystemExit as exc:
        log(str(exc.code) if isinstance(exc.code, str) else "")
        return 1 if exc.code else 0
    except Exception as exc:  # noqa: BLE001
        if is_oom(exc):
            log(f"ERROR: {type(exc).__name__}: {exc}\n{OOM_HELP}")
            status, rc = "stopped: CUDA out of memory", 2
        elif isinstance(exc, ds2.DatasetValidationError):
            log(f"ERROR: {exc}\nFix the dataset (or pass --data-root) before training.")
            status, rc = "stopped: dataset validation failed", 3
        else:
            log("ERROR: " + "".join(traceback.format_exception_only(type(exc), exc)).strip())
            traceback.print_exc()
            status, rc = f"stopped: {type(exc).__name__}", 1
    if not args.sanity and (run.epochs_done or run.ckpt_dir is not None):
        print_final_summary(run, args, status)
    return rc


if __name__ == "__main__":
    sys.exit(main())
