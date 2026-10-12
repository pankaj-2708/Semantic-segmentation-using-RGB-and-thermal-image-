"""
train.py - Semantic Segmentation Training Pipeline for MFNet (RGB + Thermal).

Features:
- MFNet model with 4-channel (RGB + Thermal) input
- Step-level and epoch-level tracking with confusion-matrix metrics
- Live tqdm progress bars with running averages
- Combined Cross-Entropy + Dice loss respecting IGNORE_INDEX
- Comprehensive checkpointing (per-epoch, last, best)
- Automated training-curve plots
- Auto-download of MFNet dataset via gdown if not present
"""

import os
import sys
import copy
import random
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import modular components
try:
    import model as model_module
    from model import MFNet, build_model
except ImportError as e:
    print(f"[Warning] Could not import model: {e}")
    model_module = None
    MFNet = None
    build_model = None

try:
    import dataset as dataset_module
    from dataset import get_dataloaders, NUM_CLASSES as DS_NUM_CLASSES
except ImportError as e:
    print(f"[Warning] Could not import dataset: {e}")
    dataset_module = None
    get_dataloaders = None
    DS_NUM_CLASSES = 9

# ==============================================================================
# Global Configuration
# ==============================================================================
LOG_EVERY_N_STEPS = 1
TQDM_SMOOTHING    = 0.1

NUM_CLASSES   = 9          # MFNet: 0=Unlabeled, 1-8 objects
IN_CHANNELS   = 4          # RGB (3) + Thermal (1)
IGNORE_INDEX  = 255
BATCH_SIZE    = 4
EPOCHS        = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
ARCH          = "mfnet"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR    = "checkpoints"
PLOT_DIR          = "plots"
BEST_METRIC_NAME  = "miou"   # "miou" or "loss"

# Default dataset path (relative to cwd when running train.py)
DEFAULT_DATA_ROOT = "../MFNet_dataset"

CLASS_NAMES = [
    "Unlabeled", "Car", "Person", "Bike",
    "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump",
]


# ==============================================================================
# Configuration Snapshot
# ==============================================================================
def get_config_snapshot() -> Dict[str, Dict[str, Any]]:
    """Returns module-level uppercase globals for train, dataset, and model."""
    snapshot = {}

    snapshot["train"] = {
        k: v for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
        and isinstance(v, (int, float, str, bool, list, tuple, dict))
    }

    if dataset_module is not None:
        snapshot["dataset"] = {
            k: v for k, v in vars(dataset_module).items()
            if k.isupper() and not k.startswith("_")
            and isinstance(v, (int, float, str, bool, list, tuple, dict))
        }
    else:
        snapshot["dataset"] = {}

    if model_module is not None:
        snapshot["model"] = {
            k: v for k, v in vars(model_module).items()
            if k.isupper() and not k.startswith("_")
            and isinstance(v, (int, float, str, bool, list, tuple, dict))
        }
    else:
        snapshot["model"] = {}

    return snapshot


# ==============================================================================
# Segmentation Metrics (Confusion-Matrix Based)
# ==============================================================================
class SegmentationMetrics:
    """
    Accumulates a confusion matrix over an epoch and derives:
      - Mean IoU (mIoU)
      - Pixel Accuracy
      - Mean Dice / F1
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: Optional[int] = IGNORE_INDEX,
        device: Optional[torch.device] = None,
    ):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.device       = device
        self.conf_matrix  = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )
        if self.device is not None:
            self.conf_matrix = self.conf_matrix.to(self.device)

    def reset(self):
        self.conf_matrix.zero_()

    def _batch_conf(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)

        preds_f   = preds.view(-1)
        targets_f = targets.view(-1)

        valid = (targets_f >= 0) & (targets_f < self.num_classes)
        if self.ignore_index is not None:
            valid = valid & (targets_f != self.ignore_index)
        valid = valid & (preds_f >= 0) & (preds_f < self.num_classes)

        tgt_v = targets_f[valid].long()
        prd_v = preds_f[valid].long()

        if tgt_v.numel() == 0:
            return torch.zeros(
                (self.num_classes, self.num_classes),
                dtype=torch.int64, device=preds.device,
            )

        indices = tgt_v * self.num_classes + prd_v
        conf = torch.bincount(indices, minlength=self.num_classes ** 2)
        return conf.reshape(self.num_classes, self.num_classes).to(torch.int64)

    @classmethod
    def _metrics_from_conf(cls, conf: torch.Tensor, eps: float = 1e-7) -> Dict[str, Any]:
        cf = conf.float()
        tp       = torch.diag(cf)
        gt_sum   = cf.sum(dim=1)   # TP + FN
        pred_sum = cf.sum(dim=0)   # TP + FP
        union    = gt_sum + pred_sum - tp
        total_px = cf.sum()

        pixel_acc = (tp.sum() / (total_px + eps)).item() if total_px > 0 else 0.0

        valid_iou = union > 0
        iou_per_class = torch.zeros(conf.shape[0], device=conf.device, dtype=torch.float32)
        if valid_iou.any():
            iou_per_class[valid_iou] = tp[valid_iou] / (union[valid_iou] + eps)
            miou = iou_per_class[valid_iou].mean().item()
        else:
            miou = 0.0

        dice_denom  = gt_sum + pred_sum
        valid_dice  = dice_denom > 0
        dice_per_class = torch.zeros(conf.shape[0], device=conf.device, dtype=torch.float32)
        if valid_dice.any():
            dice_per_class[valid_dice] = (2.0 * tp[valid_dice]) / (dice_denom[valid_dice] + eps)
            dice = dice_per_class[valid_dice].mean().item()
        else:
            dice = 0.0

        return {
            "miou":           float(miou),
            "pixel_acc":      float(pixel_acc),
            "dice":           float(dice),
            "iou_per_class":  [float(x) for x in iou_per_class.cpu().tolist()],
            "dice_per_class": [float(x) for x in dice_per_class.cpu().tolist()],
        }

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        batch_conf = self._batch_conf(preds, targets)
        if self.conf_matrix.device != batch_conf.device:
            self.conf_matrix = self.conf_matrix.to(batch_conf.device)
        self.conf_matrix += batch_conf
        return self._metrics_from_conf(batch_conf)

    def compute(self) -> Dict[str, Any]:
        return self._metrics_from_conf(self.conf_matrix)


# ==============================================================================
# Loss Functions
# ==============================================================================
class MulticlassDiceLoss(nn.Module):
    """Multiclass Dice Loss with IGNORE_INDEX support."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: Optional[int] = IGNORE_INDEX,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.eps          = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = F.softmax(logits, dim=1)
        tgt_cln = targets.clone()

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            tgt_cln[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        one_hot = (
            F.one_hot(tgt_cln.long(), num_classes=self.num_classes)
            .permute(0, 3, 1, 2).float()
        )

        if self.ignore_index is not None:
            mask_exp  = valid_mask.unsqueeze(1).float()
            probs     = probs * mask_exp
            one_hot   = one_hot * mask_exp

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality  = (probs + one_hot).sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy + Dice Loss."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: Optional[int] = IGNORE_INDEX,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        ce_ign = ignore_index if ignore_index is not None else -100
        self.ce          = nn.CrossEntropyLoss(ignore_index=ce_ign)
        self.dice        = MulticlassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


# ==============================================================================
# Checkpointing
# ==============================================================================
def save_checkpoint(
    filepath: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    history: Dict[str, list],
    best_metric_val: float,
    best_epoch: int,
):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch":               epoch,
            "global_step":         global_step,
            "model_state_dict":    raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "history":             copy.deepcopy(history),
            "best_metric_val":     best_metric_val,
            "best_epoch":          best_epoch,
            "config_snapshot":     get_config_snapshot(),
            "rng_state": {
                "torch":  torch.get_rng_state(),
                "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy":  np.random.get_state(),
                "python": random.getstate(),
            },
        },
        filepath,
    )


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = DEVICE,
) -> Tuple[int, int, Dict[str, list], float, int]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: '{filepath}'")

    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    raw  = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state_dict"])

    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    rng = ckpt.get("rng_state", {})
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8) if isinstance(rng["torch"], torch.Tensor) else rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception as e:
        print(f"[Warning] Could not restore RNG states: {e}")

    epoch          = ckpt.get("epoch", 0)
    global_step    = ckpt.get("global_step", 0)
    history        = ckpt.get("history", {"train_step": [], "val_step": [], "train_epoch": [], "val_epoch": []})
    best_metric    = ckpt.get("best_metric_val", -float("inf"))
    best_epoch     = ckpt.get("best_epoch", -1)

    print(f"--> Loaded checkpoint '{filepath}' (epoch={epoch}, step={global_step})")
    return epoch + 1, global_step, history, best_metric, best_epoch


# ==============================================================================
# Plotting
# ==============================================================================
def plot_and_save_training_curves(history: Dict[str, list], output_dir: str = PLOT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    train_steps  = history.get("train_step", [])
    train_epochs = history.get("train_epoch", [])
    val_epochs   = history.get("val_epoch", [])

    # --- 1. Step-level loss ---
    if train_steps:
        g_steps    = [s["global_step"] for s in train_steps]
        step_losses = [s["loss"] for s in train_steps]

        plt.figure(figsize=(10, 5))
        plt.plot(g_steps, step_losses, alpha=0.35, color="dodgerblue", label="Raw Step Loss")
        if len(step_losses) >= 5:
            w = max(5, min(50, len(step_losses) // 10))
            smooth = np.convolve(step_losses, np.ones(w) / w, mode="valid")
            plt.plot(g_steps[w - 1:], smooth, color="navy", lw=2, label=f"Smoothed (w={w})")
        plt.title("Step-level Train Loss"); plt.xlabel("Global Step"); plt.ylabel("Loss")
        plt.grid(True, ls="--", alpha=0.6); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "step_train_loss.png"), dpi=150); plt.close()

    # --- 2. Epoch loss ---
    if train_epochs and val_epochs:
        t_eps    = [e["epoch"] for e in train_epochs]
        t_losses = [e["loss"]  for e in train_epochs]
        v_eps    = [e["epoch"] for e in val_epochs]
        v_losses = [e["loss"]  for e in val_epochs]

        plt.figure(figsize=(10, 5))
        plt.plot(t_eps, t_losses, "o-", color="royalblue", lw=2, label="Train")
        plt.plot(v_eps, v_losses, "s--", color="crimson",  lw=2, label="Val")
        plt.title("Epoch Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True, ls="--", alpha=0.6); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epoch_loss.png"), dpi=150); plt.close()

    # --- 3. Epoch mIoU / Dice / Acc ---
    if train_epochs and val_epochs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, key, title, ylabel in [
            (axes[0], "miou",      "Mean IoU",         "mIoU"),
            (axes[1], "pixel_acc", "Pixel Accuracy",   "Accuracy"),
            (axes[2], "dice",      "Dice Coefficient",  "Dice"),
        ]:
            t_vals = [e.get(key, 0) for e in train_epochs]
            v_vals = [e.get(key, 0) for e in val_epochs]
            ax.plot(t_eps, t_vals, "o-",  lw=2, label="Train")
            ax.plot(v_eps, v_vals, "s--", lw=2, label="Val")
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.6); ax.legend()

        plt.suptitle("Epoch-level Metrics", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epoch_metrics.png"), dpi=150); plt.close()

    # --- 4. LR schedule ---
    if train_steps:
        lrs = [s["lr"] for s in train_steps if "lr" in s]
        if lrs:
            plt.figure(figsize=(10, 4))
            plt.plot(g_steps[:len(lrs)], lrs, color="darkviolet", lw=1.8)
            plt.title("Learning Rate Schedule"); plt.xlabel("Global Step"); plt.ylabel("LR")
            plt.grid(True, ls="--", alpha=0.6); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=150); plt.close()

    print(f"--> Plots saved to '{output_dir}/'")


# ==============================================================================
# Sample Predictions Visualization (image, thermal_image, predicted_mask, true_label)
# ==============================================================================
MFNET_COLORMAP = np.array([
    [0,   0,   0],    # 0: Unlabeled (Black)
    [220, 20,  60],   # 1: Car (Crimson Red)
    [0,   200, 0],    # 2: Person (Green)
    [30,  144, 255],  # 3: Bike (Dodger Blue)
    [255, 215, 0],    # 4: Curve (Gold/Yellow)
    [255, 20,  147],  # 5: Car Stop (Deep Pink)
    [0,   255, 255],  # 6: Guardrail (Cyan)
    [255, 140, 0],    # 7: Color Cone (Dark Orange)
    [138, 43,  226],  # 8: Bump (Blue Violet)
], dtype=np.uint8)
IGNORE_COLOR = np.array([128, 128, 128], dtype=np.uint8)  # 255: Ignore (Gray)


def _mask_to_rgb(mask_2d: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Converts a 2D segmentation mask [H, W] into a 3D RGB image [H, W, 3]."""
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(MFNET_COLORMAP))):
        rgb[mask_2d == c] = MFNET_COLORMAP[c]
    rgb[mask_2d == IGNORE_INDEX] = IGNORE_COLOR
    rgb[(mask_2d < 0) | (mask_2d >= num_classes)] = IGNORE_COLOR
    return rgb


def save_epoch_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    device: str = DEVICE,
    output_dir: str = PLOT_DIR,
    num_samples: int = 4,
):
    """
    Saves a comparison image showing 4-8 samples with:
    [image, thermal_image, predicted_mask, true_label] side-by-side.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    num_samples = max(4, min(8, num_samples))
    collected_images = []
    collected_targets = []
    collected_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = raw_model(images)
            preds = torch.argmax(outputs, dim=1)

            bs = images.size(0)
            for i in range(bs):
                collected_images.append(images[i].cpu())
                collected_targets.append(targets[i].cpu())
                collected_preds.append(preds[i].cpu())
                if len(collected_images) >= num_samples:
                    break
            if len(collected_images) >= num_samples:
                break

    n = len(collected_images)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n), squeeze=False)

    for i in range(n):
        img_4ch = collected_images[i].numpy()   # (4, H, W)
        pred_2d = collected_preds[i].numpy()    # (H, W)
        gt_2d   = collected_targets[i].numpy()  # (H, W)

        # 1. Original RGB image (channels 0:3) without mean/std distortion
        rgb = img_4ch[:3].transpose(1, 2, 0)    # (H, W, 3)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)

        # 2. Original Thermal image (channel 3)
        thermal = img_4ch[3]                    # (H, W)
        if thermal.max() > 1.5:
            thermal = thermal / 255.0
        thermal = np.clip(thermal, 0.0, 1.0)

        # 3. Predicted Mask (color-coded)
        pred_rgb = _mask_to_rgb(pred_2d, num_classes=NUM_CLASSES)

        # 4. True Label Mask (color-coded)
        gt_rgb = _mask_to_rgb(gt_2d, num_classes=NUM_CLASSES)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(f"Sample {i + 1}", fontsize=11, fontweight="bold")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        axes[i, 1].imshow(thermal, cmap="gray")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])

        axes[i, 3].imshow(gt_rgb)
        axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])

    axes[0, 0].set_title("image", fontsize=13, fontweight="bold")
    axes[0, 1].set_title("thermal_image", fontsize=13, fontweight="bold")
    axes[0, 2].set_title("predicted_mask", fontsize=13, fontweight="bold")
    axes[0, 3].set_title("true_label", fontsize=13, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=np.array(MFNET_COLORMAP[c]) / 255.0, label=CLASS_NAMES[c])
        for c in range(min(NUM_CLASSES, len(CLASS_NAMES)))
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(9, NUM_CLASSES),
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    plt.suptitle(f"Epoch {epoch:02d} Sample Predictions", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])

    epoch_save_path = os.path.join(output_dir, f"epoch_{epoch:03d}_predictions.png")
    latest_save_path = os.path.join(output_dir, "latest_predictions.png")
    plt.savefig(epoch_save_path, dpi=150, bbox_inches="tight")
    plt.savefig(latest_save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  * Saved sample predictions: '{epoch_save_path}'")


# ==============================================================================
# Training Loop
# ==============================================================================
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epochs: int = EPOCHS,
    start_epoch: int = 1,
    global_step: int = 0,
    history: Optional[Dict[str, list]] = None,
    best_metric_val: float = -float("inf"),
    best_epoch: int = -1,
    device: str = DEVICE,
    checkpoint_dir: str = CHECKPOINT_DIR,
    plot_dir: str = PLOT_DIR,
    vis_samples: int = 4,
) -> Dict[str, list]:

    if history is None:
        history = {"train_step": [], "val_step": [], "train_epoch": [], "val_epoch": []}

    train_metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=torch.device(device))
    val_metrics   = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=torch.device(device))

    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\nStarting training: epochs {start_epoch}→{epochs} | device={device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    for epoch in range(start_epoch, epochs + 1):
        # ------------------------------------------------------------------
        # TRAIN
        # ------------------------------------------------------------------
        model.train()
        train_metrics.reset()
        run_loss_sum, n_samples = 0.0, 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch:02d}/{epochs} [train]",
            smoothing=TQDM_SMOOTHING,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for step, (images, targets) in pbar:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bs      = images.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            step_loss       = loss.item()
            run_loss_sum   += step_loss * bs
            n_samples       += bs
            run_avg          = run_loss_sum / n_samples

            step_m  = train_metrics.update(outputs.detach(), targets)
            run_m   = train_metrics.compute()
            cur_lr  = optimizer.param_groups[0]["lr"]

            if (step + 1) % LOG_EVERY_N_STEPS == 0 or step == len(train_loader) - 1:
                history["train_step"].append({
                    "epoch": epoch, "step": step, "global_step": global_step,
                    "loss": float(step_loss),
                    "miou": float(step_m["miou"]),
                    "pixel_acc": float(step_m["pixel_acc"]),
                    "dice": float(step_m["dice"]),
                    "lr": float(cur_lr),
                })

            pbar.set_postfix(
                loss=f"{step_loss:.3f}(avg {run_avg:.3f})",
                mIoU=f"{step_m['miou']:.3f}(avg {run_m['miou']:.3f})",
                lr=f"{cur_lr:.1e}",
            )
            global_step += 1

        train_ep_loss = run_loss_sum / n_samples if n_samples > 0 else 0.0
        train_ep_m    = train_metrics.compute()
        history["train_epoch"].append({
            "epoch": epoch, "loss": float(train_ep_loss),
            "miou": float(train_ep_m["miou"]),
            "pixel_acc": float(train_ep_m["pixel_acc"]),
            "dice": float(train_ep_m["dice"]),
            "lr": float(cur_lr),
        })

        # ------------------------------------------------------------------
        # VALIDATION
        # ------------------------------------------------------------------
        model.eval()
        val_metrics.reset()
        val_loss_sum, val_n = 0.0, 0

        vbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Epoch {epoch:02d}/{epochs} [val]  ",
            smoothing=TQDM_SMOOTHING,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        with torch.no_grad():
            for step, (images, targets) in vbar:
                images  = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                bs      = images.size(0)

                outputs   = model(images)
                loss      = criterion(outputs, targets)
                step_loss = loss.item()
                val_loss_sum += step_loss * bs
                val_n        += bs
                val_run_avg  = val_loss_sum / val_n

                step_m = val_metrics.update(outputs, targets)
                run_m  = val_metrics.compute()

                if (step + 1) % LOG_EVERY_N_STEPS == 0 or step == len(val_loader) - 1:
                    history["val_step"].append({
                        "epoch": epoch, "step": step, "global_step": global_step,
                        "loss": float(step_loss),
                        "miou": float(step_m["miou"]),
                        "pixel_acc": float(step_m["pixel_acc"]),
                        "dice": float(step_m["dice"]),
                    })

                vbar.set_postfix(
                    loss=f"{step_loss:.3f}(avg {val_run_avg:.3f})",
                    mIoU=f"{step_m['miou']:.3f}(avg {run_m['miou']:.3f})",
                )

        val_ep_loss = val_loss_sum / val_n if val_n > 0 else 0.0
        val_ep_m    = val_metrics.compute()
        history["val_epoch"].append({
            "epoch": epoch, "loss": float(val_ep_loss),
            "miou": float(val_ep_m["miou"]),
            "pixel_acc": float(val_ep_m["pixel_acc"]),
            "dice": float(val_ep_m["dice"]),
        })

        # LR step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_ep_m["miou"])
            else:
                scheduler.step()

        # ------------------------------------------------------------------
        # Epoch summary
        # ------------------------------------------------------------------
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train: loss={train_ep_loss:.4f} mIoU={train_ep_m['miou']:.4f} "
            f"acc={train_ep_m['pixel_acc']:.4f} dice={train_ep_m['dice']:.4f} | "
            f"Val:   loss={val_ep_loss:.4f}  mIoU={val_ep_m['miou']:.4f}  "
            f"acc={val_ep_m['pixel_acc']:.4f}  dice={val_ep_m['dice']:.4f} | "
            f"LR={cur_lr:.2e}"
        )

        # Per-class IoU summary (val)
        iou_cls = val_ep_m.get("iou_per_class", [])
        if iou_cls:
            cls_str = "  Val IoU/class: " + " | ".join(
                f"{CLASS_NAMES[i]}:{v:.3f}"
                for i, v in enumerate(iou_cls)
                if i < len(CLASS_NAMES)
            )
            print(cls_str)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        current_metric = val_ep_m["miou"] if BEST_METRIC_NAME == "miou" else -val_ep_loss
        is_best = current_metric > best_metric_val
        if is_best:
            best_metric_val = current_metric
            best_epoch      = epoch

        ckpt_args = dict(
            epoch=epoch, global_step=global_step,
            model=model, optimizer=optimizer, scheduler=scheduler,
            history=history, best_metric_val=best_metric_val, best_epoch=best_epoch,
        )
        save_checkpoint(os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt"), **ckpt_args)
        save_checkpoint(os.path.join(checkpoint_dir, "last_checkpoint.pt"),         **ckpt_args)
        if is_best:
            save_checkpoint(os.path.join(checkpoint_dir, "best_checkpoint.pt"),    **ckpt_args)
            print(f"  * New best! Val {BEST_METRIC_NAME}={best_metric_val:.4f} at epoch {best_epoch}")

        # ------------------------------------------------------------------
        # Visualizations: Save Sample Predictions (image, thermal_image, predicted_mask, true_label)
        # ------------------------------------------------------------------
        save_epoch_predictions(
            model=model,
            val_loader=val_loader,
            epoch=epoch,
            device=device,
            output_dir=plot_dir,
            num_samples=vis_samples,
        )

    # Final plots
    print("\nGenerating training plots...")
    plot_and_save_training_curves(history, output_dir=plot_dir)

    return history


# ==============================================================================
# Argument Parsing & Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train MFNet RGB+Thermal Segmentation")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS,        help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--arch",       type=str,   default="mfnet",       choices=["mfnet"], help="Architecture (MFNet)")
    parser.add_argument("--data-root",  type=str,   default=DEFAULT_DATA_ROOT,
                        help="Path to MFNet dataset root")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--lr-mode",    type=str,   default="rescale",
                        choices=["rescale", "keep_min", "restart"],
                        help="LR behavior on resume with increased epochs: "
                             "'rescale' (adjust cosine curve to new total epochs), "
                             "'keep_min' (keep LR fixed at minimum 1e-6 for fine-tuning), "
                             "'restart' (fresh cosine decay over remaining epochs)")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--plot-dir",       type=str, default=PLOT_DIR)
    parser.add_argument("--num-workers",    type=int, default=2)
    parser.add_argument("--img-h",          type=int, default=480)
    parser.add_argument("--img-w",          type=int, default=640)
    parser.add_argument("--vis-samples",    type=int, default=4,
                        help="Number of samples (4-8) to visualize after each epoch (image, thermal_image, predicted_mask, true_label)")
    return parser.parse_args()


def main():
    args = parse_args()

    device       = DEVICE
    batch_size   = args.batch_size
    epochs       = args.epochs
    lr           = args.lr
    ckpt_dir     = args.checkpoint_dir
    plot_dir     = args.plot_dir
    img_size     = (args.img_h, args.img_w)

    print("=" * 70)
    print("MFNet RGB+Thermal Semantic Segmentation Training")
    print(f"  Model=MFNet | device={device} | batch={batch_size} | epochs={epochs} | lr={lr}")
    print(f"  img_size={img_size} | num_classes={NUM_CLASSES}")
    print("=" * 70)

    # 1. Build model (MFNet)
    if build_model is None:
        raise RuntimeError("Cannot import build_model from model.py")
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: MFNet | Trainable params: {n_params:,}")

    # 2. Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CombinedLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    # 3. Resume
    start_epoch    = 1
    global_step    = 0
    history        = None
    best_metric    = -float("inf")
    best_epoch_num = -1

    if args.resume:
        start_epoch, global_step, history, best_metric, best_epoch_num = load_checkpoint(
            filepath=args.resume, model=model, optimizer=optimizer,
            scheduler=scheduler, device=device,
        )

        remaining_epochs = max(1, epochs - start_epoch + 1)
        if args.lr_mode == "keep_min":
            print(f"[train.py] LR Mode 'keep_min': keeping LR fixed at minimum (1e-6) for fine-tuning.")
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-6
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=remaining_epochs)
        elif args.lr_mode == "restart":
            restart_lr = args.lr if args.lr != LEARNING_RATE else 1e-4
            print(f"[train.py] LR Mode 'restart': starting fresh Cosine decay from {restart_lr} over remaining {remaining_epochs} epochs.")
            for param_group in optimizer.param_groups:
                param_group["lr"] = restart_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
        else:  # "rescale" (default)
            if scheduler is not None and hasattr(scheduler, "T_max"):
                print(f"[train.py] LR Mode 'rescale': updating scheduler T_max to total epochs={epochs}")
                scheduler.T_max = epochs

    # 4. DataLoaders
    if get_dataloaders is None:
        raise RuntimeError("Cannot import get_dataloaders from dataset.py")

    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
        auto_download=True,
    )

    # 5. Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        start_epoch=start_epoch,
        global_step=global_step,
        history=history,
        best_metric_val=best_metric,
        best_epoch=best_epoch_num,
        device=device,
        checkpoint_dir=ckpt_dir,
        plot_dir=plot_dir,
        vis_samples=args.vis_samples,
    )


if __name__ == "__main__":
    main()
