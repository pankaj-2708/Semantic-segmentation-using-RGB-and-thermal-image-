"""
train.py - Semantic Segmentation Training Pipeline
Features:
- Step-level tracking and global averages
- Live tqdm progress bars with running average metrics from accumulated confusion matrix
- Mathematically exact metric aggregation (sample-weighted loss, confusion matrix mIoU/Acc/Dice)
- Comprehensive checkpointing with RNG state, full history, and 3-file config snapshot
- Automated training curve plotting (step loss with MA smoothing, epoch loss, metrics, LR schedule)
"""

import os
import sys
import copy
import random
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import modular components from model and dataset
try:
    import model as model_module
    from model import UNet, build_model
except ImportError:
    model_module = None
    UNet = None
    build_model = None

try:
    import dataset as dataset_module
    from dataset import get_dataloaders
except ImportError:
    dataset_module = None
    get_dataloaders = None

# ==============================================================================
# Globals to add at the top of train.py
# ==============================================================================
LOG_EVERY_N_STEPS = 1        # how often to record step-level values
TQDM_SMOOTHING = 0.1         # smoothing for the live display

# Global Training Parameters / Configuration
NUM_CLASSES = 7
IGNORE_INDEX = 255
BATCH_SIZE = 6
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
PLOT_DIR = "plots"
BEST_METRIC_NAME = "miou"    # metric used to determine 'best' checkpoint ("miou" or "loss")


# ==============================================================================
# Configuration Snapshot Helper (Global parameters of all three files)
# ==============================================================================
def get_config_snapshot() -> Dict[str, Dict[str, Any]]:
    """
    Extracts module-level uppercase global configuration variables from:
    1. train.py (current module)
    2. dataset.py
    3. model.py
    """
    snapshot = {}

    # 1. train.py globals
    snapshot["train"] = {
        k: v for k, v in globals().items()
        if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, bool, list, tuple, dict))
    }

    # 2. dataset.py globals
    if dataset_module is not None:
        snapshot["dataset"] = {
            k: v for k, v in vars(dataset_module).items()
            if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, bool, list, tuple, dict))
        }
    else:
        snapshot["dataset"] = {}

    # 3. model.py globals
    if model_module is not None:
        snapshot["model"] = {
            k: v for k, v in vars(model_module).items()
            if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, bool, list, tuple, dict))
        }
    else:
        snapshot["model"] = {}

    return snapshot


# ==============================================================================
# Metric Aggregation Helper (SegmentationMetrics)
# ==============================================================================
class SegmentationMetrics:
    """
    Accumulates a confusion matrix (NUM_CLASSES x NUM_CLASSES) over the whole epoch
    (on GPU, as integer counts), respecting IGNORE_INDEX.
    Derives:
      - Mean IoU (mIoU) and per-class IoU
      - Pixel Accuracy
      - Mean Dice / F1 and per-class Dice
    Provides:
      - update(preds, targets): updates running matrix and returns single-batch metrics
      - compute(): returns metrics derived from matrix accumulated so far
      - reset(): clears confusion matrix for new epoch
    """
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: Optional[int] = IGNORE_INDEX, device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.conf_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        if self.device is not None:
            self.conf_matrix = self.conf_matrix.to(self.device)

    def reset(self):
        """Reset the epoch-level accumulated confusion matrix."""
        self.conf_matrix.zero_()

    def _compute_batch_confusion_matrix(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute confusion matrix for a single batch on the tensor's device.
        preds: [B, H, W] class indices or [B, C, H, W] logits
        targets: [B, H, W] ground truth class indices
        """
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)

        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Filter out invalid / ignored pixels
        valid_mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        if self.ignore_index is not None:
            valid_mask = valid_mask & (targets_flat != self.ignore_index)

        valid_mask = valid_mask & (preds_flat >= 0) & (preds_flat < self.num_classes)

        targets_valid = targets_flat[valid_mask].long()
        preds_valid = preds_flat[valid_mask].long()

        if targets_valid.numel() == 0:
            return torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=preds.device)

        indices = targets_valid * self.num_classes + preds_valid
        conf = torch.bincount(indices, minlength=self.num_classes * self.num_classes)
        return conf.reshape(self.num_classes, self.num_classes).to(torch.int64)

    @classmethod
    def _metrics_from_conf(cls, conf: torch.Tensor, eps: float = 1e-7) -> Dict[str, Any]:
        """
        Derives IoU per class, mIoU, pixel accuracy, and Dice score from a confusion matrix.
        Rows: Ground Truth classes
        Cols: Predicted classes
        """
        conf_float = conf.float()
        tp = torch.diag(conf_float)
        gt_sum = conf_float.sum(dim=1)     # TP + FN
        pred_sum = conf_float.sum(dim=0)   # TP + FP
        union = gt_sum + pred_sum - tp

        # 1. Pixel Accuracy: sum(TP) / total_pixels
        total_pixels = conf_float.sum()
        pixel_acc = (tp.sum() / (total_pixels + eps)).item() if total_pixels > 0 else 0.0

        # 2. IoU: TP / (Union)
        valid_iou = union > 0
        iou_per_class = torch.zeros(conf.shape[0], device=conf.device, dtype=torch.float32)
        if valid_iou.any():
            iou_per_class[valid_iou] = tp[valid_iou] / (union[valid_iou] + eps)
            miou = iou_per_class[valid_iou].mean().item()
        else:
            miou = 0.0

        # 3. Dice: 2 * TP / (GT + Pred)
        dice_denom = gt_sum + pred_sum
        valid_dice = dice_denom > 0
        dice_per_class = torch.zeros(conf.shape[0], device=conf.device, dtype=torch.float32)
        if valid_dice.any():
            dice_per_class[valid_dice] = (2.0 * tp[valid_dice]) / (dice_denom[valid_dice] + eps)
            dice = dice_per_class[valid_dice].mean().item()
        else:
            dice = 0.0

        return {
            "miou": float(miou),
            "pixel_acc": float(pixel_acc),
            "dice": float(dice),
            "iou_per_class": [float(x) for x in iou_per_class.cpu().tolist()],
            "dice_per_class": [float(x) for x in dice_per_class.cpu().tolist()],
        }

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Accumulates the batch confusion matrix into the epoch confusion matrix,
        and computes metrics for this single batch.
        """
        batch_conf = self._compute_batch_confusion_matrix(preds, targets)
        if self.conf_matrix.device != batch_conf.device:
            self.conf_matrix = self.conf_matrix.to(batch_conf.device)
        self.conf_matrix += batch_conf
        return self._metrics_from_conf(batch_conf)

    def compute(self) -> Dict[str, Any]:
        """
        Derives global/running metrics from the accumulated confusion matrix.
        """
        return self._metrics_from_conf(self.conf_matrix)


# ==============================================================================
# Loss Function (Combined Cross-Entropy + Dice Loss respecting IGNORE_INDEX)
# ==============================================================================
class MulticlassDiceLoss(nn.Module):
    """
    Multiclass Dice Loss handling IGNORE_INDEX.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: Optional[int] = IGNORE_INDEX, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_cloned = targets.clone()

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            targets_cloned[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        # One-hot encode targets: [B, H, W, C] -> [B, C, H, W]
        one_hot = F.one_hot(targets_cloned.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask_expanded = valid_mask.unsqueeze(1).float()
            probs = probs * mask_expanded
            one_hot = one_hot * mask_expanded

        # Compute intersection and union per class over spatial dims
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + one_hot).sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice Loss.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: Optional[int] = IGNORE_INDEX, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        ce_ignore = ignore_index if ignore_index is not None else -100
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore)
        self.dice = MulticlassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ==============================================================================
# Checkpointing Utilities
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
    """
    Saves comprehensive checkpoint including RNG states and 3-file config snapshot.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint_data = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": copy.deepcopy(history),
        "best_metric_val": best_metric_val,
        "best_epoch": best_epoch,
        "config_snapshot": get_config_snapshot(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(checkpoint_data, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = DEVICE,
) -> Tuple[int, int, Dict[str, list], float, int]:
    """
    Restores model, optimizer, scheduler, RNG state, global_step, and full history dict.
    Returns: (start_epoch, global_step, history, best_metric_val, best_epoch)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at '{filepath}'")

    checkpoint = torch.load(filepath, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG states
    if "rng_state" in checkpoint and checkpoint["rng_state"]:
        rng = checkpoint["rng_state"]
        try:
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"])
            if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])
        except Exception as e:
            print(f"[Warning] Could not fully restore RNG states: {e}")

    epoch = checkpoint.get("epoch", 0)
    start_epoch = epoch + 1
    global_step = checkpoint.get("global_step", 0)
    history = checkpoint.get("history", {
        "train_step": [],
        "val_step": [],
        "train_epoch": [],
        "val_epoch": [],
    })
    best_metric_val = checkpoint.get("best_metric_val", -float("inf"))
    best_epoch = checkpoint.get("best_epoch", -1)

    print(f"--> Successfully loaded checkpoint '{filepath}' (Restored Epoch: {epoch}, Global Step: {global_step})")
    return start_epoch, global_step, history, best_metric_val, best_epoch


# ==============================================================================
# Final Plots Generation
# ==============================================================================
def plot_and_save_training_curves(history: Dict[str, list], output_dir: str = PLOT_DIR):
    """
    Generates and saves:
    1. Step-level train loss (with moving average smoothing) vs global_step
    2. Epoch-level train vs validation loss
    3. Epoch-level mIoU, pixel accuracy, and Dice for train vs validation
    4. Learning rate vs global_step
    5. Combined 2x2 training summary figure
    """
    os.makedirs(output_dir, exist_ok=True)

    train_steps = history.get("train_step", [])
    train_epochs = history.get("train_epoch", [])
    val_epochs = history.get("val_epoch", [])

    # 1. Step-level train loss vs global_step
    if train_steps:
        g_steps = [item["global_step"] for item in train_steps]
        step_losses = [item["loss"] for item in train_steps]

        plt.figure(figsize=(10, 5))
        plt.plot(g_steps, step_losses, alpha=0.35, color="dodgerblue", label="Raw Train Step Loss")

        if len(step_losses) >= 5:
            window = max(5, min(50, len(step_losses) // 10))
            smoothed = np.convolve(step_losses, np.ones(window) / window, mode="valid")
            smoothed_steps = g_steps[window - 1:]
            plt.plot(smoothed_steps, smoothed, color="navy", linewidth=2.0, label=f"Smoothed Loss (window={window})")

        plt.title("Step-level Train Loss vs Global Step", fontsize=14, fontweight="bold")
        plt.xlabel("Global Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "step_train_loss.png"), dpi=300)
        plt.close()

    # 2. Epoch-level train vs validation loss
    if train_epochs and val_epochs:
        t_eps = [item["epoch"] for item in train_epochs]
        t_losses = [item["loss"] for item in train_epochs]
        v_eps = [item["epoch"] for item in val_epochs]
        v_losses = [item["loss"] for item in val_epochs]

        plt.figure(figsize=(10, 5))
        plt.plot(t_eps, t_losses, "o-", color="royalblue", linewidth=2.0, label="Train Loss")
        plt.plot(v_eps, v_losses, "s--", color="crimson", linewidth=2.0, label="Val Loss")
        plt.title("Epoch-level Train vs Validation Loss", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epoch_loss.png"), dpi=300)
        plt.close()

    # 3. Epoch-level mIoU, pixel accuracy, and Dice
    if train_epochs and val_epochs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # mIoU
        t_miou = [item["miou"] for item in train_epochs]
        v_miou = [item["miou"] for item in val_epochs]
        axes[0].plot(t_eps, t_miou, "o-", color="teal", linewidth=2.0, label="Train mIoU")
        axes[0].plot(v_eps, v_miou, "s--", color="darkorange", linewidth=2.0, label="Val mIoU")
        axes[0].set_title("Mean IoU (mIoU)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("mIoU", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend(fontsize=10)

        # Pixel Accuracy
        t_acc = [item["pixel_acc"] for item in train_epochs]
        v_acc = [item["pixel_acc"] for item in val_epochs]
        axes[1].plot(t_eps, t_acc, "o-", color="forestgreen", linewidth=2.0, label="Train Pixel Acc")
        axes[1].plot(v_eps, v_acc, "s--", color="purple", linewidth=2.0, label="Val Pixel Acc")
        axes[1].set_title("Pixel Accuracy", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(fontsize=10)

        # Dice / F1
        t_dice = [item.get("dice", 0.0) for item in train_epochs]
        v_dice = [item.get("dice", 0.0) for item in val_epochs]
        axes[2].plot(t_eps, t_dice, "o-", color="steelblue", linewidth=2.0, label="Train Dice")
        axes[2].plot(v_eps, v_dice, "s--", color="firebrick", linewidth=2.0, label="Val Dice")
        axes[2].set_title("Dice Coefficient", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("Dice Score", fontsize=11)
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend(fontsize=10)

        plt.suptitle("Epoch-level Metrics: Train vs Validation", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epoch_metrics.png"), dpi=300)
        plt.close()

    # 4. Learning rate vs global_step
    if train_steps:
        lrs = [item["lr"] for item in train_steps if "lr" in item]
        if lrs:
            plt.figure(figsize=(10, 4))
            plt.plot(g_steps[:len(lrs)], lrs, color="darkviolet", linewidth=1.8)
            plt.title("Learning Rate vs Global Step", fontsize=14, fontweight="bold")
            plt.xlabel("Global Step", fontsize=12)
            plt.ylabel("Learning Rate", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=300)
            plt.close()

    # 5. Combined 2x2 Summary Figure
    if train_steps and train_epochs and val_epochs:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (0,0) Step Loss
        axes[0, 0].plot(g_steps, step_losses, alpha=0.35, color="dodgerblue", label="Raw Train Step Loss")
        if len(step_losses) >= 5:
            window = max(5, min(50, len(step_losses) // 10))
            smoothed = np.convolve(step_losses, np.ones(window) / window, mode="valid")
            smoothed_steps = g_steps[window - 1:]
            axes[0, 0].plot(smoothed_steps, smoothed, color="navy", linewidth=2.0, label=f"Smoothed (w={window})")
        axes[0, 0].set_title("Step-level Train Loss vs Global Step", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Global Step", fontsize=10)
        axes[0, 0].set_ylabel("Loss", fontsize=10)
        axes[0, 0].grid(True, linestyle="--", alpha=0.6)
        axes[0, 0].legend(fontsize=9)

        # (0,1) Epoch Loss
        axes[0, 1].plot(t_eps, t_losses, "o-", color="royalblue", linewidth=1.8, label="Train Loss")
        axes[0, 1].plot(v_eps, v_losses, "s--", color="crimson", linewidth=1.8, label="Val Loss")
        axes[0, 1].set_title("Epoch-level Loss: Train vs Val", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch", fontsize=10)
        axes[0, 1].set_ylabel("Loss", fontsize=10)
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)
        axes[0, 1].legend(fontsize=9)

        # (1,0) Epoch mIoU and Dice
        axes[1, 0].plot(t_eps, t_miou, "o-", color="teal", linewidth=1.8, label="Train mIoU")
        axes[1, 0].plot(v_eps, v_miou, "s--", color="darkorange", linewidth=1.8, label="Val mIoU")
        axes[1, 0].plot(t_eps, t_dice, "^-", color="steelblue", linewidth=1.5, label="Train Dice")
        axes[1, 0].plot(v_eps, v_dice, "v--", color="firebrick", linewidth=1.5, label="Val Dice")
        axes[1, 0].set_title("Epoch-level mIoU & Dice: Train vs Val", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch", fontsize=10)
        axes[1, 0].set_ylabel("Score", fontsize=10)
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)
        axes[1, 0].legend(fontsize=9)

        # (1,1) Learning Rate
        if lrs:
            axes[1, 1].plot(g_steps[:len(lrs)], lrs, color="darkviolet", linewidth=1.8)
        axes[1, 1].set_title("Learning Rate vs Global Step", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Global Step", fontsize=10)
        axes[1, 1].set_ylabel("Learning Rate", fontsize=10)
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)

        plt.suptitle("Training Overview Summary", fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_summary.png"), dpi=300)
        plt.close()

    print(f"--> Saved training plots to '{output_dir}/'")


# ==============================================================================
# Training Pipeline
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
) -> Dict[str, list]:
    """
    Main training and validation loop.
    Implements:
    - Step-level tracking on train and validation
    - Live tqdm progress bar with running average metrics from confusion matrix
    - Sample-weighted loss averaging
    - Epoch summary printing
    - Full checkpointing every epoch
    - Final plotting
    """
    if history is None:
        history = {
            "train_step": [],
            "val_step": [],
            "train_epoch": [],
            "val_epoch": [],
        }

    train_metrics_helper = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=torch.device(device))
    val_metrics_helper = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=torch.device(device))

    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Starting training from epoch {start_epoch} to {epochs} on device: {device}")

    for epoch in range(start_epoch, epochs + 1):
        # ----------------------------------------------------------------------
        # 1. Training Phase
        # ----------------------------------------------------------------------
        model.train()
        train_metrics_helper.reset()
        running_train_loss_sum = 0.0
        total_train_samples = 0

        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{epochs} [train]",
            smoothing=TQDM_SMOOTHING,
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for step, (images, targets) in train_pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Loss tracking: Sample-weighted average
            step_loss = loss.item()
            running_train_loss_sum += step_loss * batch_size
            total_train_samples += batch_size
            running_avg_loss = running_train_loss_sum / total_train_samples

            # Metric tracking: batch metrics & running metrics from accumulated confusion matrix
            step_metrics = train_metrics_helper.update(outputs.detach(), targets)
            running_metrics = train_metrics_helper.compute()

            current_lr = optimizer.param_groups[0]["lr"]

            # Record step-level tracking
            if (step + 1) % LOG_EVERY_N_STEPS == 0 or step == len(train_loader) - 1:
                step_record = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss": float(step_loss),
                    "miou": float(step_metrics["miou"]),
                    "pixel_acc": float(step_metrics["pixel_acc"]),
                    "dice": float(step_metrics["dice"]),
                    "lr": float(current_lr),
                }
                history["train_step"].append(step_record)

            # Live tqdm progress bar update
            train_pbar.set_postfix(
                loss=f"{step_loss:.3f} (avg {running_avg_loss:.3f})",
                mIoU=f"{step_metrics['miou']:.2f} (avg {running_metrics['miou']:.2f})",
                acc=f"{step_metrics['pixel_acc']:.2f} (avg {running_metrics['pixel_acc']:.2f})",
                lr=f"{current_lr:.1e}",
            )

            global_step += 1

        # Global (epoch-level) train metrics
        train_epoch_loss = running_train_loss_sum / total_train_samples if total_train_samples > 0 else 0.0
        train_epoch_metrics = train_metrics_helper.compute()

        history["train_epoch"].append({
            "epoch": epoch,
            "loss": float(train_epoch_loss),
            "miou": float(train_epoch_metrics["miou"]),
            "pixel_acc": float(train_epoch_metrics["pixel_acc"]),
            "dice": float(train_epoch_metrics["dice"]),
            "lr": float(current_lr),
        })

        # ----------------------------------------------------------------------
        # 2. Validation Phase
        # ----------------------------------------------------------------------
        model.eval()
        val_metrics_helper.reset()
        running_val_loss_sum = 0.0
        total_val_samples = 0

        val_pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Epoch {epoch}/{epochs} [val]",
            smoothing=TQDM_SMOOTHING,
            dynamic_ncols=True,
            file=sys.stdout,
        )

        with torch.no_grad():
            for step, (images, targets) in val_pbar:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                batch_size = images.size(0)

                outputs = model(images)
                loss = criterion(outputs, targets)

                step_loss = loss.item()
                running_val_loss_sum += step_loss * batch_size
                total_val_samples += batch_size
                running_avg_loss = running_val_loss_sum / total_val_samples

                step_metrics = val_metrics_helper.update(outputs, targets)
                running_metrics = val_metrics_helper.compute()

                if (step + 1) % LOG_EVERY_N_STEPS == 0 or step == len(val_loader) - 1:
                    val_step_record = {
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": float(step_loss),
                        "miou": float(step_metrics["miou"]),
                        "pixel_acc": float(step_metrics["pixel_acc"]),
                        "dice": float(step_metrics["dice"]),
                    }
                    history["val_step"].append(val_step_record)

                val_pbar.set_postfix(
                    loss=f"{step_loss:.3f} (avg {running_avg_loss:.3f})",
                    mIoU=f"{step_metrics['miou']:.2f} (avg {running_metrics['miou']:.2f})",
                    acc=f"{step_metrics['pixel_acc']:.2f} (avg {running_metrics['pixel_acc']:.2f})",
                )

        val_epoch_loss = running_val_loss_sum / total_val_samples if total_val_samples > 0 else 0.0
        val_epoch_metrics = val_metrics_helper.compute()

        history["val_epoch"].append({
            "epoch": epoch,
            "loss": float(val_epoch_loss),
            "miou": float(val_epoch_metrics["miou"]),
            "pixel_acc": float(val_epoch_metrics["pixel_acc"]),
            "dice": float(val_epoch_metrics["dice"]),
        })

        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_epoch_metrics["miou"])
            else:
                scheduler.step()

        # ----------------------------------------------------------------------
        # 3. Epoch Summary Line
        # ----------------------------------------------------------------------
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_epoch_loss:.4f}, mIoU: {train_epoch_metrics['miou']:.4f}, Acc: {train_epoch_metrics['pixel_acc']:.4f}, Dice: {train_epoch_metrics['dice']:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f}, mIoU: {val_epoch_metrics['miou']:.4f}, Acc: {val_epoch_metrics['pixel_acc']:.4f}, Dice: {val_epoch_metrics['dice']:.4f} | "
            f"LR: {current_lr:.6e}"
        )

        # ----------------------------------------------------------------------
        # 4. Checkpointing (Every Epoch)
        # ----------------------------------------------------------------------
        is_best = False
        current_metric = val_epoch_metrics["miou"] if BEST_METRIC_NAME == "miou" else -val_epoch_loss
        if current_metric > best_metric_val:
            best_metric_val = current_metric
            best_epoch = epoch
            is_best = True

        # Save current epoch checkpoint and last checkpoint
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
        best_ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")

        save_checkpoint(
            filepath=epoch_ckpt_path,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_metric_val=best_metric_val,
            best_epoch=best_epoch,
        )
        save_checkpoint(
            filepath=last_ckpt_path,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_metric_val=best_metric_val,
            best_epoch=best_epoch,
        )

        if is_best:
            save_checkpoint(
                filepath=best_ckpt_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_metric_val=best_metric_val,
                best_epoch=best_epoch,
            )
            print(f"  * New best model saved! (Val {BEST_METRIC_NAME}: {best_metric_val:.4f} at epoch {best_epoch})")

    # --------------------------------------------------------------------------
    # 5. Final Plots (After Training)
    # --------------------------------------------------------------------------
    print("\nGenerating final training curves and diagnostic plots...")
    plot_and_save_training_curves(history, output_dir=plot_dir)

    return history


# ==============================================================================
# Main Entry Point & Argument Parsing
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--data-root", type=str, default=None, help="Root path to LoveDA or custom dataset")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for verification and dry runs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="Directory to save checkpoints")
    parser.add_argument("--plot-dir", type=str, default=PLOT_DIR, help="Directory to save plots")
    return parser.parse_args()


def main():
    args = parse_args()

    device = DEVICE
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    checkpoint_dir = args.checkpoint_dir
    plot_dir = args.plot_dir

    print("=" * 70)
    print("Semantic Segmentation Training Pipeline")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {epochs} | LR: {lr}")
    print(f"Checkpoints: {checkpoint_dir} | Plots: {plot_dir}")
    print("=" * 70)

    # 1. Build Model
    if build_model is not None:
        model = build_model(in_channels=3, num_classes=NUM_CLASSES)
    else:
        raise RuntimeError("Could not import model architecture from model.py")

    model = model.to(device)

    # 2. Setup Optimizer & Scheduler & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CombinedLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    # 3. Load / Resume Checkpoint if requested
    start_epoch = 1
    global_step = 0
    history = None
    best_metric_val = -float("inf")
    best_epoch = -1

    if args.resume:
        start_epoch, global_step, history, best_metric_val, best_epoch = load_checkpoint(
            filepath=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    # 4. DataLoaders
    if get_dataloaders is not None:
        use_synth = args.synthetic or (args.data_root is None or not os.path.exists(args.data_root))
        train_loader, val_loader = get_dataloaders(
            data_root=args.data_root,
            batch_size=batch_size,
            use_synthetic=use_synth,
            synthetic_samples=48 if use_synth else None,
        )
        if use_synth:
            print("[Note] Using synthetic dataset for demonstration/verification.")
    else:
        raise RuntimeError("Could not import get_dataloaders from dataset.py")

    # 5. Execute Training
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
        best_metric_val=best_metric_val,
        best_epoch=best_epoch,
        device=device,
        checkpoint_dir=checkpoint_dir,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
