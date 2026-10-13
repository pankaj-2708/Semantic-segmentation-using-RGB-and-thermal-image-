"""Random visual inspection for RTFNet V2 on MFNet.

Displays (does NOT save) 5 random training and 5 random validation examples,
with RGB, thermal, GT mask, predicted mask, prediction overlay, and error map.
Run from the model repository root:
    python visualize_random_v2.py
Optional:
    python visualize_random_v2.py --data-root ../MFNet_dataset --checkpoint checkpoints/model19_v2_ada16gb/best.pth
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NUM_CLASSES = 9
CLASS_NAMES = [
    "Unlabeled/background", "Car", "Person", "Bike", "Curve",
    "Car stop", "Guardrail", "Color cone", "Bump"
]
# Soft pastel colors chosen for demos/UI. Class 0 uses a light neutral background.
PALETTE = np.array([
    [246, 247, 249],  # 0: light neutral gray
    [137, 196, 225],  # 1: soft blue
    [255, 166, 166],  # 2: light coral/red
    [164, 222, 183],  # 3: mint green
    [255, 207, 153],  # 4: pastel orange
    [199, 181, 230],  # 5: lavender
    [246, 232, 145],  # 6: soft yellow
    [160, 207, 247],  # 7: light sky blue
    [246, 180, 221],  # 8: pastel pink
], dtype=np.uint8)

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Show random MFNet RGB-T predictions without saving images.")
    parser.add_argument("--data-root", default=None, help="MFNet_dataset or MFNet_CMX path; auto-detected by default")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint; defaults to checkpoints/model19_v2_ada16gb/best.pth")
    parser.add_argument("--num-samples", type=int, default=5, help="random examples per split (default: 5)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=None, help="optional seed; omit for new random samples every run")
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA autocast for inference")
    return parser.parse_args()


def first_existing(paths):
    for path in paths:
        if path is not None and Path(path).is_dir():
            if is_supported_root(Path(path)):
                return Path(path).resolve()
    return None


def is_supported_root(root: Path) -> bool:
    prepared = all((root / x).is_dir() for x in ("RGB", "Thermal", "Label"))
    raw = (root / "images").is_dir() and (root / "labels").is_dir()
    return (prepared or raw) and all((root / f"{s}.txt").is_file() for s in ("train", "val", "test"))


def resolve_root(value: Optional[str]) -> Path:
    if value:
        p = Path(value).expanduser().resolve()
        if not is_supported_root(p):
            raise FileNotFoundError(f"Not a recognised MFNet root (expected images/labels or RGB/Thermal/Label and split .txt files): {p}")
        return p
    script = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        script.parent / "MFNet_CMX", script.parent / "MFNet_dataset", script.parent / "ir_seg_dataset",
        script / "MFNet_CMX", script / "MFNet_dataset",
        cwd / "MFNet_CMX", cwd / "MFNet_dataset", cwd / "ir_seg_dataset",
        cwd.parent / "MFNet_CMX", cwd.parent / "MFNet_dataset", cwd.parent / "ir_seg_dataset",
    ]
    root = first_existing(candidates)
    if root is None:
        raise FileNotFoundError("Could not auto-detect MFNet data root. Pass --data-root with the MFNet_dataset or MFNet_CMX folder.")
    return root


def nested_dir(parent: Path, name: str) -> Path:
    """Support both folder/name/*.png and folder/*.png layouts."""
    direct = parent / name
    if direct.is_dir() and any(direct.glob("*.png")):
        return direct
    if parent.is_dir() and any(parent.glob("*.png")):
        return parent
    return direct if direct.is_dir() else parent


def read_split(root: Path, split: str) -> list[str]:
    path = root / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    names = []
    seen = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        token = line.strip()
        if not token:
            continue
        name = Path(token.replace("\\", "/")).stem
        # The original MFNet train list can contain virtual pre-flipped entries.
        if name.endswith("_flip"):
            name = name[:-5]
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def files_for_root(root: Path):
    prepared = all((root / x).is_dir() for x in ("RGB", "Thermal", "Label"))
    if prepared:
        rgb_dir = nested_dir(root, "RGB")
        th_dir = nested_dir(root, "Thermal")
        label_dir = nested_dir(root, "Label")
        mode = "prepared"
    else:
        rgb_dir = nested_dir(root / "images", "images")
        label_dir = nested_dir(root / "labels", "labels")
        th_dir = None  # thermal is fourth channel of each original RGBA image
        mode = "raw-rgba"
    return mode, rgb_dir, th_dir, label_dir


def resolve_names(root: Path, split: str, mode: str, rgb_dir: Path, th_dir: Optional[Path], label_dir: Path) -> list[str]:
    names = read_split(root, split)
    valid = []
    missing = []
    for name in names:
        ok = (rgb_dir / f"{name}.png").is_file() and (label_dir / f"{name}.png").is_file()
        if mode == "prepared":
            ok = ok and th_dir is not None and (th_dir / f"{name}.png").is_file()
        if ok:
            valid.append(name)
        else:
            missing.append(name)
    if not valid:
        raise RuntimeError(f"No valid {split} pairs found. Check split entries and folders under {root}.")
    if missing:
        print(f"Warning: {split} split: skipped {len(missing)} entries with missing paired files; example: {missing[:5]}")
    return valid


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}\nPass --checkpoint to select your best.pth file.")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint {path} does not contain a 'model' state dictionary.")
    return ckpt


def load_sample(root: Path, name: str, mode: str, rgb_dir: Path, th_dir: Optional[Path], label_dir: Path,
                out_hw: tuple[int, int], thermal_mean: float, thermal_std: float):
    rh, rw = out_hw
    rgb_path = rgb_dir / f"{name}.png"
    label_path = label_dir / f"{name}.png"
    with Image.open(rgb_path) as im:
        arr = np.asarray(im)

    if mode == "prepared":
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        th_path = th_dir / f"{name}.png"
        with Image.open(th_path) as im:
            ta = np.asarray(im)
        if ta.ndim == 3:
            ta = ta[..., 0]
        if ta.dtype == np.uint16:
            thermal = (ta.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
        else:
            thermal = ta.astype(np.uint8)
    else:
        if arr.ndim != 3 or arr.shape[2] < 4:
            raise ValueError(f"Raw Kaggle image {rgb_path} has shape {arr.shape}; expected RGBA (RGB + thermal channel).")
        rgb = arr[..., :3].astype(np.uint8)
        thermal = arr[..., 3].astype(np.uint8)

    with Image.open(label_path) as im:
        mask = np.asarray(im)
    if mask.ndim == 3:
        raise ValueError(f"Label {label_path} is not a single-channel mask: {mask.shape}")
    mask = mask.astype(np.uint8)
    ids = set(np.unique(mask).tolist())
    if not ids.issubset(set(range(NUM_CLASSES)) | {255}):
        raise ValueError(f"Unexpected class ids in {label_path.name}: {sorted(ids)}")

    rgb_im = Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR)
    th_im = Image.fromarray(thermal).resize((rw, rh), Image.Resampling.BILINEAR)
    lab_im = Image.fromarray(mask).resize((rw, rh), Image.Resampling.NEAREST)
    rgb_u8 = np.asarray(rgb_im, dtype=np.uint8)
    thermal_u8 = np.asarray(th_im, dtype=np.uint8)
    mask_u8 = np.asarray(lab_im, dtype=np.uint8)

    rgb_f = rgb_u8.astype(np.float32) / 255.0
    rgb_t = torch.from_numpy(np.ascontiguousarray(((rgb_f - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)))
    th_f = thermal_u8.astype(np.float32) / 255.0
    th_t = torch.from_numpy(np.ascontiguousarray(((th_f - thermal_mean) / max(thermal_std, 1e-6))[None]))
    return rgb_t, th_t, mask_u8, rgb_u8, thermal_u8


def colorize(mask: np.ndarray) -> np.ndarray:
    safe = np.asarray(mask)
    out = np.empty((*safe.shape, 3), dtype=np.uint8)
    out[:] = [255, 255, 255]
    for cid in range(NUM_CLASSES):
        out[safe == cid] = PALETTE[cid]
    out[safe == 255] = [55, 65, 81]  # ignore label visibly distinct
    return out


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pastel status map: correct foreground=mint, missed foreground=soft red,
    false-positive foreground=soft blue, foreground class confusion=lavender."""
    out = np.empty((*gt.shape, 3), dtype=np.uint8)
    out[:] = [246, 247, 249]
    valid = gt != 255
    correct_fg = valid & (gt > 0) & (pred == gt)
    missed = valid & (gt > 0) & (pred == 0)
    false_pos = valid & (gt == 0) & (pred > 0)
    confusion = valid & (gt > 0) & (pred > 0) & (pred != gt)
    out[correct_fg] = [159, 222, 181]  # light green
    out[missed] = [255, 160, 166]      # light red
    out[false_pos] = [143, 198, 242]   # light blue
    out[confusion] = [204, 181, 230]   # lavender
    out[gt == 255] = [80, 90, 105]
    return out


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    here = Path(__file__).resolve().parent
    root = resolve_root(args.data_root)
    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else here / "checkpoints" / "model19_v2_ada16gb" / "best.pth"
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    import model19_v2 as model_module
    ckpt = load_checkpoint(ckpt_path, device)
    saved_args = ckpt.get("args", {}) or {}
    model_args = {
        "num_classes": NUM_CLASSES,
        "pretrained": False,
        "head": saved_args.get("head", "conv1x1"),
        "final_relu": saved_args.get("final_relu", False),
        "thermal_init": saved_args.get("thermal_init", "mean"),
        "freeze_encoder_bn": saved_args.get("bn_mode", "train") == "freeze-encoder",
        "device": device,
    }
    model = model_module.build_model(**model_args)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Reuse the exact thermal scaling stats stored by the training pipeline when available.
    thermal_mean = float(ckpt.get("thermal_mean", 0.449))
    thermal_std = float(ckpt.get("thermal_std", 0.226))
    if not np.isfinite(thermal_mean) or not np.isfinite(thermal_std) or thermal_std <= 0:
        thermal_mean, thermal_std = 0.449, 0.226
    img_size = saved_args.get("img_size", [480, 640])
    if not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
        img_size = [480, 640]
    img_size = (int(img_size[0]), int(img_size[1]))

    mode, rgb_dir, th_dir, label_dir = files_for_root(root)
    splits = {split: resolve_names(root, split, mode, rgb_dir, th_dir, label_dir) for split in ("train", "val")}
    print(f"Dataset : {root} ({'prepared RGB/Thermal/Label' if mode == 'prepared' else 'raw RGBA Kaggle layout'})")
    print(f"Model   : {ckpt_path} | checkpoint epoch {ckpt.get('epoch', 'unknown')}")
    print(f"Device  : {device}" + (f" — {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""))
    print(f"Samples : {args.num_samples} random TRAIN + {args.num_samples} random VALIDATION (no files will be saved)")
    print(f"Input   : {img_size[0]}x{img_size[1]} | thermal mean/std = {thermal_mean:.5f}/{thermal_std:.5f}")

    use_amp = device.type == "cuda" and not args.no_amp
    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
    error_patches = [
        Patch(facecolor=np.array([159, 222, 181]) / 255, label="Correct foreground"),
        Patch(facecolor=np.array([255, 160, 166]) / 255, label="Missed foreground (GT > 0, pred = 0)"),
        Patch(facecolor=np.array([143, 198, 242]) / 255, label="False positive (GT = 0, pred > 0)"),
        Patch(facecolor=np.array([204, 181, 230]) / 255, label="Wrong foreground class"),
    ]
    class_patches = [Patch(facecolor=PALETTE[i] / 255.0, label=f"{i}: {CLASS_NAMES[i]}") for i in range(NUM_CLASSES)]

    for split in ("train", "val"):
        chosen = rng.sample(splits[split], k=min(args.num_samples, len(splits[split])))
        fig, axes = plt.subplots(len(chosen), 6, figsize=(20, 3.55 * len(chosen)), squeeze=False)
        fig.patch.set_facecolor("#f7f8fb")
        title_color = "#172033"
        fig.suptitle(f"RTFNet V2 — Random {split.title()} Predictions  |  checkpoint epoch {ckpt.get('epoch', '?')}",
                     fontsize=17, fontweight="bold", color=title_color, y=0.995)
        rows = []
        for row_idx, name in enumerate(chosen):
            rgb_t, thermal_t, gt, rgb_u8, thermal_u8 = load_sample(
                root, name, mode, rgb_dir, th_dir, label_dir, img_size, thermal_mean, thermal_std)
            rgb_batch = rgb_t.unsqueeze(0).to(device)
            th_batch = thermal_t.unsqueeze(0).to(device)
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(rgb_batch, th_batch)
                else:
                    logits = model(rgb_batch, th_batch)
                pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
                conf = logits.float().softmax(1).max(1).values[0].detach().cpu().numpy()
            gt_color, pred_color = colorize(gt), colorize(pred)
            overlay = np.clip(rgb_u8.astype(np.float32) * 0.58 + pred_color.astype(np.float32) * 0.42, 0, 255).astype(np.uint8)
            err = make_error_map(gt, pred)
            panel_data = [
                (rgb_u8, f"RGB\n{name}"),
                (thermal_u8, "Thermal"),
                (gt_color, "Ground truth"),
                (pred_color, "Prediction"),
                (overlay, "Prediction overlay"),
                (err, f"Comparison / errors\nFG correct: {100 * float(((gt > 0) & (pred == gt)).sum() / max((gt > 0).sum(), 1)):.1f}% | conf: {100*float(conf.mean()):.1f}%"),
            ]
            for col, (image, title) in enumerate(panel_data):
                ax = axes[row_idx, col]
                if image.ndim == 2:
                    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(image)
                ax.set_title(title, fontsize=9, color=title_color, pad=5)
                ax.axis("off")
            rows.append((name, float((pred == gt).mean()), float(((gt > 0) & (pred > 0)).sum() / max((gt > 0).sum(), 1))))

        fig.legend(handles=class_patches + error_patches, loc="lower center", ncol=5, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.tight_layout(rect=[0, 0.12, 1, 0.97])
        print(f"{split.upper()} samples:")
        for name, pixel_acc, recall_like in rows:
            print(f"  {name}: pixel agreement={pixel_acc*100:.2f}% | foreground recall-like coverage={recall_like*100:.2f}%")

    print("Close the plot windows to exit. No output images/files are written by this script.")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
