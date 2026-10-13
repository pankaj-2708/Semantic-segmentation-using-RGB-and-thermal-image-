"""
dataset19_v2.py -- RGB-Thermal MFNet loader, validation and LABEL DIAGNOSTICS (V2).

Standalone: does not import the teammate's dataset.py or dataset19.py. Never writes to or
re-downloads the dataset. Supports both common MFNet layouts:

1) Kaggle/original: ../MFNet_dataset/images/<id>.png (RGBA: RGB + thermal in channel 4),
   ../MFNet_dataset/labels/<id>.png, train.txt, val.txt, test.txt.
2) Prepared: ../MFNet_CMX/RGB/<id>.png, Thermal/<id>.png, Label/<id>.png.

Nested archive folders such as images/images and labels/labels are also detected. The default
searches for ../MFNet_dataset first, then ../MFNet_CMX, so teammates downloading the Kaggle
version can run the same scripts without converting the data first.

Diagnostic command (run FIRST, it changes nothing):
    python dataset19_v2.py --diagnose
        -> prints per-mask class ids + pixel counts, per-split class shares, masks that contain
           only class 0, validates ids / sizes / alignment, and saves colour previews, a
           class-share bar chart and per-mask CSV files to plots/model19_v2/diagnostics/
           (existing files are never overwritten: a timestamp is appended instead).

Class-id convention (official MFNet / RTFNet ``util.get_palette``) -- fixed colour map
--------------------------------------------------------------------------------------
    0 unlabeled/background  black        1 car        (64,0,128)    2 person    (64,64,0)
    3 bike (0,128,192)  4 curve (0,0,192)  5 car stop (128,128,0)  6 guardrail (64,64,128)
    7 color cone (192,128,128)  8 bump (192,64,0)
Label files store the raw ids 0..8, so they look almost black in a normal image viewer; that
alone does NOT mean they are empty.  Class 0 is "unlabeled": the official RTFNet loss
(plain cross-entropy, no ignore_index) and its confusion-matrix evaluation both include it.

Normalisation (documented)
--------------------------
* RGB: x/255, then ImageNet mean/std (encoders are ImageNet-initialised).
* Thermal: x/255, then (x - mean)/std where, by default (``thermal_norm="dataset"``), mean/std
  are measured on the TRAIN split only (computed once, cached outside the dataset folder) so
  the thermal input is roughly zero-mean/unit-variance.  ``"imagenet_gray"`` uses the fixed
  grey equivalent of the ImageNet statistics (0.449/0.226) as in V1.  The official RTFNet
  loader only divides by 255.

Augmentation (train only, ONE set of random parameters applied to RGB, thermal and label)
-----------------------------------------------------------------------------------------
random horizontal flip (p=0.5) and random crop keeping >= (1-crop_rate) of each side
(official RTFNet RandomCrop, crop_rate 0.1), then resize back.  Images: bilinear.  Label:
NEAREST, so ids stay discrete.  val/test: no augmentation.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NUM_CLASSES = 9
IGNORE_INDEX = 255
CLASS_NAMES = ["Unlabeled/Background", "Car", "Person", "Bike", "Curve",
               "Car Stop", "Guardrail", "Color Cone", "Bump"]
PALETTE = np.array([                      # official RTFNet get_palette()
    [0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192],
    [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0]], dtype=np.uint8)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAY_MEAN = float(np.mean(IMAGENET_MEAN))   # 0.449
GRAY_STD = float(np.mean(IMAGENET_STD))     # 0.226
NATIVE_HW = (480, 640)
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 784, "val": 392, "test": 393}
THERMAL_NORMS = ("dataset", "imagenet_gray")

_RES = getattr(Image, "Resampling", Image)
_BILINEAR, _NEAREST = _RES.BILINEAR, _RES.NEAREST
_FLIP_LR = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT


class DatasetValidationError(RuntimeError):
    """Raised when the prepared dataset is incomplete or inconsistent."""


# --------------------------------------------------------------------------- #
# Paths and split files
# --------------------------------------------------------------------------- #
def _has_pngs(folder: Path) -> bool:
    return folder.is_dir() and any(folder.glob("*.png"))


def _resolve_flat_or_nested(root: Path, name: str) -> Path:
    """Resolve name/ or common nested name/name/ layout created by some ZIP extracts."""
    direct = root / name
    nested = direct / name
    if _has_pngs(direct):
        return direct
    if _has_pngs(nested):
        return nested
    # Keep an existing nested directory as the likely location even if it is empty, so
    # validation can report the missing files at the actual directory.
    if nested.is_dir() and direct.is_dir():
        return nested
    return direct


def _looks_like_dataset_root(root: Path) -> bool:
    if not root.is_dir():
        return False
    prepared = all((root / d).is_dir() for d in ("RGB", "Thermal", "Label"))
    raw = (root / "images").is_dir() and (root / "labels").is_dir()
    return prepared or raw


def dataset_layout(root: str | Path) -> Dict[str, object]:
    """Return resolved modality directories and layout kind for prepared or raw MFNet."""
    root = Path(root).expanduser().resolve()
    if all((root / d).is_dir() for d in ("RGB", "Thermal", "Label")):
        return {
            "kind": "prepared",
            "rgb_dir": _resolve_flat_or_nested(root, "RGB"),
            "thermal_dir": _resolve_flat_or_nested(root, "Thermal"),
            "label_dir": _resolve_flat_or_nested(root, "Label"),
        }
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return {
            "kind": "kaggle",
            "rgb_dir": _resolve_flat_or_nested(root, "images"),
            "thermal_dir": None,  # thermal is stored in RGBA channel 4 of images/<id>.png
            "label_dir": _resolve_flat_or_nested(root, "labels"),
        }
    raise DatasetValidationError(
        f"Cannot recognize MFNet layout at {root}. Expected either RGB/Thermal/Label or "
        "images/labels folders with train.txt, val.txt and test.txt."
    )


def default_data_root() -> Path:
    """Prefer the original Kaggle MFNet_dataset, then the prepared MFNet_CMX layout."""
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    bases = [here.parent, here, cwd.parent, cwd]
    names = ("MFNet_dataset", "MFNet_CMX", "ir_seg_dataset")
    seen = set()
    candidates = []
    for base in bases:
        for name in names:
            candidate = (base / name).resolve()
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
                # Some Kaggle ZIPs add one extra root folder.
                candidates.extend([(candidate / "MFNet_dataset").resolve(), (candidate / "MFNet_CMX").resolve()])
    for candidate in candidates:
        if _looks_like_dataset_root(candidate):
            return candidate
    # Keep a predictable fallback so the validation message points to the expected path.
    return (here.parent / "MFNet_dataset").resolve()


def resolve_data_root(data_root: Optional[str | Path] = None) -> Path:
    if data_root is None:
        return default_data_root()
    root = Path(data_root).expanduser().resolve()
    if _looks_like_dataset_root(root):
        return root
    # If a user passes the project parent instead of the dataset root, find a conventional child.
    for name in ("MFNet_dataset", "MFNet_CMX", "ir_seg_dataset"):
        candidate = root / name
        if _looks_like_dataset_root(candidate):
            return candidate.resolve()
    return root


def read_split_file(root: Path, split: str) -> Tuple[List[str], Dict[str, object]]:
    path = Path(root) / f"{split}.txt"
    if not path.is_file():
        raise DatasetValidationError(f"Split file not found: {path}")
    names, seen, dups, n, virtual_flips = [], set(), [], 0, 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n += 1
            if s.lower().endswith(".png"):
                s = s[:-4]
            # MFNet train.txt often represents flipped copies with virtual *_flip entries.
            # Remove these entries and perform synchronized random flipping in the loader instead.
            if split == "train" and s.endswith("_flip"):
                s = s[:-5]
                virtual_flips += 1
            if s in seen:
                dups.append(s)
                continue
            seen.add(s)
            names.append(s)
    return names, {"path": str(path), "lines": n, "duplicates": dups,
                   "virtual_flips_normalized": virtual_flips}

def unique_path(path: Path) -> Path:
    """Never overwrite an existing output: append a timestamp instead."""
    if not path.exists():
        return path
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


# --------------------------------------------------------------------------- #
# Image readers
# --------------------------------------------------------------------------- #
def read_thermal_array(path: Path) -> np.ndarray:
    """Thermal PNG -> float32 [H,W] on a 0..255 scale (handles L, replicated RGB, 16-bit)."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 257.0
    return arr.astype(np.float32)


def read_label_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"label {path.name} must be single-channel, got shape {arr.shape}")
    return arr.astype(np.uint8)


def label_histogram(path: Path) -> np.ndarray:
    return np.bincount(read_label_array(path).ravel(), minlength=256).astype(np.int64)[:256]


def label_path(root: Path, name: str) -> Path:
    return Path(dataset_layout(root)["label_dir"]) / f"{name}.png"


def image_path(root: Path, name: str) -> Path:
    return Path(dataset_layout(root)["rgb_dir"]) / f"{name}.png"


def thermal_array_for_sample(root: Path, name: str) -> np.ndarray:
    """Read thermal pixels from a separate Thermal PNG or the 4th channel of raw RGBA MFNet."""
    layout = dataset_layout(root)
    if layout["kind"] == "prepared":
        return read_thermal_array(Path(layout["thermal_dir"]) / f"{name}.png")
    path = Path(layout["rgb_dir"]) / f"{name}.png"
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise DatasetValidationError(
            f"Raw Kaggle image {path} has shape {arr.shape}; expected RGBA where channel 4 stores thermal data. "
            "If your download has separate thermal files, use a prepared RGB/Thermal/Label layout."
        )
    thermal = arr[..., 3]
    if thermal.dtype == np.uint16:
        return thermal.astype(np.float32) / 257.0
    return thermal.astype(np.float32)


def compute_label_counts(root: Path, names: Sequence[str]) -> np.ndarray:
    """Total pixel count per label id (int64[256]) over the given sample names."""
    hist = np.zeros(256, dtype=np.int64)
    for n in names:
        hist += label_histogram(label_path(root, n))
    return hist


def compute_thermal_stats(root: Path, names: Sequence[str]) -> Tuple[float, float]:
    """Mean/std of thermal intensity scaled to 0..1 over the given samples."""
    s = s2 = 0.0
    n = 0
    for name in names:
        a = thermal_array_for_sample(root, name).astype(np.float64) / 255.0
        s += a.sum(); s2 += (a ** 2).sum(); n += a.size
    if n == 0:
        raise DatasetValidationError("Cannot calculate thermal statistics: no training pixels found")
    mean = s / n
    return float(mean), float(np.sqrt(max(s2 / n - mean ** 2, 0.0)))


def thermal_stats_for_train(root: Path, cache_path: Optional[Path] = None,
                            verbose: bool = True) -> Tuple[float, float]:
    """Train-split thermal mean/std, cached in ``cache_path`` (never inside the dataset folder)."""
    names = read_split_file(root, "train")[0]
    key = {"root": str(Path(root).resolve()), "n_train": len(names)}
    if cache_path is not None and Path(cache_path).is_file():
        try:
            c = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if c.get("key") == key:
                return float(c["mean"]), float(c["std"])
        except Exception:  # noqa: BLE001
            pass
    if verbose:
        print(f"Computing thermal mean/std over {len(names)} train images ...", flush=True)
    mean, std = compute_thermal_stats(root, names)
    if std < 1e-6:
        raise DatasetValidationError(f"thermal std is ~0 ({std}); thermal images look constant/invalid")
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({"key": key, "mean": mean, "std": std}), encoding="utf-8")
    return mean, std


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    root: Path
    source_layout: str = ""
    counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    folder_counts: Dict[str, int] = field(default_factory=dict)
    image_sizes: Counter = field(default_factory=Counter)
    label_pixels: Dict[str, np.ndarray] = field(default_factory=dict)
    unexpected_label_ids: Dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            shown = "\n  - ".join(self.errors[:20])
            more = f"\n  ... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
            raise DatasetValidationError(f"Dataset validation FAILED for {self.root}:\n  - {shown}{more}")

    def print_summary(self) -> None:
        print("=" * 72)
        print(f"Dataset root : {self.root}")
        if self.source_layout:
            print(f"Source layout: {self.source_layout}")
        for d, n in self.folder_counts.items():
            print(f"  {d + '/':<9} {n} PNG files")
        for s in SPLITS:
            if s in self.counts:
                exp = EXPECTED_COUNTS.get(s)
                note = "" if exp is None or exp == self.counts[s] else f"  (differs from expected {exp})"
                print(f"  {s:<6} split: {self.counts[s]} samples{note}")
        if self.image_sizes:
            print("  image sizes (WxH): " + ", ".join(f"{w}x{h} x{n}" for (w, h), n in self.image_sizes.most_common(4)))
        if self.label_pixels:
            for s, hist in self.label_pixels.items():
                tot = hist.sum()
                if tot:
                    print(f"  label share [{s:<5}]: " + ", ".join(
                        f"{i}:{100 * hist[i] / tot:.2f}%" for i in range(NUM_CLASSES)))
        for w in self.warnings:
            print(f"  WARNING: {w}")
        for e in self.errors:
            print(f"  ERROR  : {e}")
        print("  RESULT: " + ("OK" if self.ok else "FAILED"))
        print("=" * 72)


def _stems(folder: Path) -> set:
    return {p.stem for p in folder.glob("*.png")} if folder.is_dir() else set()


def validate_dataset(root: str | Path, splits: Sequence[str] = SPLITS, scan_labels: bool = True,
                     deep: bool = False, verbose: bool = True) -> ValidationReport:
    """Validate either raw Kaggle MFNet (RGBA images + labels) or prepared RGB/Thermal/Label data."""
    root = Path(root).expanduser().resolve()
    rep = ValidationReport(root=root)

    def finish():
        if verbose:
            rep.print_summary()
        return rep

    if not root.is_dir():
        rep.errors.append(f"data root does not exist: {root}")
        return finish()
    try:
        layout = dataset_layout(root)
    except DatasetValidationError as exc:
        rep.errors.append(str(exc))
        return finish()

    raw_layout = layout["kind"] == "kaggle"
    if raw_layout:
        rep.source_layout = "Kaggle/original MFNet (images/*.png RGBA: RGB + thermal channel 4; labels/*.png)"
        dirs = {"RGB": Path(layout["rgb_dir"]), "Label": Path(layout["label_dir"])}
        image_stems = _stems(dirs["RGB"])
        label_stems = _stems(dirs["Label"])
        stems = {"RGB": image_stems, "Thermal": image_stems, "Label": label_stems}
        rep.folder_counts = {"images (RGB+thermal)": len(image_stems), "labels": len(label_stems)}
        no_img = sorted(label_stems - image_stems)
        if no_img:
            rep.errors.append(f"{len(no_img)} labels have no matching source image, e.g. {no_img[:5]}")
        unlabeled_imgs = sorted(image_stems - label_stems)
        if unlabeled_imgs:
            rep.warnings.append(f"{len(unlabeled_imgs)} images have no label and will not be used as supervised samples; e.g. {unlabeled_imgs[:5]}")
    else:
        rep.source_layout = "Prepared MFNet (RGB/*.png + Thermal/*.png + Label/*.png)"
        dirs = {"RGB": Path(layout["rgb_dir"]), "Thermal": Path(layout["thermal_dir"]),
                "Label": Path(layout["label_dir"])}
        stems = {key: _stems(value) for key, value in dirs.items()}
        rep.folder_counts = {key: len(value) for key, value in stems.items()}
        for key, folder in dirs.items():
            if not folder.is_dir():
                rep.errors.append(f"missing folder: {folder}")
            if not stems[key]:
                rep.errors.append(f"no PNG files found in {key} directory: {folder}")
        for a, b in (("RGB", "Thermal"), ("RGB", "Label"), ("Thermal", "Label")):
            oa, ob = sorted(stems[a] - stems[b]), sorted(stems[b] - stems[a])
            if oa:
                rep.errors.append(f"{len(oa)} files in {a}/ without a {b}/ partner, e.g. {oa[:3]}")
            if ob:
                rep.errors.append(f"{len(ob)} files in {b}/ without an {a}/ partner, e.g. {ob[:3]}")

    # All train/val/test split names must have all required data. The raw Kaggle thermal
    # channel comes from the same RGBA image as RGB and thus uses the same stem set.
    split_names: Dict[str, List[str]] = {}
    for split in splits:
        try:
            names, info = read_split_file(root, split)
        except DatasetValidationError as exc:
            rep.errors.append(str(exc))
            continue
        split_names[split] = names
        rep.counts[split] = len(names)
        if info["duplicates"]:
            rep.warnings.append(f"{split}.txt had {len(info['duplicates'])} duplicate/virtual-flip entries; normalized to {len(names)} unique ids")
        if not names:
            rep.errors.append(f"{split}.txt is empty")
        for key in ("RGB", "Thermal", "Label"):
            missing = [name for name in names if name not in stems[key]]
            if missing:
                rep.errors.append(f"{split}: {len(missing)} entries have no {key} data, e.g. {missing[:3]}")

    done = list(split_names)
    for i in range(len(done)):
        for j in range(i + 1, len(done)):
            inter = set(split_names[done[i]]) & set(split_names[done[j]])
            if inter:
                rep.errors.append(f"split overlap {done[i]}/{done[j]}: {len(inter)} shared samples")
    if rep.errors:
        return finish()

    for split, names in split_names.items():
        hist = np.zeros(256, dtype=np.int64)
        for name in names:
            try:
                rgb_path = dirs["RGB"] / f"{name}.png"
                label_file = dirs["Label"] / f"{name}.png"
                with Image.open(rgb_path) as im:
                    sz_rgb, img_mode = im.size, im.mode
                    if raw_layout:
                        if img_mode != "RGBA":
                            arr = np.asarray(im)
                            if arr.ndim != 3 or arr.shape[2] != 4:
                                rep.errors.append(f"{split}/{name}: raw image mode {img_mode!r} has no 4th thermal channel")
                        elif deep:
                            arr = np.asarray(im)
                            if arr.ndim != 3 or arr.shape[2] != 4:
                                rep.errors.append(f"{split}/{name}: expected RGBA image with 4 channels, got {arr.shape}")
                if raw_layout:
                    sz_th = sz_rgb
                else:
                    with Image.open(dirs["Thermal"] / f"{name}.png") as im:
                        sz_th = im.size
                with Image.open(label_file) as im:
                    sz_lab, mode = im.size, im.mode
            except Exception as exc:  # noqa: BLE001
                rep.errors.append(f"{split}/{name}: cannot open image ({type(exc).__name__}: {exc})")
                continue
            rep.image_sizes[sz_rgb] += 1
            if not (sz_rgb == sz_th == sz_lab):
                rep.errors.append(f"{split}/{name}: size mismatch RGB{sz_rgb} Thermal{sz_th} Label{sz_lab}")
            if mode not in ("L", "P", "I;16", "I"):
                rep.errors.append(f"{split}/{name}: label mode {mode!r} (expected single-channel L/P)")
                continue
            if deep:
                try:
                    if raw_layout:
                        with Image.open(rgb_path) as im:
                            arr = np.asarray(im)
                        if arr.ndim != 3 or arr.shape[2] != 4:
                            rep.errors.append(f"{split}/{name}: expected raw RGBA RGB+thermal file, got {arr.shape}")
                    else:
                        with Image.open(dirs["RGB"] / f"{name}.png") as im:
                            np.asarray(im.convert("RGB"))
                        with Image.open(dirs["Thermal"] / f"{name}.png") as im:
                            np.asarray(im)
                except Exception as exc:  # noqa: BLE001
                    rep.errors.append(f"{split}/{name}: corrupt image ({type(exc).__name__}: {exc})")
            if scan_labels:
                try:
                    hist += label_histogram(label_file)
                except Exception as exc:  # noqa: BLE001
                    rep.errors.append(f"{split}/{name}: cannot decode label ({type(exc).__name__}: {exc})")
        if scan_labels:
            rep.label_pixels[split] = hist
    if len(rep.image_sizes) > 1:
        rep.warnings.append(f"more than one image size present: {dict(rep.image_sizes)}")
    if scan_labels and rep.label_pixels:
        total = sum(rep.label_pixels.values())
        for cid in range(NUM_CLASSES, 256):
            if cid != IGNORE_INDEX and total[cid] > 0:
                rep.unexpected_label_ids[cid] = int(total[cid])
        if rep.unexpected_label_ids:
            rep.errors.append(f"unexpected label ids (not 0..{NUM_CLASSES - 1} or {IGNORE_INDEX}): {rep.unexpected_label_ids}")
        if total[IGNORE_INDEX] > 0:
            rep.warnings.append(f"label id {IGNORE_INDEX} present in {int(total[IGNORE_INDEX])} pixels (treated as ignore_index)")
    for split in splits:
        exp = EXPECTED_COUNTS.get(split)
        if split in rep.counts and exp is not None and rep.counts[split] != exp:
            rep.warnings.append(f"{split} has {rep.counts[split]} samples; expected {exp} after virtual-flip normalization")
    return finish()


# --------------------------------------------------------------------------- #
# Synchronised augmentation (one parameter set for all three images)
# --------------------------------------------------------------------------- #
def sample_aug_params(w: int, h: int, flip_prob: float, crop_rate: float,
                      crop_prob: float = 1.0) -> Dict[str, object]:
    flip = random.random() < flip_prob
    box = None
    if crop_rate > 0 and random.random() < crop_prob:
        mx, my = int(w * crop_rate), int(h * crop_rate)
        box = (random.randint(0, mx), random.randint(0, my),
               random.randint(w - mx, w), random.randint(h - my, h))
    return {"flip": flip, "box": box}


def transform_triplet(rgb: Image.Image, thermal: Image.Image, label: Image.Image,
                      out_hw: Tuple[int, int], params: Optional[Dict[str, object]] = None):
    """Apply identical geometry to RGB, thermal ('F' mode) and label; label uses NEAREST."""
    if params:
        if params["flip"]:
            rgb, thermal, label = (rgb.transpose(_FLIP_LR), thermal.transpose(_FLIP_LR),
                                   label.transpose(_FLIP_LR))
        if params["box"] is not None:
            b = params["box"]
            rgb, thermal, label = rgb.crop(b), thermal.crop(b), label.crop(b)
    oh, ow = out_hw
    if rgb.size != (ow, oh):
        rgb = rgb.resize((ow, oh), _BILINEAR)
        thermal = thermal.resize((ow, oh), _BILINEAR)
        label = label.resize((ow, oh), _NEAREST)
    return rgb, thermal, label


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RGBTDataset(Dataset):
    """Returns ``(rgb [3,H,W] f32, thermal [1,H,W] f32, mask [H,W] int64)``; ``sample_name(i)`` gives the id."""

    def __init__(self, data_root: str | Path, split: str, img_size: Tuple[int, int] = NATIVE_HW,
                 augment: bool = False, flip_prob: float = 0.5, crop_rate: float = 0.1,
                 crop_prob: float = 1.0, names: Optional[Sequence[str]] = None,
                 thermal_mean: float = GRAY_MEAN, thermal_std: float = GRAY_STD):
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        h, w = int(img_size[0]), int(img_size[1])
        if h < 32 or w < 32:
            raise ValueError(f"img_size must be at least 32x32, got {(h, w)}")
        if not 0.0 <= crop_rate < 0.5:
            raise ValueError("crop_rate must be in [0, 0.5)")
        if thermal_std <= 0:
            raise ValueError("thermal_std must be > 0")
        self.root, self.split, self.img_size = Path(data_root).expanduser().resolve(), split, (h, w)
        self.layout = dataset_layout(self.root)
        self.augment, self.flip_prob = bool(augment), float(flip_prob)
        self.crop_rate, self.crop_prob = float(crop_rate), float(crop_prob)
        self.thermal_mean, self.thermal_std = float(thermal_mean), float(thermal_std)
        self.names = list(names) if names is not None else read_split_file(self.root, split)[0]
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.names)

    def sample_name(self, index: int) -> str:
        return self.names[index]

    def load_raw(self, name: str):
        try:
            rgb_file = Path(self.layout["rgb_dir"]) / f"{name}.png"
            label_file = Path(self.layout["label_dir"]) / f"{name}.png"
            if self.layout["kind"] == "prepared":
                with Image.open(rgb_file) as im:
                    rgb = im.convert("RGB")
                thermal_arr = read_thermal_array(Path(self.layout["thermal_dir"]) / f"{name}.png")
                th = Image.fromarray(thermal_arr)
            else:
                # Original Kaggle MFNet stores RGB and thermal together in the RGBA PNG.
                with Image.open(rgb_file) as im:
                    raw = np.asarray(im)
                if raw.ndim != 3 or raw.shape[2] != 4:
                    raise ValueError(f"expected raw RGBA RGB+thermal image, got shape {raw.shape} mode from {rgb_file}")
                rgb = Image.fromarray(raw[..., :3].astype(np.uint8))
                th = Image.fromarray(raw[..., 3].astype(np.uint8))
            lab = Image.fromarray(read_label_array(label_file))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to load sample {name!r} from {self.root}: {exc}") from exc
        if not (rgb.size == th.size == lab.size):
            raise RuntimeError(f"sample {name!r}: size mismatch RGB{rgb.size} Thermal{th.size} Label{lab.size}")
        return rgb, th, lab

    def __getitem__(self, index: int):
        name = self.names[index]
        rgb, th, lab = self.load_raw(name)
        params = sample_aug_params(*rgb.size, self.flip_prob, self.crop_rate, self.crop_prob) \
            if self.augment else None
        rgb, th, lab = transform_triplet(rgb, th, lab, self.img_size, params)
        rgb_np = (np.asarray(rgb, dtype=np.float32) / 255.0 - self._mean) / self._std
        th_np = (np.asarray(th, dtype=np.float32) / 255.0 - self.thermal_mean) / self.thermal_std
        return (torch.from_numpy(np.ascontiguousarray(rgb_np.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(th_np[None])),
                torch.from_numpy(np.asarray(lab, dtype=np.int64).copy()))


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(data_root: Optional[str | Path] = None, img_size: Tuple[int, int] = NATIVE_HW,
                      batch_size: int = 2, eval_batch_size: Optional[int] = None, num_workers: int = 0,
                      seed: int = 42, augment_train: bool = True, flip_prob: float = 0.5,
                      crop_rate: float = 0.1, splits: Sequence[str] = SPLITS, validate: bool = True,
                      scan_labels: bool = True, deep: bool = False, pin_memory: Optional[bool] = None,
                      thermal_norm: str = "dataset", thermal_cache: Optional[Path] = None,
                      thermal_override: Optional[Tuple[float, float]] = None, verbose: bool = True):
    """Returns ``(datasets, loaders, info)``; info has the report and the thermal stats used."""
    if thermal_norm not in THERMAL_NORMS:
        raise ValueError(f"thermal_norm must be one of {THERMAL_NORMS}")
    root = resolve_data_root(data_root)
    report = validate_dataset(root, splits, scan_labels=scan_labels, deep=deep, verbose=verbose) if validate else None
    if report is not None:
        report.raise_if_failed()
    if thermal_override is not None:             # e.g. the exact values stored in a checkpoint
        t_mean, t_std = float(thermal_override[0]), float(thermal_override[1])
    elif thermal_norm == "dataset":
        t_mean, t_std = thermal_stats_for_train(root, thermal_cache, verbose)
    else:
        t_mean, t_std = GRAY_MEAN, GRAY_STD
    datasets = {s: RGBTDataset(root, s, img_size=img_size, augment=(augment_train and s == "train"),
                               flip_prob=flip_prob, crop_rate=crop_rate,
                               thermal_mean=t_mean, thermal_std=t_std) for s in splits}
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    eval_bs = eval_batch_size or batch_size
    loaders = {}
    for s, ds in datasets.items():
        is_train = s == "train"
        g = torch.Generator()
        g.manual_seed(seed)
        loaders[s] = DataLoader(ds, batch_size=batch_size if is_train else eval_bs, shuffle=is_train,
                                num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
                                persistent_workers=num_workers > 0,
                                worker_init_fn=_worker_init_fn if num_workers else None,
                                generator=g if is_train else None)
    if verbose:
        print(f"Input size HxW {img_size[0]}x{img_size[1]} | batch {batch_size} (eval {eval_bs}) | "
              f"workers {num_workers} | augment_train={augment_train}")
        print(f"Thermal normalisation: {thermal_norm} (mean {t_mean:.4f}, std {t_std:.4f}, on 0..1 scale)")
        print("Samples: " + ", ".join(f"{s}={len(d)}" for s, d in datasets.items()))
    info = {"report": report, "thermal_norm": thermal_norm, "thermal_mean": t_mean, "thermal_std": t_std,
            "root": root}
    return datasets, loaders, info


# --------------------------------------------------------------------------- #
# Visual helpers
# --------------------------------------------------------------------------- #
def denormalize_rgb(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float().numpy().transpose(1, 2, 0)
    x = x * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    return (x.clip(0, 1) * 255).astype(np.uint8)


def denormalize_thermal(t: torch.Tensor, mean: float = GRAY_MEAN, std: float = GRAY_STD) -> np.ndarray:
    x = t.detach().cpu().float().numpy()[0] * std + mean
    return (x.clip(0, 1) * 255).astype(np.uint8)


def colorize_mask(mask) -> np.ndarray:
    """[H,W] class ids -> uint8 RGB using the fixed PALETTE (id 255 / unknown ids -> white)."""
    m = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    out = np.full(m.shape + (3,), 255, dtype=np.uint8)
    for cid in range(NUM_CLASSES):
        out[m == cid] = PALETTE[cid]
    return out


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=PALETTE[i] / 255.0, edgecolor="k", label=f"{i} {n}") for i, n in enumerate(CLASS_NAMES)]


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _fmt_counts(hist: np.ndarray) -> str:
    ids = [i for i in range(256) if hist[i] > 0]
    return ", ".join(f"{i}:{int(hist[i]):,}" for i in ids)


def check_pipeline_alignment(n_trials: int = 20, seed: int = 0) -> Tuple[bool, str]:
    """Self-test of the augmentation pipeline on SYNTHETIC arrays (explicitly not real data).

    RGB, thermal and label are all derived from one random ids map; after the shared flip/crop/
    resize the id recovered from the RGB and thermal channels must still match the label for
    nearly every pixel (bilinear blending only differs at class boundaries)."""
    rng = np.random.RandomState(seed)
    state = random.getstate()
    random.seed(seed)
    worst = 1.0
    try:
        for _ in range(n_trials):
            ids = np.kron(rng.randint(0, NUM_CLASSES, size=(12, 16)), np.ones((40, 40), dtype=np.int64)).astype(np.uint8)
            rgb = Image.fromarray(np.repeat((ids * 25)[..., None], 3, axis=2).astype(np.uint8))
            th = Image.fromarray((ids * 25).astype(np.float32))
            lab = Image.fromarray(ids)
            params = sample_aug_params(*lab.size, 0.5, 0.1, 1.0)
            r, t, y = transform_triplet(rgb, th, lab, (360, 480), params)
            y = np.asarray(y).astype(np.int64)
            for arr in (np.asarray(r)[..., 0].astype(np.float32), np.asarray(t, dtype=np.float32)):
                match = float((np.rint(arr / 25.0) == y).mean())
                worst = min(worst, match)
            if not set(np.unique(y)).issubset(set(range(NUM_CLASSES))):
                return False, "label ids changed during augmentation"
    finally:
        random.setstate(state)
    return worst > 0.90, f"worst RGB/thermal-vs-label agreement over {n_trials} random augmentations: {100 * worst:.2f}%"


def diagnose(root: Path, out_dir: Path, print_masks: int = 5, preview_n: int = 4,
             img_size: Tuple[int, int] = NATIVE_HW, thermal_norm: str = "dataset") -> bool:
    """Full label/alignment diagnostic.  Writes only to out_dir.  Returns True if all checks pass."""
    ok = True
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = validate_dataset(root, SPLITS, scan_labels=True, deep=False, verbose=True)
    if not rep.ok:
        print("Dataset validation failed -> fix that first.")
        return False

    print("\n=== 1) Per-mask class ids and pixel counts (CSV has every mask) ===")
    per_split_hist: Dict[str, np.ndarray] = {}
    only0: Dict[str, List[str]] = {}
    present_masks: Dict[str, np.ndarray] = {}
    for s in SPLITS:
        names = read_split_file(root, s)[0]
        total = np.zeros(256, dtype=np.int64)
        presence = np.zeros(NUM_CLASSES, dtype=np.int64)
        only0[s] = []
        csv_path = unique_path(out_dir / f"mask_stats_{s}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample"] + [f"class{i}_px" for i in range(NUM_CLASSES)] + ["other_ids_px", "unique_ids"])
            shown = 0
            for n in names:
                h = label_histogram(label_path(root, n))
                total += h
                ids = [i for i in range(256) if h[i] > 0]
                for i in ids:
                    if i < NUM_CLASSES:
                        presence[i] += 1
                if ids == [0]:
                    only0[s].append(n)
                w.writerow([n] + [int(h[i]) for i in range(NUM_CLASSES)] + [int(h[NUM_CLASSES:].sum()), " ".join(map(str, ids))])
                if print_masks < 0 or shown < print_masks:
                    print(f"  [{s}] {n}: ids {ids} | {_fmt_counts(h)}")
                    shown += 1
        per_split_hist[s], present_masks[s] = total, presence
        print(f"  [{s}] full per-mask table -> {csv_path}")

    print("\n=== 2) Class share of pixels per split ===")
    head = f"{'class':<24}" + "".join(f"{s:>12}" for s in SPLITS) + "   | masks containing class (train/val/test)"
    print(head)
    for i in range(NUM_CLASSES):
        row = f"{i} {CLASS_NAMES[i]:<22}"
        for s in SPLITS:
            row += f"{100 * per_split_hist[s][i] / max(per_split_hist[s].sum(), 1):11.3f}%"
        row += "   | " + " / ".join(f"{present_masks[s][i]}/{rep.counts[s]}" for s in SPLITS)
        print(row)
    bad_ids = {s: int(h[NUM_CLASSES:].sum() - h[IGNORE_INDEX]) for s, h in per_split_hist.items()}
    print("  ids outside 0..8 (excluding 255): " + ", ".join(f"{s}={v}" for s, v in bad_ids.items()))
    ok &= all(v == 0 for v in bad_ids.values())

    print("\n=== 3) Masks that contain ONLY class 0 ===")
    for s in SPLITS:
        lst = only0[s]
        print(f"  {s}: {len(lst)} of {rep.counts[s]} masks" + (f"  e.g. {lst[:5]}" if lst else ""))

    print("\n=== 4) Alignment and transformed-mask checks ===")
    ds_cache = {}
    t_mean, t_std = (thermal_stats_for_train(root, None, False) if thermal_norm == "dataset" else (GRAY_MEAN, GRAY_STD))
    for s, aug in (("train", True), ("val", False), ("test", False)):
        ds = RGBTDataset(root, s, img_size, augment=aug, thermal_mean=t_mean, thermal_std=t_std)
        ds_cache[s] = ds
        idx = np.unique(np.linspace(0, len(ds) - 1, num=min(12, len(ds))).round().astype(int))
        good = True
        for i in idx:
            rgb_r, th_r, lab_r = ds.load_raw(ds.names[i])               # raw sizes already checked equal
            for _ in range(3 if aug else 1):
                rgb, th, y = ds[i]
                ids = set(torch.unique(y).tolist())
                good &= (rgb.shape[1:] == th.shape[1:] == y.shape == tuple(img_size)
                         and rgb.shape[0] == 3 and th.shape[0] == 1 and y.dtype == torch.int64
                         and ids.issubset(set(range(NUM_CLASSES)) | {IGNORE_INDEX})
                         and bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(th).all()))
        print(f"  {s}{' (augmented)' if aug else ''}: {len(idx)} samples checked -> shapes aligned, ids valid, finite: {good}")
        ok &= good
    good, msg = check_pipeline_alignment()
    print(f"  synchronised-augmentation self-test (synthetic arrays): {'PASS' if good else 'FAIL'} - {msg}")
    ok &= good
    print(f"  thermal normalisation in use: {thermal_norm} (mean {t_mean:.4f}, std {t_std:.4f})")

    print("\n=== 5) Plots ===")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -> plots skipped (pip install matplotlib)")
        plt = None
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        wdt = 0.27
        for k, s in enumerate(SPLITS):
            share = 100 * per_split_hist[s][:NUM_CLASSES] / max(per_split_hist[s].sum(), 1)
            ax.bar(np.arange(NUM_CLASSES) + (k - 1) * wdt, share, wdt, label=s)
        ax.set_yscale("log"); ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels([f"{i}\n{n.split('/')[0]}" for i, n in enumerate(CLASS_NAMES)], fontsize=8)
        ax.set_ylabel("% of pixels (log)"); ax.set_title("Label pixel share per class and split"); ax.legend()
        fig.tight_layout()
        p = unique_path(out_dir / "label_distribution.png"); fig.savefig(p, dpi=110); plt.close(fig)
        print(f"  {p}")
        for s in SPLITS:
            ds = ds_cache[s]
            ds_plain = RGBTDataset(root, s, img_size, augment=False, thermal_mean=t_mean, thermal_std=t_std)
            names = ds.names
            # pick evenly spaced samples, but prefer ones that contain many different classes
            cand = np.unique(np.linspace(0, len(names) - 1, num=min(40, len(names))).round().astype(int))
            def n_cls(i):
                return int((label_histogram(label_path(root, names[i]))[1:NUM_CLASSES] > 0).sum())
            ranked = sorted(cand, key=lambda i: -n_cls(i))[:preview_n]
            fig, axes = plt.subplots(len(ranked), 4, figsize=(16, 3.6 * len(ranked) + 1.0), squeeze=False)
            for r, i in enumerate(ranked):
                rgb, th, y = ds_plain[i]
                rgb_u8, th_u8, col = denormalize_rgb(rgb), denormalize_thermal(th, t_mean, t_std), colorize_mask(y)
                over = (0.55 * rgb_u8 + 0.45 * col).astype(np.uint8)
                for c, (img, cm, ttl) in enumerate([(rgb_u8, None, f"RGB [{names[i]}]"), (th_u8, "gray", "Thermal"),
                                                    (col, None, "Ground truth (fixed palette)"),
                                                    (over, None, "GT overlay on RGB")]):
                    axes[r, c].imshow(img, cmap=cm); axes[r, c].axis("off"); axes[r, c].set_title(ttl, fontsize=9)
            fig.legend(handles=legend_handles(), loc="lower center", ncol=5, fontsize=9, frameon=False)
            fig.suptitle(f"{s}: RGB / thermal / ground truth (predictions are added by train19_v2.py)", fontsize=12)
            fig.tight_layout(rect=(0, 0.07, 1, 0.97))
            p = unique_path(out_dir / f"label_preview_{s}.png"); fig.savefig(p, dpi=95); plt.close(fig)
            print(f"  {p}")
    print("\nNOTE: class 0 = 'unlabeled/background' in MFNet.  The official RTFNet code trains with plain "
          "cross-entropy that INCLUDES class 0 and reports IoU for all 9 classes.")
    print("DATASET19_V2 DIAGNOSTIC " + ("PASSED" if ok else "FAILED"))
    return bool(ok)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate / diagnose the prepared RGB-T dataset (read-only)")
    p.add_argument("--data-root", default=None, help="default: auto-detect ../MFNet_dataset (Kaggle raw) first, then ../MFNet_CMX (prepared)")
    p.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), default=list(NATIVE_HW))
    p.add_argument("--diagnose", action="store_true", help="full label / alignment diagnostic + preview plots")
    p.add_argument("--out-dir", default=None, help="default: plots/model19_v2/diagnostics next to this file")
    p.add_argument("--print-masks", type=int, default=5, help="masks per split printed to the console (-1 = all)")
    p.add_argument("--preview-samples", type=int, default=4)
    p.add_argument("--thermal-norm", choices=THERMAL_NORMS, default="dataset")
    p.add_argument("--no-label-scan", action="store_true")
    p.add_argument("--deep", action="store_true", help="also fully decode every RGB/thermal PNG")
    a = p.parse_args()
    root = resolve_data_root(a.data_root)
    if a.diagnose:
        out = Path(a.out_dir) if a.out_dir else Path(__file__).resolve().parent / "plots" / "model19_v2" / "diagnostics"
        return 0 if diagnose(root, out, a.print_masks, a.preview_samples, tuple(a.img_size), a.thermal_norm) else 1
    rep = validate_dataset(root, scan_labels=not a.no_label_scan, deep=a.deep)
    if not rep.ok:
        return 1
    ds = RGBTDataset(root, "train", tuple(a.img_size), augment=True)
    rgb, th, y = ds[0]
    print(f"sample {ds.sample_name(0)!r}: rgb {tuple(rgb.shape)} thermal {tuple(th.shape)} mask {tuple(y.shape)} "
          f"ids {torch.unique(y).tolist()}")
    print("DATASET19_V2 CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
