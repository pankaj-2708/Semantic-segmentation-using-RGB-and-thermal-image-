"""
dataset.py - MFNet Dataset Loader for RGB + Thermal Semantic Segmentation.

Dataset structure (MFNet):
  <data_root>/
    images/       - 4-channel PNG files (3 RGB + 1 Thermal), 640x480
    labels/       - Single-channel PNG masks, values 0-8
    train.txt     - Filenames (without extension) for training split
    val.txt       - Filenames (without extension) for validation split
    test.txt      - Filenames (without extension) for test split

MFNet class map (9 classes):
  0: Unlabeled / Background
  1: Car
  2: Person
  3: Bike
  4: Curve
  5: Car Stop
  6: Guardrail
  7: Color Cone
  8: Bump

Download:
  The dataset is automatically downloaded from Google Drive on first run if the
  data_root folder does not exist or is missing the expected subfolders.
  Google Drive folder ID: 18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw
"""

import os
import sys
import zipfile
import tarfile
import shutil
import subprocess
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Global Parameters / Configuration
# ==========================================
IMAGE_HEIGHT = 480
IMAGE_WIDTH  = 640
IMAGE_SIZE   = (IMAGE_HEIGHT, IMAGE_WIDTH)
NUM_CLASSES  = 9         # MFNet: 0-8
IN_CHANNELS  = 4         # 3 RGB + 1 Thermal
IGNORE_INDEX = 255
NUM_WORKERS  = 2

# Normalisation stats for RGB channels (ImageNet-like)
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
# Normalisation stats for thermal channel
THR_MEAN = [0.5]
THR_STD  = [0.5]

# MFNet class names
CLASS_NAMES = [
    "Unlabeled", "Car", "Person", "Bike",
    "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump",
]

# Google Drive folder ID for the MFNet dataset
GDRIVE_FOLDER_ID = "18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw"


# ==========================================
# Download Helper
# ==========================================

def _try_download_dataset(data_root: str) -> bool:
    """
    Attempts to download the MFNet dataset from Google Drive using gdown.
    Returns True if download succeeded, False otherwise.
    """
    try:
        import gdown
    except ImportError:
        print("[dataset.py] gdown not installed. Run: pip install gdown")
        return False

    print(f"[dataset.py] Attempting to download MFNet dataset to '{data_root}'...")
    print(f"[dataset.py] Google Drive Folder ID: {GDRIVE_FOLDER_ID}")

    os.makedirs(data_root, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

    try:
        result = gdown.download_folder(
            url,
            output=data_root,
            quiet=False,
            use_cookies=False,
            remaining_ok=True,   # Allow folders with >50 files
        )
        if result:
            print(f"[dataset.py] Download completed: {result}")
            return True
        else:
            print("[dataset.py] gdown returned no result (may have failed silently).")
            return False
    except Exception as e:
        print(f"[dataset.py] gdown download failed: {e}")
        return False


def _try_kaggle_download(data_root: str) -> bool:
    """Try downloading MFNet dataset via kaggle CLI."""
    try:
        os.makedirs(data_root, exist_ok=True)
        print(f"[dataset.py] Attempting Kaggle dataset download into '{data_root}'...")
        cmd = ["kaggle", "datasets", "download", "-d", "danialqashqai/mfnet-dataset", "--unzip", "-p", data_root]
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Check if extracted into a nested directory
        nested = os.path.join(data_root, "MFNet-dataset", "ir_seg_dataset")
        if os.path.isdir(nested):
            for item in os.listdir(nested):
                s = os.path.join(nested, item)
                d = os.path.join(data_root, item)
                if not os.path.exists(d):
                    shutil.move(s, d)
            try:
                shutil.rmtree(os.path.join(data_root, "MFNet-dataset"))
            except Exception:
                pass
        return res.returncode == 0
    except Exception as e:
        print(f"[dataset.py] Kaggle download failed: {e}")
        return False


def _dataset_is_ready(data_root: str) -> bool:
    """Check if the dataset appears to be present and usable."""
    images_dir = os.path.join(data_root, "images")
    labels_dir = os.path.join(data_root, "labels")
    train_txt  = os.path.join(data_root, "train.txt")

    if not os.path.isdir(images_dir):
        return False
    if not os.path.isdir(labels_dir):
        return False
    if not os.path.isfile(train_txt):
        return False

    # Check that there are actual image files
    imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
    lbls = [f for f in os.listdir(labels_dir) if f.lower().endswith(".png")]
    return len(imgs) > 0 and len(lbls) > 0


def ensure_dataset(data_root: str) -> bool:
    """
    Ensures the MFNet dataset is present at data_root.
    Downloads it if missing. Returns True if dataset is ready.
    """
    if _dataset_is_ready(data_root):
        print(f"[dataset.py] Dataset found at '{data_root}'.")
        return True

    print(f"[dataset.py] Dataset not found or incomplete at '{data_root}'. Attempting download...")
    ok = _try_download_dataset(data_root)

    if not ok or not _dataset_is_ready(data_root):
        print("[dataset.py] Google Drive download did not complete. Trying Kaggle mirror...")
        ok = _try_kaggle_download(data_root)

    if _dataset_is_ready(data_root):
        print(f"[dataset.py] Dataset ready at '{data_root}'.")
        return True

    print(
        "[dataset.py] ERROR: Could not download or verify MFNet dataset.\n"
        "  Manual download options:\n"
        f"    1. Google Drive: https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}\n"
        "    2. Kaggle dataset: danialqashqai/mfnet-dataset\n"
        "  Expected layout:\n"
        f"    {data_root}/images/  (4-channel PNG files)\n"
        f"    {data_root}/labels/  (greyscale mask PNG files)\n"
        f"    {data_root}/train.txt\n"
        f"    {data_root}/val.txt"
    )
    return False


# ==========================================
# Dataset Classes
# ==========================================

class MFNetDataset(Dataset):
    """
    MFNet RGB-Thermal Semantic Segmentation Dataset.

    Loads 4-channel images (RGB + Thermal) from PNG files and
    corresponding single-channel label masks.

    Args:
        data_root:     Path to the dataset root directory.
        split:         One of 'train', 'val', 'test'.
        img_size:      (H, W) to resize images and masks.
        augment:       Whether to apply data augmentation (train only).
        ignore_index:  Label value to ignore during loss computation.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Tuple[int, int] = IMAGE_SIZE,
        augment: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.data_root   = data_root
        self.split       = split
        self.img_size    = img_size        # (H, W)
        self.augment     = augment and (split == "train")
        self.ignore_index = ignore_index
        self.num_classes  = NUM_CLASSES

        self.images_dir  = os.path.join(data_root, "images")
        self.labels_dir  = os.path.join(data_root, "labels")

        # Read split file
        split_file = os.path.join(data_root, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"Split file not found: '{split_file}'. "
                f"Expected 'train.txt', 'val.txt', or 'test.txt' in '{data_root}'."
            )

        with open(split_file, "r") as f:
            self.names = [line.strip() for line in f if line.strip()]

        # Verify files exist (supports on-the-fly flip for _flip split entries)
        self.samples = []
        missing = []
        for name in self.names:
            img_p = os.path.join(self.images_dir, f"{name}.png")
            lbl_p = os.path.join(self.labels_dir, f"{name}.png")
            if os.path.isfile(img_p) and os.path.isfile(lbl_p):
                self.samples.append((img_p, lbl_p, False))
            elif name.endswith("_flip"):
                orig_name = name[:-5]
                orig_img_p = os.path.join(self.images_dir, f"{orig_name}.png")
                orig_lbl_p = os.path.join(self.labels_dir, f"{orig_name}.png")
                if os.path.isfile(orig_img_p) and os.path.isfile(orig_lbl_p):
                    self.samples.append((orig_img_p, orig_lbl_p, True))
                else:
                    missing.append(name)
            else:
                missing.append(name)

        if missing:
            print(
                f"[dataset.py] WARNING: {len(missing)}/{len(self.names)} files missing "
                f"for split='{split}'. First few: {missing[:5]}"
            )

        print(
            f"[dataset.py] Split='{split}' | "
            f"{len(self.samples)} samples found "
            f"({len(missing)} missing from split file)."
        )

        # Setup albumentations if available
        self._setup_transforms()

    def _setup_transforms(self):
        """Configure augmentation / normalization pipeline."""
        self._use_albumentations = False
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            if self.augment:
                self._transform = A.Compose([
                    A.Resize(self.img_size[0], self.img_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    ToTensorV2(),
                ])
            else:
                self._transform = A.Compose([
                    A.Resize(self.img_size[0], self.img_size[1]),
                    ToTensorV2(),
                ])
            self._use_albumentations = True
        except ImportError:
            pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p, lbl_p, need_flip = self.samples[idx]

        # Load 4-channel image (RGBA where A = thermal)
        pil_img = Image.open(img_p)
        img_arr = np.array(pil_img, copy=True)          # (H, W, 4) uint8

        if img_arr.ndim == 2:
            # Greyscale image — replicate to 3 channels, add zeros for thermal
            img_arr = np.stack([img_arr, img_arr, img_arr, np.zeros_like(img_arr)], axis=-1)
        elif img_arr.shape[2] == 3:
            # RGB only — add zeros for thermal channel
            img_arr = np.concatenate(
                [img_arr, np.zeros((*img_arr.shape[:2], 1), dtype=np.uint8)], axis=-1
            )
        # Now img_arr is (H, W, 4) in uint8 [0, 255]

        # Load label mask
        lbl_pil = Image.open(lbl_p)
        lbl_arr = np.array(lbl_pil, copy=True)
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[:, :, 0]         # Take first channel if RGB

        if need_flip:
            img_arr = np.fliplr(img_arr).copy()
            lbl_arr = np.fliplr(lbl_arr).copy()

        # Apply transforms and scale to [0, 1] (standard MFNet convention: raw image / 255.0)
        if self._use_albumentations:
            out = self._transform(image=img_arr, mask=lbl_arr)
            image_tensor = out["image"].float() / 255.0    # (4, H, W) in [0.0, 1.0]
            mask_tensor  = out["mask"].long()              # (H, W)
        else:
            # Manual fallback
            H, W = self.img_size
            pil_4ch = Image.fromarray(img_arr, mode="RGBA").resize((W, H), Image.BILINEAR)
            pil_lbl = Image.fromarray(lbl_arr).resize((W, H), Image.NEAREST)

            arr4 = np.array(pil_4ch, dtype=np.float32) / 255.0  # (H, W, 4) in [0.0, 1.0]
            image_tensor = torch.from_numpy(arr4.transpose(2, 0, 1)).float()
            mask_tensor  = torch.from_numpy(np.array(pil_lbl, dtype=np.int64)).long()

        # Clamp mask: set out-of-range values to ignore_index
        invalid = (mask_tensor < 0) | (mask_tensor >= self.num_classes)
        mask_tensor[invalid] = self.ignore_index

        return image_tensor, mask_tensor


# ==========================================
# DataLoader Factory
# ==========================================

def get_dataloaders(
    data_root: str = "../MFNet_dataset",
    batch_size: int = 4,
    img_size: Tuple[int, int] = IMAGE_SIZE,
    num_workers: int = NUM_WORKERS,
    auto_download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds and returns (train_loader, val_loader) for MFNet RGB+Thermal segmentation.

    Args:
        data_root:         Path to MFNet dataset root.
        batch_size:        Batch size.
        img_size:          (H, W) for resizing images and masks.
        num_workers:       DataLoader worker count.
        auto_download:     Whether to attempt auto-download if missing.
    """
    if not _dataset_is_ready(data_root):
        nested = os.path.join(data_root, "ir_seg_dataset")
        if _dataset_is_ready(nested):
            data_root = nested
        elif auto_download:
            ensure_dataset(data_root)
            if not _dataset_is_ready(data_root) and _dataset_is_ready(nested):
                data_root = nested

    if not _dataset_is_ready(data_root):
        raise FileNotFoundError(
            f"MFNet dataset is not found or incomplete at '{data_root}'.\n"
            "Please ensure the dataset is downloaded with 'images/', 'labels/', and 'train.txt'/'val.txt'."
        )

    print(f"[dataset.py] Loading MFNet dataset from: {data_root}")
    train_ds = MFNetDataset(
        data_root=data_root,
        split="train",
        img_size=img_size,
        augment=True,
    )
    val_ds = MFNetDataset(
        data_root=data_root,
        split="val",
        img_size=img_size,
        augment=False,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    return train_loader, val_loader
