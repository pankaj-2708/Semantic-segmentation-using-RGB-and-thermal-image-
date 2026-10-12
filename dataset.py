"""
dataset.py - Dataset loaders and transforms for Semantic Segmentation.
Supports LoveDA dataset as well as synthetic dataset for dry-runs.
"""

import os
import glob
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Global Parameters / Configuration
# ==========================================
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
NUM_CLASSES = 7
IGNORE_INDEX = 255
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]
NUM_WORKERS = 0  # Safe default on Windows
PIN_MEMORY = True


class SegmentationDataset(Dataset):
    """
    Segmentation dataset for loading image and mask pairs.
    Mask values outside [0, NUM_CLASSES - 1] are mapped to IGNORE_INDEX.
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        img_size: Tuple[int, int] = IMAGE_SIZE,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        is_train: bool = True,
    ):
        self.image_files = []
        self.mask_files = []
        self.img_size = img_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.is_train = is_train

        for img_dir, mask_dir in zip(image_paths, mask_paths):
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                for fname in sorted(os.listdir(img_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        self.image_files.append(os.path.join(img_dir, fname))
                        self.mask_files.append(os.path.join(mask_dir, fname))

        # Try importing albumentations; fallback to torchvision/numpy if unavailable
        self.has_albumentations = False
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            self.has_albumentations = True
            if self.is_train:
                self.transform = A.Compose([
                    A.Resize(img_size[0], img_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(img_size[0], img_size[1]),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2(),
                ])
        except ImportError:
            self.has_albumentations = False

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p = self.image_files[idx]
        msk_p = self.mask_files[idx]

        image = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(msk_p))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.has_albumentations:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            # Fallback PIL/torch resizing and normalization
            pil_img = Image.fromarray(image).resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            pil_msk = Image.fromarray(mask).resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

            img_np = np.array(pil_img, dtype=np.float32) / 255.0
            img_np = (img_np - np.array(NORM_MEAN)) / np.array(NORM_STD)
            image_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(np.array(pil_msk, dtype=np.int64)).long()

        # Handle mask boundaries and ignore index
        # Any label >= num_classes or < 0 is treated as ignore_index
        invalid_mask = (mask_tensor >= self.num_classes) | (mask_tensor < 0)
        mask_tensor[invalid_mask] = self.ignore_index

        return image_tensor, mask_tensor


class SyntheticSegmentationDataset(Dataset):
    """
    Synthetic dataset for testing and dry-running training loops.
    """
    def __init__(
        self,
        num_samples: int = 24,
        img_size: Tuple[int, int] = (256, 256),
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(idx)
        image = torch.randn(3, self.img_size[0], self.img_size[1])
        # Random class masks including occasional ignore_index
        mask = torch.randint(0, self.num_classes + 1, (self.img_size[0], self.img_size[1]), dtype=torch.long)
        mask[mask == self.num_classes] = self.ignore_index
        return image, mask


def get_dataloaders(
    data_root: Optional[str] = None,
    batch_size: int = 6,
    img_size: Tuple[int, int] = IMAGE_SIZE,
    num_workers: int = NUM_WORKERS,
    use_synthetic: bool = False,
    synthetic_samples: int = 24,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and validation DataLoaders.
    """
    if use_synthetic or not data_root or not os.path.exists(data_root):
        train_ds = SyntheticSegmentationDataset(
            num_samples=synthetic_samples,
            img_size=img_size,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
        )
        val_ds = SyntheticSegmentationDataset(
            num_samples=max(8, synthetic_samples // 4),
            img_size=img_size,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
        )
    else:
        train_img_dirs = [
            os.path.join(data_root, "Train/Train/Rural/images_png"),
            os.path.join(data_root, "Train/Train/Urban/images_png"),
        ]
        train_msk_dirs = [
            os.path.join(data_root, "Train/Train/Rural/masks_png"),
            os.path.join(data_root, "Train/Train/Urban/masks_png"),
        ]
        val_img_dirs = [
            os.path.join(data_root, "Val/Val/Rural/images_png"),
            os.path.join(data_root, "Val/Val/Urban/images_png"),
        ]
        val_msk_dirs = [
            os.path.join(data_root, "Val/Val/Rural/masks_png"),
            os.path.join(data_root, "Val/Val/Urban/masks_png"),
        ]

        train_ds = SegmentationDataset(
            image_paths=train_img_dirs,
            mask_paths=train_msk_dirs,
            img_size=img_size,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            is_train=True,
        )
        val_ds = SegmentationDataset(
            image_paths=val_img_dirs,
            mask_paths=val_msk_dirs,
            img_size=img_size,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            is_train=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
