"""
download_dataset.py - Helper script to download the MFNet RGB+Thermal dataset from Kaggle.

Usage:
    python download_dataset.py [--output-dir ../MFNet_dataset]

The MFNet dataset contains:
  - 1569 images (820 daytime, 749 nighttime)
  - 4-channel PNG files (RGB + Thermal in the alpha channel)
  - Single-channel label masks (0-8 class IDs)
  - train.txt / val.txt / test.txt split files

Kaggle Dataset: danialqashqai/mfnet-dataset
"""

import os
import sys
import shutil
import argparse
import subprocess


KAGGLE_DATASET_ID = "danialqashqai/mfnet-dataset"


def try_kaggle(output_dir: str) -> bool:
    """Download MFNet dataset via kaggle CLI."""
    try:
        import kaggle
    except ImportError:
        print("[downloader] Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    print(f"[downloader] Attempting Kaggle download from: {KAGGLE_DATASET_ID}")
    print(f"[downloader] Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    try:
        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET_ID,
            "--unzip",
            "-p", output_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[downloader] Kaggle download failed: {res.stderr or res.stdout}")
            return False

        # Flatten nested directory structure if needed
        nested_candidates = [
            os.path.join(output_dir, "MFNet-dataset", "ir_seg_dataset"),
            os.path.join(output_dir, "MFNet-dataset"),
            os.path.join(output_dir, "ir_seg_dataset"),
        ]
        for nested in nested_candidates:
            if os.path.isdir(nested) and check_dataset(nested):
                for item in os.listdir(nested):
                    s = os.path.join(nested, item)
                    d = os.path.join(output_dir, item)
                    if not os.path.exists(d):
                        shutil.move(s, d)
                try:
                    shutil.rmtree(os.path.join(output_dir, "MFNet-dataset"))
                except Exception:
                    pass
                break

        print("[downloader] Kaggle download and extraction succeeded!")
        return True
    except Exception as e:
        print(f"[downloader] Kaggle download error: {e}")
        return False


def check_dataset(output_dir: str) -> bool:
    """Verify that the dataset structure is complete."""
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    train_txt  = os.path.join(output_dir, "train.txt")

    checks = {
        "images/ dir": os.path.isdir(images_dir),
        "labels/ dir": os.path.isdir(labels_dir),
        "train.txt":   os.path.isfile(train_txt),
    }

    for name, ok in checks.items():
        status = "[OK]" if ok else "[X]"
        print(f"  {status} {name}")

    if all(checks.values()):
        n_imgs = len([f for f in os.listdir(images_dir) if f.endswith(".png")])
        n_lbls = len([f for f in os.listdir(labels_dir) if f.endswith(".png")])
        print(f"  Images: {n_imgs}  Labels: {n_lbls}")
        return n_imgs > 0 and n_lbls > 0
    return False


def print_manual_instructions(output_dir: str):
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("The automated download failed.")
    print()
    print("Please download manually:")
    print(f"  1. Kaggle CLI: kaggle datasets download -d {KAGGLE_DATASET_ID} --unzip -p {output_dir}")
    print(f"  2. Or download from: https://www.kaggle.com/datasets/{KAGGLE_DATASET_ID}")
    print("  3. The dataset should have this structure:")
    print(f"     {output_dir}/")
    print(f"     {output_dir}/images/   <- 4-channel PNG files (RGB+Thermal)")
    print(f"     {output_dir}/labels/   <- Single-channel mask PNG files")
    print(f"     {output_dir}/train.txt")
    print(f"     {output_dir}/val.txt")
    print(f"     {output_dir}/test.txt")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download MFNet Dataset from Kaggle")
    parser.add_argument(
        "--output-dir", type=str, default="../MFNet_dataset",
        help="Directory to save the dataset"
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    print(f"Target directory: {output_dir}")

    # Check if already present
    if check_dataset(output_dir):
        print("\n[downloader] Dataset already present and complete!")
        return

    # Download from Kaggle
    ok = try_kaggle(output_dir)

    print("\n[downloader] Verifying dataset after download...")
    if check_dataset(output_dir):
        print("[downloader] Dataset downloaded and verified successfully!")
    else:
        print_manual_instructions(output_dir)


if __name__ == "__main__":
    main()
