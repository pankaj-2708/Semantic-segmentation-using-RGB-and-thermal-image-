"""
download_dataset.py - Helper script to download the MFNet RGB+Thermal dataset.

Usage:
    python download_dataset.py [--output-dir MFNet_dataset]

The MFNet dataset contains:
  - 1569 images (820 daytime, 749 nighttime)
  - 4-channel PNG files (RGB + Thermal in the alpha channel)
  - Single-channel label masks (0-8 class IDs)
  - train.txt / val.txt / test.txt split files

Google Drive Folder: https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw
"""

import os
import sys
import argparse
import subprocess


GDRIVE_FOLDER_ID = "18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw"


def try_gdown(output_dir: str) -> bool:
    """Try downloading via gdown."""
    try:
        import gdown
    except ImportError:
        print("[downloader] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    print(f"[downloader] Attempting gdown download from: {url}")
    print(f"[downloader] Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    try:
        result = gdown.download_folder(
            url,
            output=output_dir,
            quiet=False,
            use_cookies=False,
            # remaining_ok=True,
        )
        if result:
            print(f"[downloader] gdown succeeded: {result}")
            return True
        return False
    except Exception as e:
        print(f"[downloader] gdown failed: {e}")
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
        status = "✓" if ok else "✗"
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
    print("The automated download failed (old-style Google Drive file IDs")
    print("require authentication that gdown cannot provide without cookies).")
    print()
    print("Please download manually:")
    print(f"  1. Open: https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
    print("  2. Download all files (use 'Download all' or select each folder)")
    print("  3. The dataset should have this structure:")
    print(f"     {output_dir}/")
    print(f"     {output_dir}/images/   ← 4-channel PNG files (RGB+Thermal)")
    print(f"     {output_dir}/labels/   ← Single-channel mask PNG files")
    print(f"     {output_dir}/train.txt")
    print(f"     {output_dir}/val.txt")
    print(f"     {output_dir}/test.txt")
    print()
    print("Alternative: The RTFNet preprocessed version is also compatible:")
    print("  http://gofile.me/4jm56/CfukComo1  (may need gofile.me account)")
    print()
    print("For now, training will use SYNTHETIC data as a fallback.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download MFNet Dataset")
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

    # Try gdown
    ok = try_gdown(output_dir)

    print("\n[downloader] Verifying dataset after download...")
    if check_dataset(output_dir):
        print("[downloader] Dataset downloaded and verified successfully!")
    else:
        print_manual_instructions(output_dir)


if __name__ == "__main__":
    main()
