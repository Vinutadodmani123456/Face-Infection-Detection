"""
============================================================
Facial Skin Condition Detection
Fast Multi-threaded Dataset Cleaning Pipeline
============================================================

Performs automated cleaning on the dataset:
  1. Detects and removes exact duplicate images (MD5 hashing).
  2. Detects and removes corrupt/unreadable image files.
  3. Detects and removes non-facial / body part images (hands, legs, back, etc.).

Usage:
    python src/clean_dataset.py

Author: MCA Project
"""

import os
import sys
import hashlib
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"

CLASS_FOLDERS = {
    "Acne":                  DATASET_DIR / "Acne",
    "Rosacea":               DATASET_DIR / "Rosacea",
    "Melasma":               DATASET_DIR / "Melasma",
    "Seborrheic Dermatitis": DATASET_DIR / "Seborrheic_Dermatitis",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_facial_skin_image(img_path: Path) -> tuple[bool, str]:
    """
    Check if an image is readable, uncorrupted, and represents a facial skin image.
    Rejects body parts (hands, back, feet) and non-facial images.
    """
    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        return False, "Corrupt or unreadable image"

    h, w = img.shape[:2]
    if h < 30 or w < 30:
        return False, "Image dimensions too small"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Face cascade check
    cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    faces = cascade_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        faces = cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(faces) > 0:
        return True, "Face detected"

    # 2. Skin Color Analysis (For close-up facial skin patches)
    B = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    R = img[:, :, 2].astype(float)
    rgb_skin = (R > G) & (G > B) & (R > 40) & (G > 20) & (B > 10) & ((R - G) >= 6)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]
    Y  = ycrcb[:, :, 0]
    ycrcb_skin = (Y >= 40) & (Cr >= 130) & (Cr <= 180) & (Cb >= 75) & (Cb <= 130)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    hsv_skin = ((H <= 25) | (H >= 155)) & (S >= 10) & (S <= 190)

    strict_skin = rgb_skin & ycrcb_skin & hsv_skin
    skin_percentage = float((np.sum(strict_skin) / strict_skin.size) * 100.0)

    # Check for non-facial body geometry or non-skin artwork colors
    non_skin_mask = (H >= 35) & (H <= 150) & (S >= 40) & (V >= 40)
    non_skin_percentage = float((np.sum(non_skin_mask) / non_skin_mask.size) * 100.0)

    if skin_percentage < 15.0 or non_skin_percentage > 25.0:
        return False, "Not a valid facial skin region"

    return True, "Valid skin patch"


def process_single_file(filepath: Path) -> tuple[str, bool, str]:
    """Process a single file: hash and face/skin validation."""
    try:
        fhash = get_file_hash(filepath)
    except Exception:
        return "", False, "Corrupt file hash"

    is_valid, reason = is_facial_skin_image(filepath)
    return fhash, is_valid, reason


def clean_dataset():
    print("=" * 60, flush=True)
    print("  FACIAL SKIN DATASET CLEANING PIPELINE (Fast Multi-threaded)", flush=True)
    print("=" * 60, flush=True)

    seen_hashes = set()
    total_scanned = 0
    total_duplicates_removed = 0
    total_invalid_removed = 0
    total_kept = 0

    for class_name, folder in CLASS_FOLDERS.items():
        if not folder.exists():
            print(f"  [SKIP] Folder not found: {folder}", flush=True)
            continue

        print(f"\n[{class_name}] Cleaning folder: {folder.name} …", flush=True)

        image_files = [
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

        class_scanned = len(image_files)
        class_dups = 0
        class_invalid = 0
        class_kept = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_single_file, image_files))

        for filepath, (fhash, is_valid, reason) in zip(image_files, results):
            total_scanned += 1

            if not fhash or fhash in seen_hashes:
                filepath.unlink(missing_ok=True)
                class_dups += 1
                total_duplicates_removed += 1
                continue

            seen_hashes.add(fhash)

            if not is_valid:
                filepath.unlink(missing_ok=True)
                class_invalid += 1
                total_invalid_removed += 1
                continue

            class_kept += 1
            total_kept += 1

        print(f"  Scanned            : {class_scanned}", flush=True)
        print(f"  Duplicates Removed : {class_dups}", flush=True)
        print(f"  Invalid/Non-Facial : {class_invalid}", flush=True)
        print(f"  Clean Images Kept  : {class_kept}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  CLEANING SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Total Images Scanned          : {total_scanned}", flush=True)
    print(f"  Total Duplicate Files Removed : {total_duplicates_removed}", flush=True)
    print(f"  Total Non-Facial/Corrupt Files: {total_invalid_removed}", flush=True)
    print(f"  Total Clean Facial Images Kept : {total_kept}", flush=True)
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    clean_dataset()
