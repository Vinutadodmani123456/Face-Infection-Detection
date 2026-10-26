"""
============================================================
Facial Skin Condition Detection
Phase 1 — Dataset Analysis
============================================================

This module:
  - Scans all five class folders under dataset/
  - Verifies image readability and format
  - Counts images per class
  - Detects class imbalance
  - Detects potential duplicate images (via file size hashing)
  - Displays dataset statistics
  - Generates a summary report

Run:
    python src/dataset_analysis.py

Author: MCA Project
"""

import os
import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

# Root of the project (two levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR   = PROJECT_ROOT / "dataset"
FEATURES_DIR  = PROJECT_ROOT / "features"
REPORTS_DIR   = PROJECT_ROOT / "reports"

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# The four disease classes and their folder names
CLASS_FOLDERS = {
    "Acne":                    "Acne",
    "Rosacea":                 "Rosacea",
    "Melasma":                 "Melasma",
    "Seborrheic Dermatitis":   "Seborrheic_Dermatitis",
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def compute_file_hash(filepath: Path, block_size: int = 65536) -> str:
    """
    Compute an MD5 hash of a file's binary content.
    Used for exact-duplicate detection.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image(filepath: Path) -> tuple[bool, str]:
    """
    Try to read an image using OpenCV.

    Returns
    -------
    (True, "")              if the image is readable
    (False, reason_string)  if not readable
    """
    try:
        img = cv2.imread(str(filepath))
        if img is None:
            return False, "OpenCV returned None (corrupted or unreadable file)"
        if img.size == 0:
            return False, "Image has zero pixels"
        return True, ""
    except Exception as e:
        return False, str(e)


def get_image_dimensions(filepath: Path) -> tuple[int, int, int] | None:
    """
    Return (height, width, channels) of an image, or None on failure.
    """
    try:
        img = cv2.imread(str(filepath))
        if img is None:
            return None
        return img.shape if len(img.shape) == 3 else (*img.shape, 1)
    except Exception:
        return None


# ── Core Analysis ─────────────────────────────────────────────────────────────

def scan_dataset() -> dict:
    """
    Walk through every class folder, validate every image,
    and collect detailed statistics.

    Returns a dict with keys:
        class_stats   — per-class counts and details
        overall_stats — totals across all classes
        invalid_files — list of unreadable file paths + reasons
        duplicates    — list of (file_a, file_b) duplicate pairs
        image_sizes   — list of (height, width) tuples (valid images only)
    """

    print("\n" + "=" * 60)
    print("  FACIAL SKIN CONDITION DETECTION — Dataset Analysis")
    print("=" * 60)
    print(f"\nDataset directory : {DATASET_DIR}")

    # ── Check that the dataset directory exists ───────────────────────────
    if not DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("        Create the directory and add images before running this script.")
        sys.exit(1)

    class_stats   = {}
    invalid_files = []
    all_hashes    = {}   # hash → filepath  (for duplicate detection)
    duplicates    = []
    image_sizes   = []

    grand_total_found   = 0
    grand_total_valid   = 0
    grand_total_invalid = 0

    for class_name, folder_name in CLASS_FOLDERS.items():
        class_dir = DATASET_DIR / folder_name

        # ── Check that the class folder exists ───────────────────────────
        if not class_dir.exists():
            print(f"\n[WARNING] Folder not found for class '{class_name}': {class_dir}")
            class_stats[class_name] = {
                "folder":       str(class_dir),
                "folder_found": False,
                "total_files":  0,
                "valid":        0,
                "invalid":      0,
                "skipped_ext":  0,
                "image_records": [],
            }
            continue

        # ── Walk the folder (recursively) ────────────────────────────────
        all_files   = list(class_dir.rglob("*"))
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        skipped     = [f for f in all_files if f.is_file() and f.suffix.lower() not in SUPPORTED_EXTENSIONS]

        valid_count   = 0
        invalid_count = 0
        image_records = []   # list of dicts with per-image info

        print(f"\n[Scanning] {class_name:<28}  — {len(image_files)} image file(s) found …")

        for img_path in image_files:
            readable, reason = is_valid_image(img_path)

            if readable:
                # Duplicate detection via MD5 hash
                file_hash = compute_file_hash(img_path)
                if file_hash in all_hashes:
                    duplicates.append((all_hashes[file_hash], img_path))
                else:
                    all_hashes[file_hash] = img_path

                dims = get_image_dimensions(img_path)
                if dims:
                    image_sizes.append((dims[0], dims[1]))

                image_records.append({
                    "filepath":  str(img_path),
                    "filename":  img_path.name,
                    "class":     class_name,
                    "valid":     True,
                    "reason":    "",
                    "height":    dims[0] if dims else None,
                    "width":     dims[1] if dims else None,
                    "size_kb":   round(img_path.stat().st_size / 1024, 2),
                })
                valid_count += 1
            else:
                invalid_files.append((str(img_path), reason))
                image_records.append({
                    "filepath":  str(img_path),
                    "filename":  img_path.name,
                    "class":     class_name,
                    "valid":     False,
                    "reason":    reason,
                    "height":    None,
                    "width":     None,
                    "size_kb":   round(img_path.stat().st_size / 1024, 2),
                })
                invalid_count += 1

        class_stats[class_name] = {
            "folder":        str(class_dir),
            "folder_found":  True,
            "total_files":   len(image_files),
            "valid":         valid_count,
            "invalid":       invalid_count,
            "skipped_ext":   len(skipped),
            "image_records": image_records,
        }

        grand_total_found   += len(image_files)
        grand_total_valid   += valid_count
        grand_total_invalid += invalid_count

    overall_stats = {
        "total_found":   grand_total_found,
        "total_valid":   grand_total_valid,
        "total_invalid": grand_total_invalid,
    }

    return {
        "class_stats":   class_stats,
        "overall_stats": overall_stats,
        "invalid_files": invalid_files,
        "duplicates":    duplicates,
        "image_sizes":   image_sizes,
    }


# ── Report Generation ─────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """Print a formatted console summary of the dataset scan results."""

    cs = results["class_stats"]
    os_ = results["overall_stats"]

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)

    print(f"\n{'Class':<28} {'Folder Found':<14} {'Total':<8} {'Valid':<8} {'Invalid':<8}")
    print("-" * 70)

    for class_name, stats in cs.items():
        found   = "YES" if stats["folder_found"] else "NO ⚠"
        total   = stats["total_files"]
        valid   = stats["valid"]
        invalid = stats["invalid"]
        print(f"  {class_name:<26} {found:<14} {total:<8} {valid:<8} {invalid:<8}")

    print("-" * 70)
    print(f"  {'TOTAL':<26} {'':14} {os_['total_found']:<8} {os_['total_valid']:<8} {os_['total_invalid']:<8}")

    # ── Class imbalance check ─────────────────────────────────────────────
    valid_counts = {k: v["valid"] for k, v in cs.items() if v["folder_found"]}
    if valid_counts:
        max_cls = max(valid_counts, key=valid_counts.get)
        min_cls = min(valid_counts, key=valid_counts.get)
        max_n   = valid_counts[max_cls]
        min_n   = valid_counts[min_cls]

        print("\n── Class Imbalance Check ─────────────────────────────────")
        if max_n > 0 and min_n > 0:
            ratio = max_n / min_n
            print(f"  Largest  class : {max_cls} ({max_n} images)")
            print(f"  Smallest class : {min_cls} ({min_n} images)")
            print(f"  Imbalance ratio: {ratio:.2f}x")
            if ratio > 3:
                print("  [WARNING] Significant class imbalance detected (ratio > 3x).")
                print("            Consider using class_weight='balanced' in ML models.")
            else:
                print("  [OK] Class distribution is reasonably balanced.")
        else:
            print("  [WARNING] One or more classes have zero valid images.")

    # ── Duplicates ────────────────────────────────────────────────────────
    dups = results["duplicates"]
    print(f"\n── Duplicate Detection ──────────────────────────────────────")
    if dups:
        print(f"  [WARNING] {len(dups)} duplicate image pair(s) detected:")
        for a, b in dups[:10]:   # show max 10
            print(f"    • {Path(a).name}  ≡  {Path(b).name}")
        if len(dups) > 10:
            print(f"    … and {len(dups) - 10} more.")
    else:
        print("  [OK] No exact duplicate images detected.")

    # ── Invalid files ─────────────────────────────────────────────────────
    inv = results["invalid_files"]
    print(f"\n── Invalid / Corrupted Files ────────────────────────────────")
    if inv:
        print(f"  [WARNING] {len(inv)} unreadable image(s) found:")
        for path, reason in inv[:10]:
            print(f"    • {Path(path).name}: {reason}")
    else:
        print("  [OK] All images are readable.")

    # ── Image size statistics ─────────────────────────────────────────────
    sizes = results["image_sizes"]
    if sizes:
        heights = [s[0] for s in sizes]
        widths  = [s[1] for s in sizes]
        print(f"\n── Image Dimension Statistics ───────────────────────────────")
        print(f"  Height — min: {min(heights):4d}  max: {max(heights):4d}  mean: {np.mean(heights):.0f}")
        print(f"  Width  — min: {min(widths):4d}  max: {max(widths):4d}  mean: {np.mean(widths):.0f}")

    print("\n" + "=" * 60)


def save_summary_csv(results: dict) -> None:
    """Save per-image details to features/dataset_summary.csv for traceability."""

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "dataset_summary.csv"

    rows = []
    for class_name, stats in results["class_stats"].items():
        rows.extend(stats["image_records"])

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"\n[SAVED] Dataset summary CSV → {out_path}")
    else:
        print("\n[INFO] No image records to save (dataset is empty).")


def plot_class_distribution(results: dict) -> None:
    """
    Generate and save two charts:
      1. Bar chart of valid images per class
      2. Pie chart of class distribution
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cs = results["class_stats"]
    classes = list(cs.keys())
    counts  = [cs[c]["valid"] for c in classes]

    # ── Color palette ─────────────────────────────────────────────────────
    palette = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dataset Class Distribution", fontsize=16, fontweight="bold", y=1.02)

    # Bar chart
    sns.barplot(
        x=classes, y=counts,
        palette=palette, ax=axes[0], edgecolor="black"
    )
    axes[0].set_title("Valid Images per Class", fontsize=13)
    axes[0].set_xlabel("Skin Condition")
    axes[0].set_ylabel("Number of Images")
    axes[0].tick_params(axis="x", rotation=20)
    for bar, count in zip(axes[0].patches, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    # Pie chart
    non_zero = [(c, n) for c, n in zip(classes, counts) if n > 0]
    if non_zero:
        pie_labels, pie_counts = zip(*non_zero)
        axes[1].pie(
            pie_counts,
            labels=pie_labels,
            autopct="%1.1f%%",
            colors=palette[: len(non_zero)],
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        axes[1].set_title("Class Proportion", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    save_path = REPORTS_DIR / "class_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Class distribution chart → {save_path}")


def print_recommendations(results: dict) -> None:
    """Print actionable recommendations based on dataset analysis."""

    cs = results["class_stats"]
    os_ = results["overall_stats"]

    print("\n── Recommendations ──────────────────────────────────────────")

    if os_["total_valid"] == 0:
        print("  ❌ No valid images found. Add images to dataset/ folders before training.")
        return

    # Minimum images per class for reasonable ML training
    MIN_IMAGES = 30
    for class_name, stats in cs.items():
        if stats["valid"] < MIN_IMAGES:
            print(f"  ⚠ '{class_name}' has only {stats['valid']} valid images.")
            print(f"    Recommended minimum: {MIN_IMAGES} images per class.")

    valid_counts = [v["valid"] for v in cs.values() if v["folder_found"]]
    if valid_counts:
        ratio = max(valid_counts) / max(min(valid_counts), 1)
        if ratio > 3:
            print("  ⚠ Class imbalance detected. Use class_weight='balanced' during training.")
        else:
            print("  ✓ Dataset appears balanced enough for initial training.")

    if results["duplicates"]:
        print(f"  ⚠ {len(results['duplicates'])} duplicate(s) found. Remove them before training.")

    if results["invalid_files"]:
        print(f"  ⚠ {len(results['invalid_files'])} corrupted file(s) found. Remove or replace them.")

    print("  ✓ When dataset is ready, run: python src/feature_extraction.py")
    print("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_analysis() -> dict:
    """
    Main entry point.
    Runs the full dataset scan, prints summary, saves CSV, and creates charts.
    Returns the results dict for use in notebooks.
    """
    results = scan_dataset()
    print_summary(results)
    save_summary_csv(results)
    plot_class_distribution(results)
    print_recommendations(results)
    return results


if __name__ == "__main__":
    run_analysis()
