"""
============================================================
Facial Skin Condition Detection
Phase 3 — Feature Extraction
============================================================

Extracts a 58-dimensional hand-crafted feature vector from each image:
  - Color Features   (12): RGB + HSV channel mean & std-dev
  - GLCM Features    (20): Contrast, Dissimilarity, Homogeneity,
                            Energy, Correlation @ 4 angles
  - LBP Features     (26): Local Binary Pattern histogram (P=8, R=1)

Functions
---------
extract_color_features(img_bgr)   → np.ndarray (12,)
extract_glcm_features(gray)       → np.ndarray (20,)
extract_lbp_features(gray)        → np.ndarray (26,)
extract_all_features(img_bgr)     → np.ndarray (58,)
run_feature_extraction()          → saves features/features.csv

Run standalone:
    python src/feature_extraction.py

Author: MCA Project
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# scikit-image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Local imports
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from preprocessing import load_and_preprocess_image

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
FEATURES_DIR = PROJECT_ROOT / "features"

CLASS_FOLDERS = {
    "Acne":                  "Acne",
    "Rosacea":               "Rosacea",
    "Melasma":               "Melasma",
    "Seborrheic Dermatitis": "Seborrheic_Dermatitis",
}
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# LBP parameters
LBP_P      = 8      # number of circularly symmetric neighbour set points
LBP_R      = 1      # radius of circle
LBP_METHOD = "uniform"

# GLCM parameters
GLCM_DISTANCES = [1]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS     = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
# 5 properties × 4 angles = 20 features


# ── Feature Extraction Functions ──────────────────────────────────────────────

def extract_color_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract 12 colour statistics from RGB and HSV colour spaces.

    Feature layout (12 total):
        R_mean, R_std, G_mean, G_std, B_mean, B_std  (6)
        H_mean, H_std, S_mean, S_std, V_mean, V_std  (6)

    Parameters
    ----------
    img_bgr : np.ndarray
        Preprocessed BGR image (H × W × 3).

    Returns
    -------
    np.ndarray, shape (12,)
    """
    # RGB channel stats
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    rgb_feats = np.array([
        r.mean(), r.std(),
        g.mean(), g.std(),
        b.mean(), b.std(),
    ], dtype=np.float32)

    # HSV channel stats
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

    hsv_feats = np.array([
        h.mean(), h.std(),
        s.mean(), s.std(),
        v.mean(), v.std(),
    ], dtype=np.float32)

    return np.concatenate([rgb_feats, hsv_feats])   # (12,)


def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    """
    Extract 20 GLCM texture features.

    Computed for distances=[1] at angles [0°, 45°, 90°, 135°]:
        contrast, dissimilarity, homogeneity, energy, correlation
    → 5 properties × 4 angles = 20 features

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (H × W), uint8.

    Returns
    -------
    np.ndarray, shape (20,)
    """
    # Quantise to 256 levels (standard for GLCM)
    glcm = graycomatrix(
        gray,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )

    feats = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop).flatten()  # shape: (1, 4) → flatten to (4,)
        feats.extend(values.tolist())

    return np.array(feats, dtype=np.float32)   # (20,)


def extract_lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    Extract a normalised LBP histogram with 26 bins.

    Uses uniform LBP with P=8 neighbours, radius=1.
    Number of bins = P + 2 = 10 for uniform patterns,
    but scikit-image returns P*(P-1)+3 = 59 bins for 'uniform' —
    we take the first 26 most informative ones.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (H × W), uint8.

    Returns
    -------
    np.ndarray, shape (26,)
    """
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2   # uniform: 10 bins for P=8
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    # Pad / trim to exactly 26 dims to match specification
    hist_26 = np.zeros(26, dtype=np.float32)
    copy_len = min(len(hist), 26)
    hist_26[:copy_len] = hist[:copy_len].astype(np.float32)
    return hist_26   # (26,)


def extract_all_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract the full 58-dimensional feature vector from a preprocessed BGR image.

    Layout: [color(12) | glcm(20) | lbp(26)] = 58

    Parameters
    ----------
    img_bgr : np.ndarray
        Preprocessed BGR image (128 × 128 × 3).

    Returns
    -------
    np.ndarray, shape (58,)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    color_feats = extract_color_features(img_bgr)    # (12,)
    glcm_feats  = extract_glcm_features(gray)        # (20,)
    lbp_feats   = extract_lbp_features(gray)         # (26,)

    return np.concatenate([color_feats, glcm_feats, lbp_feats])   # (58,)


# ── Column Names ──────────────────────────────────────────────────────────────

def _get_feature_names() -> list[str]:
    """Return the ordered list of feature column names (58 total)."""
    names = []
    # Color (12)
    for ch in ["R", "G", "B"]:
        names += [f"{ch}_mean", f"{ch}_std"]
    for ch in ["H", "S", "V"]:
        names += [f"{ch}_mean", f"{ch}_std"]
    # GLCM (20)
    angles_deg = [0, 45, 90, 135]
    for prop in GLCM_PROPS:
        for a in angles_deg:
            names.append(f"GLCM_{prop}_{a}deg")
    # LBP (26)
    for i in range(26):
        names.append(f"LBP_bin_{i}")
    return names


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_feature_extraction() -> pd.DataFrame:
    """
    Scan all class folders, extract features from every valid image,
    and save the result to features/features.csv.

    Returns
    -------
    pd.DataFrame
        The extracted feature DataFrame (rows = images, cols = features + label).
    """
    print("\n" + "=" * 60)
    print("  FEATURE EXTRACTION — Building features/features.csv")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("        Run dataset_analysis.py first, then add images.")
        sys.exit(1)

    feature_names = _get_feature_names()
    records = []
    skipped = 0

    for class_name, folder in CLASS_FOLDERS.items():
        class_dir = DATASET_DIR / folder
        if not class_dir.exists():
            print(f"\n[WARNING] Folder missing: {class_dir} — skipping class '{class_name}'")
            continue

        images = sorted([f for f in class_dir.rglob("*")
                         if f.is_file() and f.suffix.lower() in SUPPORTED_EXT])

        if not images:
            print(f"\n[WARNING] No images found in '{class_name}'")
            continue

        print(f"\n[{class_name}] Processing {len(images)} image(s) …")

        for img_path in tqdm(images, desc=f"  {class_name:<28}", unit="img"):
            img = load_and_preprocess_image(img_path)
            if img is None:
                skipped += 1
                continue
            try:
                feats = extract_all_features(img)
                row   = dict(zip(feature_names, feats.tolist()))
                row["label"]    = class_name
                row["filepath"] = str(img_path)
                records.append(row)
            except Exception as e:
                print(f"\n  [ERROR] {img_path.name}: {e}")
                skipped += 1

    if not records:
        print("\n[ERROR] No features extracted. Check that dataset/ folders contain images.")
        sys.exit(1)

    df = pd.DataFrame(records)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "features.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Total images processed : {len(records)}")
    print(f"  Total images skipped   : {skipped}")
    print(f"  Feature vector size    : {len(feature_names)}")
    print(f"  Saved → {out_path}")
    print(f"{'=' * 60}")

    # Class distribution
    print("\n  Class distribution in features.csv:")
    for cls, cnt in df["label"].value_counts().items():
        print(f"    {cls:<30} {cnt}")

    print("\n  Next step → python src/train_models.py")
    return df


if __name__ == "__main__":
    run_feature_extraction()
