"""
============================================================
Facial Skin Condition Detection
Comprehensive Automated Rejection & Acceptance Test Suite
============================================================

Tests:
  1. Supported Disease Images (Acne, Melasma, Rosacea, Seborrheic Dermatitis)
  2. Unsupported Skin Conditions (Out-of-Distribution Rejection)
  3. Normal / Clear Skin Assessment
  4. Non-Face Images (Wallpapers, textures, abstract patterns)
  5. Group Photos (Multiple faces)
  6. Blurry Images (Gaussian blur threshold validation)
  7. Dark / Underexposed Images (Low luminance validation)
  8. Overexposed / Washed-out Images (High luminance validation)
  9. Corrupted / 0-byte Image Files

Author: MCA Project
"""

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from predict import predict_single_image, validate_skin_image

def run_suite():
    print("=" * 75)
    print("  DERMSCAN AI — AUTOMATED REJECTION & ACCEPTANCE VALIDATION SUITE")
    print("=" * 75)

    test_results = []

    def log_result(test_category, test_name, expected, actual_status, actual_detail, passed, skipped=False):
        if skipped:
            mark = "⊘ SKIP"
        else:
            mark = "✓ PASS" if passed else "✗ FAIL"
        print(f"[{mark}] {test_category:<22} | {test_name:<22} | Expected: {expected:<14} | Got: {actual_status}")
        test_results.append({
            "category": test_category,
            "test_name": test_name,
            "expected": expected,
            "status": actual_status,
            "detail": actual_detail,
            "passed": passed,
            "skipped": skipped,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: Supported Disease Classes
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 1. Testing Supported Disease Classes ---")
    acne_imgs = list(PROJECT_ROOT.glob("dataset/Acne/**/*.jpg"))
    seb_imgs  = list(PROJECT_ROOT.glob("dataset/Seborrheic_Dermatitis/*.jpg"))
    ros_imgs  = list(PROJECT_ROOT.glob("dataset/Rosacea/**/*.jpg"))
    mel_imgs  = list(PROJECT_ROOT.glob("dataset/Melasma/**/*.jpg"))

    if acne_imgs:
        res = predict_single_image(acne_imgs[0])
        log_result(
            "Supported Disease", "Acne Sample", "ACCEPTED",
            "ACCEPTED" if not res.get("is_rejected") and res.get("predicted_class") == "Acne" else "REJECTED",
            f"Class: {res.get('predicted_class')}, Conf: {res.get('confidence')}%",
            not res.get("is_rejected") and res.get("predicted_class") == "Acne"
        )
    else:
        log_result("Supported Disease", "Acne Sample", "ACCEPTED", "UNAVAILABLE", "No Acne images found in dataset", False, skipped=True)

    if seb_imgs:
        res = predict_single_image(seb_imgs[0])
        log_result(
            "Supported Disease", "Seborrheic Derm", "ACCEPTED",
            "ACCEPTED" if not res.get("is_rejected") and res.get("predicted_class") == "Seborrheic Dermatitis" else "REJECTED",
            f"Class: {res.get('predicted_class')}, Conf: {res.get('confidence')}%",
            not res.get("is_rejected") and res.get("predicted_class") == "Seborrheic Dermatitis"
        )
    else:
        log_result("Supported Disease", "Seborrheic Derm", "ACCEPTED", "UNAVAILABLE", "No Seborrheic Dermatitis images found in dataset", False, skipped=True)

    if ros_imgs:
        res = predict_single_image(ros_imgs[0])
        log_result(
            "Supported Disease", "Rosacea Sample", "ACCEPTED",
            "ACCEPTED" if not res.get("is_rejected") and res.get("predicted_class") == "Rosacea" else "REJECTED",
            f"Class: {res.get('predicted_class')}, Conf: {res.get('confidence')}%",
            not res.get("is_rejected") and res.get("predicted_class") == "Rosacea"
        )
    else:
        log_result("Supported Disease", "Rosacea Sample", "ACCEPTED", "UNAVAILABLE", "No Rosacea images found in dataset", False, skipped=True)

    if len(mel_imgs) > 1:
        res = predict_single_image(mel_imgs[1])
        log_result(
            "Supported Disease", "Melasma Sample", "ACCEPTED",
            "ACCEPTED" if not res.get("is_rejected") and res.get("predicted_class") == "Melasma" else "REJECTED",
            f"Class: {res.get('predicted_class')}, Conf: {res.get('confidence')}%",
            not res.get("is_rejected") and res.get("predicted_class") == "Melasma"
        )
    else:
        log_result("Supported Disease", "Melasma Sample", "ACCEPTED", "UNAVAILABLE", "No Melasma images found in dataset", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: Normal / Healthy Skin (Normal_Facial_Skin dataset)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 2. Testing Normal / Clear Skin ---")
    normal_faces = [
        (PROJECT_ROOT / "dataset" / "Normal_Facial_Skin" / "IMG_20240920_105902.jpg", "Normal Skin Sample 1"),
        (PROJECT_ROOT / "dataset" / "Normal_Facial_Skin" / "IMG-20221125-WA0006.jpg", "Normal Skin Sample 2"),
    ]
    for p, desc in normal_faces:
        if p.exists():
            res = predict_single_image(p)
            is_ok = res.get("is_clear_skin") and res.get("predicted_class") == "Normal Facial Skin"
            log_result(
                "Normal Skin", desc, "NORMAL_SKIN",
                "NORMAL_SKIN" if is_ok else res.get("predicted_class"),
                res.get("message") or res.get("predicted_class"),
                is_ok
            )
        else:
            log_result("Normal Skin", desc, "NORMAL_SKIN", "UNAVAILABLE", f"File not found: {p}", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: Unsupported / Unknown Facial Conditions (Unknown_Facial_Conditions)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 3. Testing Unsupported & Unknown Skin Conditions ---")
    ood_tests = [
        ("models/test_sz.jpg", "Atypical Skin/Lesion"),
        ("test_normal_skin.jpg", "Untrained Skin Texture"),
        ("dataset/Unknown_Facial_Conditions/03Eczema0915.jpeg", "Unknown Condition (Eczema)"),
        ("dataset/Unknown_Facial_Conditions/126.jpg", "Unknown Condition (126)"),
    ]
    for rel_path, desc in ood_tests:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            res = predict_single_image(p)
            is_ok = res.get("is_rejected", False)
            log_result(
                "Unknown Condition", desc, "REJECTED",
                "REJECTED" if is_ok else (res.get("predicted_class") or "ACCEPTED"),
                res.get("title") or res.get("message") or res.get("predicted_class"),
                is_ok
            )
        else:
            log_result("Unknown Condition", desc, "UNKNOWN_COND", "UNAVAILABLE", f"File not found: {p}", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4: Invalid Images Folder (Buildings, Non-faces, Group photos)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 4. Testing Invalid Images Folder ---")
    invalid_tests = [
        ("dataset/Invalid images/Screenshot 2026-09-15 204154.png", "Architecture (Screenshot)"),
        ("dataset/Invalid images/me.jpg", "Invalid User Photo"),
        ("dataset/Invalid images/IMG-20210402-WA0003.jpg", "Invalid Group/Face Photo"),
        ("fractal_test.jpg", "Fractal Pattern"),
        ("models/test_temple.jpg", "Temple Architecture"),
    ]
    for rel_path, desc in invalid_tests:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            res = predict_single_image(p)
            is_invalid = res.get("is_rejected", False) and (res.get("title") == "Invalid Image" or res.get("error_code") == "INVALID_IMAGE")
            log_result(
                "Invalid Image", desc, "INVALID_IMAGE",
                "INVALID_IMAGE" if is_invalid else "ACCEPTED",
                res.get("title") or res.get("message") or res.get("error"),
                is_invalid
            )
        else:
            log_result("Invalid Image", desc, "INVALID_IMAGE", "UNAVAILABLE", f"File not found: {p}", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5: Blurry Images
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 5. Testing Blurry Images ---")
    if acne_imgs:
        base_img = cv2.imread(str(acne_imgs[0]))
        blurry_img = cv2.GaussianBlur(base_img, (31, 31), 12)
        tmp_blur = PROJECT_ROOT / "temp_test_blurry.jpg"
        cv2.imwrite(str(tmp_blur), blurry_img)

        res = predict_single_image(tmp_blur)
        log_result(
            "Quality: Blurry", "Gaussian Blurred Face", "REJECTED",
            "REJECTED" if res.get("is_rejected") else "ACCEPTED",
            res.get("message") or res.get("error"),
            res.get("is_rejected", False) and res.get("error_code") == "IMAGE_TOO_BLURRY"
        )
        tmp_blur.unlink(missing_ok=True)
    else:
        log_result("Quality: Blurry", "Gaussian Blurred Face", "REJECTED", "UNAVAILABLE", "Base image unavailable", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 6: Dark / Underexposed Images
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 6. Testing Dark / Underexposed Images ---")
    if acne_imgs:
        base_img = cv2.imread(str(acne_imgs[0]))
        dark_img = (base_img * 0.08).astype(np.uint8)
        tmp_dark = PROJECT_ROOT / "temp_test_dark.jpg"
        cv2.imwrite(str(tmp_dark), dark_img)

        res = predict_single_image(tmp_dark)
        log_result(
            "Quality: Dark", "Underexposed Face", "REJECTED",
            "REJECTED" if res.get("is_rejected") else "ACCEPTED",
            res.get("message") or res.get("error"),
            res.get("is_rejected", False) and res.get("error_code") == "IMAGE_TOO_DARK"
        )
        tmp_dark.unlink(missing_ok=True)
    else:
        log_result("Quality: Dark", "Underexposed Face", "REJECTED", "UNAVAILABLE", "Base image unavailable", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 7: Overexposed / Washed-out Images
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 7. Testing Overexposed Images ---")
    if acne_imgs:
        base_img = cv2.imread(str(acne_imgs[0]))
        bright_img = np.clip(base_img.astype(float) + 180, 0, 255).astype(np.uint8)
        tmp_bright = PROJECT_ROOT / "temp_test_bright.jpg"
        cv2.imwrite(str(tmp_bright), bright_img)

        res = predict_single_image(tmp_bright)
        log_result(
            "Quality: Overexposed", "Washed-out Face", "REJECTED",
            "REJECTED" if res.get("is_rejected") else "ACCEPTED",
            res.get("message") or res.get("error"),
            res.get("is_rejected", False) and res.get("error_code") == "IMAGE_TOO_BRIGHT"
        )
        tmp_bright.unlink(missing_ok=True)
    else:
        log_result("Quality: Overexposed", "Washed-out Face", "REJECTED", "UNAVAILABLE", "Base image unavailable", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 8: Group Photo (Multiple Faces)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 8. Testing Group Photo ---")
    if len(acne_imgs) > 0 and len(ros_imgs) > 0:
        img1 = cv2.imread(str(acne_imgs[0]))
        img2 = cv2.imread(str(ros_imgs[0]))
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        combined = np.hstack([cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))])
        tmp_group = PROJECT_ROOT / "temp_test_group.jpg"
        cv2.imwrite(str(tmp_group), combined)

        res = predict_single_image(tmp_group)
        log_result(
            "Face Gate: Group", "Dual Face Photo", "REJECTED",
            "REJECTED" if res.get("is_rejected") else "ACCEPTED",
            res.get("message") or res.get("error"),
            res.get("is_rejected", False) and res.get("error_code") in ("MULTIPLE_FACES_DETECTED", "OUTSIDE_SUPPORTED_CATEGORIES")
        )
        tmp_group.unlink(missing_ok=True)
    else:
        log_result("Face Gate: Group", "Dual Face Photo", "REJECTED", "UNAVAILABLE", "Base images unavailable", False, skipped=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 9: Corrupted Image File
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 9. Testing Corrupted File ---")
    tmp_corrupt = PROJECT_ROOT / "temp_test_corrupt.jpg"
    with open(tmp_corrupt, "wb") as f:
        f.write(b"not an image header at all 1234567890")

    res = predict_single_image(tmp_corrupt)
    log_result(
        "Integrity: Corrupted", "Truncated Binary", "REJECTED",
        "REJECTED" if res.get("is_rejected") else "ACCEPTED",
        res.get("message") or res.get("error"),
        res.get("is_rejected", False) and res.get("error_code") in ("CORRUPT_IMAGE", "CANNOT_READ")
    )
    tmp_corrupt.unlink(missing_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 10: 0-Byte Empty File
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 10. Testing Empty (0-Byte) File ---")
    tmp_empty = PROJECT_ROOT / "temp_test_empty.jpg"
    with open(tmp_empty, "wb") as f:
        pass

    res = predict_single_image(tmp_empty)
    log_result(
        "Integrity: Empty", "0-Byte File", "REJECTED",
        "REJECTED" if res.get("is_rejected") else "ACCEPTED",
        res.get("message") or res.get("error"),
        res.get("is_rejected", False) and res.get("error_code") in ("CORRUPT_IMAGE", "CANNOT_READ", "EMPTY_FILE")
    )
    tmp_empty.unlink(missing_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 11: Missing Model / File Verification
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 11. Testing Missing File Error Handling ---")
    non_existent_file = PROJECT_ROOT / "non_existent_image_12345.jpg"
    res = predict_single_image(non_existent_file)
    log_result(
        "Integrity: Missing File", "Non-existent File", "REJECTED",
        "REJECTED" if res.get("is_rejected") else "ACCEPTED",
        res.get("message") or res.get("error"),
        res.get("is_rejected", False) and res.get("error_code") == "FILE_NOT_FOUND"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    total_tests = len(test_results)
    skipped_tests = sum(1 for t in test_results if t.get("skipped", False))
    passed_tests = sum(1 for t in test_results if t["passed"] and not t.get("skipped", False))
    failed_tests = total_tests - passed_tests - skipped_tests
    print(f"  TOTAL TESTS: {total_tests} | PASSED: {passed_tests} | FAILED: {failed_tests} | SKIPPED: {skipped_tests}")
    print("=" * 75)

    return test_results

if __name__ == "__main__":
    run_suite()
