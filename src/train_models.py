"""
============================================================
Facial Skin Condition Detection
Phase 4 — Model Training
============================================================

Trains 7 traditional ML classifiers on the extracted features:
  1. Support Vector Machine (SVM)
  2. Random Forest
  3. K-Nearest Neighbours (KNN)
  4. Decision Tree
  5. Logistic Regression
  6. Naive Bayes (Gaussian)
  7. XGBoost

Pipeline
--------
1. Load features/features.csv
2. Encode labels
3. Split into 80% train / 20% test (stratified)
4. Scale features with StandardScaler
5. Train + 5-fold stratified cross-validate each model
6. Identify best model by CV accuracy
7. Retrain best model on full training set
8. Save: models/best_model.pkl, models/scaler.pkl,
         models/label_encoder.pkl, models/model_results.pkl

Run standalone:
    python src/train_models.py

Author: MCA Project
"""

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    print("[WARNING] xgboost not installed — XGBoost will be skipped.")

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
FEATURES_DIR  = PROJECT_ROOT / "features"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"

FEATURES_CSV  = FEATURES_DIR / "features.csv"

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5


# ── Model Definitions ─────────────────────────────────────────────────────────

def _build_models() -> dict:
    """Return a dict of {model_name: estimator} for all 7 classifiers."""
    models = {
        "SVM": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
    }
    if _XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


# ── Training Pipeline ─────────────────────────────────────────────────────────

def run_training() -> dict:
    """
    Full training pipeline.

    Returns
    -------
    dict with keys:
        model_results   — per-model performance metrics
        best_model_name — name of the best-performing model
        X_test          — held-out feature matrix
        y_test          — held-out true labels
        label_encoder   — fitted LabelEncoder
        scaler          — fitted StandardScaler
        best_model      — trained best model
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING — 7 Traditional ML Classifiers")
    print("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────
    if not FEATURES_CSV.exists():
        print(f"\n[ERROR] Feature file not found: {FEATURES_CSV}")
        print("        Run python src/feature_extraction.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    print(f"\n  Loaded {len(df)} samples from features.csv")
    print(f"  Classes: {df['label'].value_counts().to_dict()}")

    if len(df) < 10:
        print("\n[ERROR] Not enough samples for training (< 10).")
        sys.exit(1)

    # ── Prepare X and y ───────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in ("label", "filepath")]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    print(f"\n  Features : {X.shape[1]}")
    print(f"  Samples  : {X.shape[0]}")
    print(f"  Classes  : {list(le.classes_)}")

    # ── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")

    # ── Scale features ────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Cross-validate & train all models ─────────────────────────────────
    models       = _build_models()
    skf          = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_results = {}

    print(f"\n  Training {len(models)} models with {CV_FOLDS}-fold cross-validation …\n")
    print(f"  {'Model':<22} {'CV Acc (mean)':<16} {'CV Acc (std)':<14} {'Test Acc'}")
    print("  " + "-" * 66)

    sample_weights = compute_sample_weight("balanced", y_train)

    best_name   = None
    best_cv_acc = -1.0

    for name, clf in tqdm(models.items(), desc="  Training", unit="model"):
        # Cross-validation on training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=skf,
                                    scoring="accuracy", n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()

        # Final train on full train set (use sample weights for XGBoost to balance classes)
        if name == "XGBoost":
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)

        # Test accuracy
        y_pred     = clf.predict(X_test)
        test_acc   = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred,
                                           target_names=le.classes_,
                                           output_dict=True,
                                           zero_division=0)

        model_results[name] = {
            "model":          clf,
            "cv_mean":        float(cv_mean),
            "cv_std":         float(cv_std),
            "test_accuracy":  float(test_acc),
            "y_pred":         y_pred,
            "clf_report":     clf_report,
        }

        print(f"  {name:<22} {cv_mean:.4f} ± {cv_std:.4f}    {test_acc:.4f}")

        if cv_mean > best_cv_acc:
            best_cv_acc = cv_mean
            best_name   = name

    # ── Save artefacts ────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_model = model_results[best_name]["model"]
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler,     MODELS_DIR / "scaler.pkl")
    joblib.dump(le,         MODELS_DIR / "label_encoder.pkl")

    # Save SVM model specifically for DermScan prediction pipeline
    if "SVM" in model_results:
        joblib.dump(model_results["SVM"]["model"], MODELS_DIR / "svm_model.pkl")

    # Fit and save NearestNeighbors on scaled training features for open-set rejection
    from sklearn.neighbors import NearestNeighbors
    dataset_nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    dataset_nn.fit(X_train)
    joblib.dump(dataset_nn, MODELS_DIR / "dataset_nn.pkl")

    # Save full results for evaluate_models.py
    results_payload = {
        "model_results":   model_results,
        "best_model_name": best_name,
        "X_test":          X_test,
        "y_test":          y_test,
        "label_encoder":   le,
        "scaler":          scaler,
        "best_model":      best_model,
        "feature_cols":    feature_cols,
    }
    with open(MODELS_DIR / "model_results.pkl", "wb") as f:
        pickle.dump(results_payload, f)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  ✓ Best model : {best_name}")
    print(f"    CV Accuracy: {best_cv_acc:.4f}")
    print(f"    Test Acc   : {model_results[best_name]['test_accuracy']:.4f}")
    print(f"\n  Saved:")
    print(f"    → models/best_model.pkl")
    print(f"    → models/scaler.pkl")
    print(f"    → models/label_encoder.pkl")
    print(f"    → models/model_results.pkl")
    print(f"\n  Next step → python src/evaluate_models.py")
    print("=" * 60)

    return results_payload


if __name__ == "__main__":
    run_training()
