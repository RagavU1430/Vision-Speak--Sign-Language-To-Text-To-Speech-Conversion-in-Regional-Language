"""
VisionSpeak — MLP Optimization Pipeline
========================================
Phase 3: Hyperparameter tuning + cross-validation
Phase 5: Targeted data expansion for collision classes
Phase 7: Retrain with enhanced features, compare vs original

Usage:
  python optimize_mlp.py                      # full pipeline with defaults
  python optimize_mlp.py --csv dataset/landmarks_v1.csv --tune
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import json
import argparse
import joblib

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import extract_enhanced_features, extract_enhanced_features_v2, suppress_c_stderr

DATASET_DIR = os.path.join("archive", "asl_alphabet_train", "asl_alphabet_train")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Augmentation (reuses logic from train_mlp.py) ──────────────────────────

def augment_landmarks_and_recompute(coords, use_v2=False):
    angle = np.random.uniform(-15, 15) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_mat = np.array([
        [cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]
    ], dtype=np.float32)
    rotated = np.dot(coords, rot_mat.T)
    scale = np.random.uniform(0.95, 1.05)
    scaled = rotated * scale
    noise = np.random.normal(0, 0.005, size=coords.shape).astype(np.float32)
    jittered = scaled + noise
    fn = extract_enhanced_features_v2 if use_v2 else extract_enhanced_features
    return fn(jittered)


def augment_dataset(X_raw, y_raw, factor=3, use_v2=False):
    X_aug, y_aug = [], []
    for i in range(len(X_raw)):
        sample = X_raw[i]
        label = y_raw[i]
        X_aug.append(sample)
        y_aug.append(label)
        coords = sample[:63].reshape(21, 3)
        for _ in range(factor):
            X_aug.append(augment_landmarks_and_recompute(coords, use_v2))
            y_aug.append(label)
    return np.array(X_aug, dtype=np.float32), np.array(y_aug)


# ── CSV loading ────────────────────────────────────────────────────────────

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y, feature_cols


# ── PHASE 3: Hyperparameter tuning ────────────────────────────────────────

def tune_hyperparameters(X_train, y_train):
    print("\n" + "=" * 60)
    print("  PHASE 3 — Hyperparameter Tuning (GridSearchCV)")
    print("=" * 60)

    mlp = MLPClassifier(
        activation="relu", solver="adam", batch_size=128,
        max_iter=200, early_stopping=True, validation_fraction=0.1,
        random_state=42, n_iter_no_change=10, verbose=False,
    )

    param_grid = {
        "hidden_layer_sizes": [
            (256, 128),
            (512, 256),
            (512, 256, 128),
            (512, 256, 128, 64),
        ],
        "learning_rate_init": [0.001, 0.0005],
        "alpha": [0.0001, 0.001],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        mlp, param_grid, cv=cv, scoring="accuracy",
        n_jobs=1, verbose=2, refit=False,
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV acc : {grid.best_score_ * 100:.2f}%")
    print(f"\n  Top-5 results:")
    results = pd.DataFrame(grid.cv_results_)
    top = results.sort_values("rank_test_score").head(5)
    for _, r in top.iterrows():
        print(f"    {r['params']}  →  {r['mean_test_score'] * 100:.2f}%")

    return grid.best_params_


# ── Training with given params ─────────────────────────────────────────────

def train_mlp(X_train, y_train, X_test, y_test, hidden_layers,
              learning_rate=0.001, alpha=0.0001, label=""):

    print(f"\n  Training {label}...")
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu", solver="adam",
        learning_rate_init=learning_rate,
        alpha=alpha,
        batch_size=128, max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        random_state=42, verbose=False,
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100.0
    print(f"  {label} test accuracy: {acc:.2f}%")
    return mlp, acc


# ── Confusion matrix plot ─────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.BuGn)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=9, color="white")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9, color="white")
    ax.set_xlabel("Predicted", fontsize=13, color="white", labelpad=10)
    ax.set_ylabel("True", fontsize=13, color="white", labelpad=10)
    ax.set_title(title, fontsize=16, color="white", pad=18, fontweight="bold")
    ax.tick_params(axis="both", colors="white")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ── PHASE 5: Targeted expansion for collision classes ─────────────────────

def identify_collision_classes(X_test, y_test, le):
    """Run a quick model to find which classes confuse most."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_test)
    probe = MLPClassifier(
        (256, 128), activation="relu", solver="adam",
        max_iter=100, random_state=42, verbose=False,
    )
    probe.fit(Xs, y_test)
    y_pred = probe.predict(Xs)

    cm = confusion_matrix(y_test, y_pred)
    class_names = le.classes_

    collision_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j:
                continue
            total = cm[i, j] + cm[j, i]
            if total > 0:
                collision_pairs.append((class_names[i], class_names[j], total))

    collision_pairs.sort(key=lambda x: -x[2])
    top = collision_pairs[:10]
    print(f"\n  Top collision pairs (probe):")
    for a, b, c in top:
        print(f"    {a} ↔ {b}: {c} errors")

    # Classes that need expansion
    collide_classes = set()
    for a, b, c in top:
        collide_classes.add(a)
        collide_classes.add(b)
    return sorted(collide_classes)


def expand_collision_classes(collide_classes, csv_path, output_path, extra_per_class=2000):
    """
    For collision classes only, extract additional samples from the raw dataset
    beyond what's already in the CSV. Append new rows to output CSV.
    """
    print("\n" + "=" * 60)
    print("  PHASE 5 — Targeted Data Expansion")
    print("=" * 60)
    print(f"  Collision classes : {collide_classes}")
    print(f"  Extra per class   : {extra_per_class}")

    existing = pd.read_csv(csv_path)
    existing_counts = existing["label"].value_counts().to_dict()

    with suppress_c_stderr():
        import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    new_rows = []
    for label in collide_classes:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        files = sorted([
            fn for fn in os.listdir(folder)
            if fn.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        already_have = existing_counts.get(label, 0)
        # Pick files starting after what we already have
        extra_needed = min(extra_per_class, len(files) - already_have)
        if extra_needed <= 0:
            print(f"  {label}: already have {already_have}, no extra needed")
            continue
        selected = files[already_have: already_have + extra_needed]
        found = 0
        for fn in tqdm(selected, desc=f"  {label} +{extra_needed}", unit="img", leave=False):
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                features = extract_enhanced_features(hand)
                new_rows.append(list(features) + [label])
                found += 1
        print(f"  {label}: added {found} (had {already_have})")

    hands.close()

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=existing.columns)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"\n  Total before: {len(existing)}, added: {len(new_df)}, total: {len(combined)}")
        print(f"  Saved → {output_path}")
    else:
        print(f"  No new rows added.")
    return output_path


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VisionSpeak MLP Optimization")
    parser.add_argument("--csv", default=os.path.join("dataset", "landmarks_v1.csv"), help="Input landmarks CSV")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--expand", action="store_true", help="Run targeted expansion")
    parser.add_argument("--epochs", type=int, default=300, help="Max training iterations")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    X, y, feature_cols = load_csv(args.csv)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_features = X.shape[1]
    print(f"\n  Loaded {len(X)} samples, {num_features} features, {len(le.classes_)} classes")

    # Determine if this is v1 (99 features) or v2 (136 features)
    use_v2 = (num_features > 99)
    print(f"  Feature set: {'v2 (136)' if use_v2 else 'v1 (99)'}")

    # ── Split ────────────────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"  Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # ── PHASE 3: Hyperparameter tuning ──────────────────────────────────
    if args.tune:
        scaler_tune = StandardScaler()
        X_tune = scaler_tune.fit_transform(X_train_raw)
        best_params = tune_hyperparameters(X_tune, y_train)
        hidden = best_params["hidden_layer_sizes"]
        lr = best_params["learning_rate_init"]
        alpha = best_params["alpha"]
    else:
        hidden = (512, 256, 128)
        lr = 0.001
        alpha = 0.0001
        best_params = {"hidden_layer_sizes": hidden, "learning_rate_init": lr, "alpha": alpha}
        print(f"\n  Using default params: {best_params}")

    # ── Augment + scale + train ─────────────────────────────────────────
    print(f"\n  Augmenting training set x4 (factor=3)...")
    X_aug, y_aug = augment_dataset(X_train_raw, y_train, factor=3, use_v2=use_v2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_aug)
    X_test = scaler.transform(X_test_raw)

    model, acc = train_mlp(
        X_train, y_train, X_test, y_test,
        hidden_layers=hidden, learning_rate=lr, alpha=alpha,
        label="MLP",
    )

    # ── Report ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix_optimized.png")
    plot_confusion_matrix(cm, le.classes_, cm_path,
                          f"Optimized MLP — {acc:.2f}%")

    # ── Save artefacts ──────────────────────────────────────────────────
    suffix = "_v2" if use_v2 else "_v1"
    model_path = os.path.join(MODEL_DIR, f"mlp_model{suffix}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler{suffix}.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    print(f"\n  Model   → {model_path}")
    print(f"  Scaler  → {scaler_path}")
    print(f"  Encoder → {encoder_path}")

    # ── Save comparison report ──────────────────────────────────────────
    report = {
        "csv": args.csv,
        "samples": len(X),
        "features": num_features,
        "feature_set": "v2" if use_v2 else "v1",
        "best_params": {str(k): str(v) for k, v in best_params.items()},
        "test_accuracy_pct": round(acc, 2),
        "tuned": args.tune,
    }
    report_path = os.path.join(MODEL_DIR, f"training_report{suffix}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report  → {report_path}")

    # ── PHASE 5: Targeted expansion ────────────────────────────────────
    if args.expand:
        collide_classes = identify_collision_classes(X_test_raw, y_test, le)
        csv_v2 = args.csv.replace(".csv", "_expanded.csv")
        expand_collision_classes(collide_classes, args.csv, csv_v2, extra_per_class=2000)

    print("\n" + "=" * 60)
    print(f"  DONE — Accuracy: {acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
