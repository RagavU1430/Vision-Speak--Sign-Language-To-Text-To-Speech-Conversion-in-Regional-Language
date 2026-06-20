"""
Sign Language A-Z Recognition -- Deep MLP Training Pipeline (Keras)
===================================================================
Upgraded pipeline that:
  1. Scans the raw dataset (detects corrupt images, counts per class)
  2. Extracts 21 hand landmarks ? 99 enhanced features using MediaPipe
  3. Trains a deep Keras MLP with BatchNormalization & Dropout
  4. Evaluates on test set with full metrics + collision analysis
  5. Produces a ranked collision report with weak-class recommendations

Architecture:
  Input(auto)
  ? Dense(256, ReLU) ? BatchNorm ? Dropout(0.3)
  ? Dense(128, ReLU) ? BatchNorm ? Dropout(0.3)
  ? Dense(64,  ReLU) ? BatchNorm ? Dropout(0.2)
  ? Dense(32,  ReLU)
  ? Dense(num_classes, Softmax)
"""

# -- Suppress all warnings BEFORE any other imports --------------------------
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# -- Standard / third-party imports ------------------------------------------
import csv
import json
import string
import joblib
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from utils import extract_enhanced_features, KerasMLPWrapper

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -- Paths relative to project root ------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "archive", "asl_alphabet_train", "asl_alphabet_train")
CSV_PATH = os.path.join(ROOT_DIR, "dataset", "extracted_landmarks.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pkl")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.keras")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model_best.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CM_NORM_PATH = os.path.join(MODEL_DIR, "confusion_matrix_normalized.png")
TRAINING_CURVES_PATH = os.path.join(MODEL_DIR, "training_curves.png")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
PER_CLASS_ACC_PATH = os.path.join(MODEL_DIR, "per_class_accuracy.png")
COLLISION_BAR_PATH = os.path.join(MODEL_DIR, "collision_pairs.png")
COLLISION_REPORT_PATH = os.path.join(MODEL_DIR, "collision_report.json")
DATASET_STATS_PATH = os.path.join(MODEL_DIR, "dataset_stats.json")

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
LABELS = list(string.ascii_uppercase)
LIMIT_SAMPLES_PER_GESTURE = None
BATCH_SIZE = 32
EPOCHS = 5
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)



# =============================================================================
#  SECTION 1 -- Dataset Analysis
# =============================================================================
def analyze_raw_dataset():
    """
    Scan the raw image dataset directory.
    Counts total images, per-class distribution, detects corrupt/missing images.
    Removes corrupt images automatically.
    Returns dict with full statistics.
    """
    print("\n" + "=" * 60)
    print("  DATASET ANALYSIS -- Raw Images")
    print("=" * 60)

    per_class = {}
    corrupt_images = []
    missing_labels = []
    total_valid = 0
    all_files_count = 0

    for label in sorted(LABELS):
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            print(f"  [WARN] Folder not found for label '{label}'")
            missing_labels.append(label)
            per_class[label] = 0
            continue

        files = [
            fn for fn in os.listdir(folder)
            if fn.lower().endswith(VALID_EXTENSIONS)
        ]
        all_files_count += len(files)
        count = 0
        class_corrupt = []
        for fn in files:
            img_path = os.path.join(folder, fn)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    class_corrupt.append((img_path, "cv2.imread returned None"))
                    continue
                if img.size == 0:
                    class_corrupt.append((img_path, "Empty image"))
                    continue
                count += 1
            except Exception as e:
                class_corrupt.append((img_path, str(e)))

        per_class[label] = count
        total_valid += count
        corrupt_images.extend(class_corrupt)
        flag = " [CORRUPT REMOVED]" if class_corrupt else ""
        print(f"  {label}: {count:4d} / {len(files):4d} valid{flag}")

    valid_classes = {k: v for k, v in per_class.items() if v > 0}
    min_label = min(valid_classes, key=valid_classes.get) if valid_classes else None
    max_label = max(valid_classes, key=valid_classes.get) if valid_classes else None
    min_count = valid_classes.get(min_label, 0) if min_label else 0
    max_count = valid_classes.get(max_label, 0) if max_label else 0
    balance_ratio = max_count / max(min_count, 1)

    print(f"\n  {'-' * 40}")
    print(f"  Total files on disk : {all_files_count}")
    print(f"  Valid images        : {total_valid}")
    print(f"  Classes             : {len(valid_classes)} / {len(LABELS)}")
    print(f"  Corrupted removed   : {len(corrupt_images)}")
    print(f"  Missing dirs        : {len(missing_labels)}")
    print(f"  Smallest class      : {min_label} ({min_count})")
    print(f"  Largest class       : {max_label} ({max_count})")
    print(f"  Balance ratio       : {balance_ratio:.2f}x")

    if corrupt_images:
        print(f"\n  Removing {len(corrupt_images)} corrupt image(s)...")
        removed = 0
        for img_path, err in corrupt_images:
            try:
                os.remove(img_path)
                removed += 1
            except Exception as e:
                print(f"    Could not remove {os.path.basename(img_path)}: {e}")
        print(f"  Removed {removed} corrupt image(s)")

    stats = {
        "total_files_on_disk": all_files_count,
        "total_valid_images": total_valid,
        "per_class": per_class,
        "corrupt_removed": len(corrupt_images),
        "missing_labels": missing_labels,
        "min_class": (min_label, min_count),
        "max_class": (max_label, max_count),
        "class_balance_ratio": balance_ratio,
    }
    return stats


# =============================================================================
#  SECTION 2 -- Landmark Extraction
# =============================================================================
def extract_landmarks() -> str:
    """
    Walk through each A-Z subfolder, run MediaPipe Hands on every image,
    skip corrupted/invalid images, and write the 99-feature vector + label
    to a CSV file.

    Returns the path to the saved CSV.
    """
    print("\n" + "=" * 60)
    print("  STEP 1 -- Extracting Hand Landmarks with MediaPipe")
    print("=" * 60)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    header = [f"f{i}" for i in range(99)]
    header.append("label")

    total_images = 0
    processed = 0
    skipped_corrupt = 0
    skipped_no_hand = 0
    skipped_dim = 0

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label in sorted(LABELS):
            folder = os.path.join(DATASET_DIR, label)
            if not os.path.isdir(folder):
                print(f"  [WARN] Folder not found for label '{label}', skipping.")
                continue

            files = sorted([
                fn for fn in os.listdir(folder)
                if fn.lower().endswith(VALID_EXTENSIONS)
            ])
            if LIMIT_SAMPLES_PER_GESTURE is not None:
                files = files[:LIMIT_SAMPLES_PER_GESTURE]
            total_images += len(files)

            for fn in tqdm(files, desc=f"  {label}", unit="img", leave=False):
                img_path = os.path.join(folder, fn)
                try:
                    img = cv2.imread(img_path)
                    if img is None or img.size == 0:
                        skipped_corrupt += 1
                        continue
                except Exception:
                    skipped_corrupt += 1
                    continue

                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    skipped_corrupt += 1
                    continue

                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    try:
                        row = list(extract_enhanced_features(hand))
                        if len(row) != 99:
                            skipped_dim += 1
                            continue
                        row.append(label)
                        writer.writerow(row)
                        processed += 1
                    except Exception:
                        skipped_no_hand += 1
                else:
                    skipped_no_hand += 1

    hands.close()

    print(f"\n  {'-' * 40}")
    print(f"  Total images scanned : {total_images}")
    print(f"  Successfully extracted: {processed}")
    print(f"  Skipped (corrupt)    : {skipped_corrupt}")
    print(f"  Skipped (no hand)    : {skipped_no_hand}")
    print(f"  Skipped (bad dim)    : {skipped_dim}")
    print(f"  Feature dimension    : 99")
    print(f"  Output CSV           : {CSV_PATH}")
    return CSV_PATH


# =============================================================================
#  SECTION 3 -- Feature Augmentation
# =============================================================================
def augment_landmarks_and_recompute(coords):
    angle = np.random.uniform(-15, 15) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    rotated = np.dot(coords, rot_mat.T)

    scale = np.random.uniform(0.95, 1.05)
    scaled = rotated * scale

    noise = np.random.normal(0, 0.005, size=coords.shape).astype(np.float32)
    jittered = scaled + noise

    return extract_enhanced_features(jittered)


def augment_dataset(X_raw, y_raw, factor=3):
    X_aug = []
    y_aug = []
    for i in range(len(X_raw)):
        sample = X_raw[i]
        label = y_raw[i]
        X_aug.append(sample)
        y_aug.append(label)
        coords = sample[:63].reshape(21, 3)
        for _ in range(factor):
            X_aug.append(augment_landmarks_and_recompute(coords))
            y_aug.append(label)
    return np.array(X_aug, dtype=np.float32), np.array(y_aug)


# =============================================================================
#  SECTION 4 -- Analyze Landmark Dataset (from CSV)
# =============================================================================
def analyze_landmark_dataset(df):
    """Print statistics about the loaded landmark CSV."""
    print("\n" + "=" * 60)
    print("  LANDMARK DATASET STATISTICS")
    print("=" * 60)

    feature_cols = [c for c in df.columns if c != "label"]
    num_features = len(feature_cols)
    num_samples = len(df)
    labels = sorted(df["label"].unique())
    per_class = df["label"].value_counts().sort_index()

    print(f"  Total samples       : {num_samples}")
    print(f"  Feature dimensions  : {num_features}")
    print(f"  Number of classes   : {len(labels)}")
    print(f"  Classes             : {''.join(labels)}")
    print(f"\n  Per-class breakdown:")
    min_count = per_class.min()
    max_count = per_class.max()
    for label in labels:
        count = per_class[label]
        bar = "?" * int((count / max_count) * 30)
        print(f"    {label}: {count:4d} {bar}")
    print(f"\n  Min class size      : {per_class.idxmin()} ({min_count})")
    print(f"  Max class size      : {per_class.idxmax()} ({max_count})")
    print(f"  Balance ratio       : {max_count / max(min_count, 1):.2f}x")
    print(f"  Total samples       : {num_samples}")

    return {
        "num_samples": num_samples,
        "num_features": num_features,
        "num_classes": len(labels),
        "classes": list(labels),
        "per_class": per_class.to_dict(),
        "min_class": str(per_class.idxmin()),
        "min_class_count": int(min_count),
        "max_class": str(per_class.idxmax()),
        "max_class_count": int(max_count),
        "balance_ratio": round(max_count / max(min_count, 1), 2),
    }


# =============================================================================
#  SECTION 5 -- MLP Architecture & Training
# =============================================================================
def build_mlp(num_features, num_classes):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(num_features,)),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(csv_path: str):
    """
    Load landmark CSV, split 80/10/10 stratified, augment, scale,
    train deep Keras MLP, persist model + encoder + scaler.
    """
    print("\n" + "=" * 60)
    print("  STEP 2 -- Training Deep Keras MLP")
    print("=" * 60)

    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks

    df = pd.read_csv(csv_path)

    # Analyze and report dataset stats
    dataset_stats = analyze_landmark_dataset(df)

    feature_cols = [c for c in df.columns if c != "label"]
    num_features = len(feature_cols)

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    class_names = le.classes_

    # Stratified 80/10/10 split
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=(TEST_SPLIT + VAL_SPLIT),
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.5,
        random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"\n  Split summary:")
    print(f"    Training   : {len(X_train_raw):5d} ({TRAIN_SPLIT * 100:.0f}%)")
    print(f"    Validation : {len(X_val_raw):5d} ({VAL_SPLIT * 100:.0f}%)")
    print(f"    Testing    : {len(X_test_raw):5d} ({TEST_SPLIT * 100:.0f}%)")

    # Augment training set
    print(f"\n  Augmenting training set (rotation, scaling, noise)...")
    X_train_aug, y_train_aug = augment_dataset(X_train_raw, y_train, factor=3)
    print(f"    Training (augmented): {len(X_train_aug)}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_aug)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Build model
    print(f"\n  Building MLP...")
    print(f"    Input features : {num_features}")
    print(f"    Output classes : {num_classes}")

    model = build_mlp(num_features, num_classes)

    print(f"\n  Model architecture:")
    print(f"  {'-' * 56}")
    model.summary(print_fn=lambda x: print(f"  {x}"))
    print(f"  {'-' * 56}")

    # Callbacks
    cb_early = callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1,
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5,
        min_lr=1e-6, verbose=1,
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        BEST_MODEL_PATH, monitor="val_loss",
        save_best_only=True, verbose=1,
    )

    # Train
    print(f"\n  Training...\n")
    history = model.fit(
        X_train, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb_early, cb_reduce_lr, cb_checkpoint],
        verbose=2,
    )

    # Plot training curves
    _plot_training_curves(history)

    # Save artefacts
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(KERAS_MODEL_PATH)
    print(f"\n  Model saved: {KERAS_MODEL_PATH}")

    keras_model_abs = os.path.abspath(KERAS_MODEL_PATH)
    wrapper = KerasMLPWrapper(keras_model_abs)
    joblib.dump(wrapper, MODEL_PATH)

    joblib.dump(le, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    history_dict = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"  Wrapper saved: {MODEL_PATH}")
    print(f"  Encoder saved: {ENCODER_PATH}")
    print(f"  Scaler saved : {SCALER_PATH}")
    print(f"  History saved: {HISTORY_PATH}")

    # Final training summary
    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    best_val_acc = max(history.history["val_accuracy"])

    print(f"\n  {'-' * 40}")
    print(f"  Final train acc  : {final_acc:.4f}")
    print(f"  Final val acc    : {final_val_acc:.4f}")
    print(f"  Best val acc     : {best_val_acc:.4f}")
    print(f"  Final train loss : {final_loss:.4f}")
    print(f"  Final val loss   : {final_val_loss:.4f}")

    return wrapper, le, scaler, X_test, y_test, history, dataset_stats


# =============================================================================
#  SECTION 6 -- Training Curves
# =============================================================================
def _plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    epochs = range(1, len(history.history["loss"]) + 1)

    ax1.set_facecolor("#1a1a2e")
    ax1.plot(epochs, history.history["loss"], color="#00d4aa", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history.history["val_loss"], color="#ff6b6b", linewidth=2, label="Val Loss")
    ax1.set_title("Training & Validation Loss", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss", color="white")
    ax1.legend(facecolor="#2a2a4e", edgecolor="gray", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, alpha=0.2)

    ax2.set_facecolor("#1a1a2e")
    ax2.plot(epochs, history.history["accuracy"], color="#00d4aa", linewidth=2, label="Train Acc")
    ax2.plot(epochs, history.history["val_accuracy"], color="#ff6b6b", linewidth=2, label="Val Acc")
    ax2.set_title("Training & Validation Accuracy", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Accuracy", color="white")
    ax2.legend(facecolor="#2a2a4e", edgecolor="gray", labelcolor="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Training curves: {TRAINING_CURVES_PATH}")


# =============================================================================
#  SECTION 7 -- Evaluation & Collision Analysis
# =============================================================================
def compute_metrics(y_true, y_pred, class_names):
    """Compute precision, recall, F1, and per-class accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    per_class = {}
    cm = confusion_matrix(y_true, y_pred)
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        prec_i = tp / max(tp + fp, 1)
        rec_i = tp / max(tp + fn, 1)
        f1_i = 2 * prec_i * rec_i / max(prec_i + rec_i, 1e-6)
        per_class[name] = {
            "accuracy": float(rec_i * 100),
            "precision": float(prec_i * 100),
            "recall": float(rec_i * 100),
            "f1": float(f1_i * 100),
            "support": int(cm[i, :].sum()),
        }

    return accuracy, precision, recall, f1, per_class, cm


def rank_collisions(cm, class_names):
    """
    Identify and rank all confused letter pairs.
    Returns sorted list of collision dicts and merged bidirectional pairs.
    """
    n = len(class_names)
    collisions = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total_i = cm[i, :].sum()
            confusion_pct = (cm[i, j] / total_i) * 100.0 if total_i > 0 else 0.0
            if confusion_pct > 0:
                collisions.append({
                    "true_label": str(class_names[i]),
                    "predicted_as": str(class_names[j]),
                    "confusion_pct": round(confusion_pct, 2),
                    "samples": int(cm[i, j]),
                })

    collisions.sort(key=lambda x: x["confusion_pct"], reverse=True)

    merged = {}
    for c in collisions:
        a, b = sorted([c["true_label"], c["predicted_as"]])
        pair_key = f"{a}<->{b}"
        rev_key = f"{b}<->{a}"
        if pair_key not in merged and rev_key not in merged:
            merged[pair_key] = {
                "rank": 0,
                "pair": pair_key,
                "letter_1": a,
                "letter_2": b,
                "total_confusion_pct": 0.0,
                "total_samples": 0,
                "a_as_b_pct": 0.0,
                "b_as_a_pct": 0.0,
                "a_as_b_samples": 0,
                "b_as_a_samples": 0,
            }
        key = pair_key if pair_key in merged else rev_key
        merged[key]["total_confusion_pct"] += c["confusion_pct"]
        merged[key]["total_samples"] += c["samples"]
        if c["true_label"] == a:
            merged[key]["a_as_b_pct"] = c["confusion_pct"]
            merged[key]["a_as_b_samples"] = c["samples"]
        else:
            merged[key]["b_as_a_pct"] = c["confusion_pct"]
            merged[key]["b_as_a_samples"] = c["samples"]

    ranked = sorted(merged.values(), key=lambda x: x["total_confusion_pct"], reverse=True)
    for i, pair in enumerate(ranked):
        pair["rank"] = i + 1

    return collisions, ranked


def plot_confusion_matrix(cm, class_names, path, title="Confusion Matrix"):
    """Save styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"shrink": 0.8},
                linewidths=0.5, linecolor="#2a2a4e",
                annot_kws={"fontsize": 7})

    ax.set_xlabel("Predicted Label", fontsize=14, color="white", labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, color="white", labelpad=10)
    ax.set_title(title, fontsize=16, color="white", pad=18, fontweight="bold")
    ax.tick_params(axis="both", colors="white")
    plt.setp(ax.get_xticklabels(), fontsize=8, color="white")
    plt.setp(ax.get_yticklabels(), fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Confusion matrix: {path}")


def plot_normalized_confusion(cm, class_names, path):
    """Save normalized confusion matrix highlighting errors."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="RdYlGn_r",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"shrink": 0.8},
                linewidths=0.5, linecolor="#2a2a4e",
                annot_kws={"fontsize": 7},
                vmin=0, vmax=1.0)

    ax.set_xlabel("Predicted Label", fontsize=14, color="white", labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, color="white", labelpad=10)
    ax.set_title("Normalized Confusion Matrix (1.0 = Perfect)", fontsize=16, color="white", pad=18, fontweight="bold")
    ax.tick_params(axis="both", colors="white")
    plt.setp(ax.get_xticklabels(), fontsize=8, color="white")
    plt.setp(ax.get_yticklabels(), fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Normalized confusion matrix: {path}")


def plot_per_class_accuracy(per_class, path):
    """Bar chart of per-class accuracy sorted ascending."""
    letters = sorted(per_class.keys())
    accs = [per_class[l]["accuracy"] for l in letters]
    colors = ["#ff6b6b" if a < 85 else "#ffd93d" if a < 95 else "#6bcb77" for a in accs]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(letters, accs, color=colors, edgecolor="white", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, color="white")

    ax.axhline(y=85, color="#ff6b6b", linestyle="--", alpha=0.5, label="85% threshold")
    ax.axhline(y=95, color="#6bcb77", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Letter", fontsize=12, color="white")
    ax.set_ylabel("Accuracy (%)", fontsize=12, color="white")
    ax.set_title("Per-Class Accuracy (Red < 85%, Yellow < 95%, Green >= 95%)",
                 fontsize=14, color="white", fontweight="bold")
    ax.tick_params(axis="both", colors="white")
    ax.legend(loc="lower left", facecolor="#2a2a4e", edgecolor="white", labelcolor="white")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Per-class accuracy: {path}")


def plot_collision_pairs(ranked, path, top_n=10):
    """Horizontal bar chart of top N collision pairs."""
    top = ranked[:top_n]
    if not top:
        # Save a clean placeholder image indicating no collisions
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.text(0.5, 0.5, "No Letter Collisions Detected!",
                ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Collision pairs chart (empty): {path}")
        return

    labels = [c["pair"] for c in top]
    vals = [c["total_confusion_pct"] for c in top]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top)))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(range(len(top)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=10, color="white")
    ax.invert_yaxis()

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="white")

    ax.set_xlabel("Total Confusion (%)", fontsize=12, color="white")
    ax.set_title("Top 10 Letter Collision Pairs", fontsize=14, color="white", fontweight="bold")
    ax.tick_params(axis="both", colors="white")
    ax.set_xlim(0, max(vals) * 1.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Collision pairs chart: {path}")


def generate_recommendations(per_class, ranked_pairs):
    """Identify letters that need more training data."""
    weak = {k: v for k, v in per_class.items() if v["accuracy"] < 85.0}
    moderate = {k: v for k, v in per_class.items() if 85.0 <= v["accuracy"] < 95.0}

    recommendations = []

    # Letters with accuracy < 85%
    for letter in sorted(weak.keys(), key=lambda l: weak[l]["accuracy"]):
        info = weak[letter]
        involved_pairs = [p for p in ranked_pairs if letter in (p["letter_1"], p["letter_2"])]
        top_foe = involved_pairs[0] if involved_pairs else None
        foe = (top_foe["letter_2"] if top_foe["letter_1"] == letter else top_foe["letter_1"]) if top_foe else "?"
        recommendations.append({
            "letter": letter,
            "accuracy": round(info["accuracy"], 1),
            "f1": round(info["f1"], 1),
            "priority": "HIGH",
            "reason": f"Accuracy {info['accuracy']:.1f}% -- confused most with '{foe}'",
            "support": info["support"],
        })

    for letter in sorted(moderate.keys(), key=lambda l: moderate[l]["accuracy"]):
        info = moderate[letter]
        recommendations.append({
            "letter": letter,
            "accuracy": round(info["accuracy"], 1),
            "f1": round(info["f1"], 1),
            "priority": "MEDIUM",
            "reason": f"Accuracy {info['accuracy']:.1f}% -- below 95% threshold",
            "support": info["support"],
        })

    return recommendations


def evaluate_model(model, X_test, y_test, class_names):
    """Full evaluation: metrics, confusion matrix, collision analysis, recommendations."""
    print("\n" + "=" * 60)
    print("  STEP 3 -- Evaluation & Collision Analysis")
    print("=" * 60)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy, precision, recall, f1, per_class, cm = compute_metrics(
        y_test, y_pred, class_names
    )

    print(f"\n  Test metrics:")
    print(f"    Accuracy : {accuracy * 100:.2f}%")
    print(f"    Precision: {precision * 100:.2f}%")
    print(f"    Recall   : {recall * 100:.2f}%")
    print(f"    F1 Score : {f1 * 100:.2f}%")

    print(f"\n  Per-class accuracy:")
    for letter in sorted(per_class.keys(), key=lambda l: per_class[l]["accuracy"]):
        info = per_class[letter]
        marker = "[WARN]" if info["accuracy"] < 85 else "[OK]" if info["accuracy"] >= 95 else "~"
        print(f"    {letter}: {info['accuracy']:5.1f}%  (prec={info['precision']:.1f}%, rec={info['recall']:.1f}%, f1={info['f1']:.1f}%)  {marker}")

    # Collision analysis
    _, ranked_pairs = rank_collisions(cm, class_names)

    print(f"\n  Top 10 collision pairs:")
    print(f"  {'Rank':>4} {'Pair':>8} {'Confusion':>10} {'Samples':>8} {'Direction':>20}")
    print(f"  {'-' * 52}")
    for pair in ranked_pairs[:10]:
        print(f"  {pair['rank']:>4} {pair['pair']:>8} {pair['total_confusion_pct']:>8.1f}%  {pair['total_samples']:>5}  "
              f"{pair['letter_1']}?{pair['letter_2']}({pair['a_as_b_pct']:.0f}%) / {pair['letter_2']}?{pair['letter_1']}({pair['b_as_a_pct']:.0f}%)")

    # Recommendations
    recommendations = generate_recommendations(per_class, ranked_pairs)

    print(f"\n  Letters needing more training data:")
    if recommendations:
        print(f"  {'Letter':>6} {'Accuracy':>9} {'F1':>6} {'Priority':>8} {'Reason':>40}")
        print(f"  {'-' * 72}")
        for rec in recommendations:
            print(f"  {rec['letter']:>6} {rec['accuracy']:>7.1f}% {rec['f1']:>5.1f} {rec['priority']:>8}  {rec['reason']}")
    else:
        print(f"  All classes have >= 95% accuracy -- great!")

    # Plots
    plot_confusion_matrix(cm, class_names, CM_PATH)
    plot_normalized_confusion(cm, class_names, CM_NORM_PATH)
    plot_per_class_accuracy(per_class, PER_CLASS_ACC_PATH)
    plot_collision_pairs(ranked_pairs, COLLISION_BAR_PATH)

    return accuracy, precision, recall, f1, per_class, ranked_pairs, recommendations, cm, y_pred


# =============================================================================
#  MAIN PIPELINE
# =============================================================================
def run_pipeline(reuse_csv=True):
    """
    Full training pipeline:
      1. Scan raw dataset
      2. Extract landmarks (or reuse existing CSV)
      3. Train MLP
      4. Evaluate & generate collision report
      5. Save all artefacts
    """
    # Step 0: Analyze raw dataset
    raw_stats = analyze_raw_dataset()

    # Save raw dataset stats
    with open(DATASET_STATS_PATH, "w") as f:
        json.dump(raw_stats, f, indent=2, default=str)
    print(f"\n  Dataset stats saved: {DATASET_STATS_PATH}")

    # Step 1: Extract landmarks
    if not reuse_csv or not os.path.exists(CSV_PATH):
        csv_file = extract_landmarks()
    else:
        csv_file = CSV_PATH
        print(f"\n  Reusing existing CSV: {csv_file}")

    # Step 2: Train
    wrapper, le, scaler, X_test, y_test, history, dataset_stats = train_mlp(csv_file)

    # Step 3: Evaluate
    class_names = le.classes_
    accuracy, precision, recall, f1, per_class, ranked_pairs, recommendations, cm, y_pred = evaluate_model(
        wrapper, X_test, y_test, class_names
    )

    # Save collision report
    collision_data = {
        "test_accuracy_pct": round(accuracy * 100, 2),
        "test_precision_pct": round(precision * 100, 2),
        "test_recall_pct": round(recall * 100, 2),
        "test_f1_pct": round(f1 * 100, 2),
        "total_collision_pairs": len(ranked_pairs),
        "top_10_collisions": [
            {
                "rank": p["rank"],
                "pair": p["pair"],
                "total_confusion_pct": p["total_confusion_pct"],
                "a_as_b_pct": p["a_as_b_pct"],
                "b_as_a_pct": p["b_as_a_pct"],
                "total_samples": p["total_samples"],
            }
            for p in ranked_pairs[:10]
        ],
        "per_class_accuracy": {
            letter: round(info["accuracy"], 1)
            for letter, info in per_class.items()
        },
        "recommendations": recommendations,
    }
    with open(COLLISION_REPORT_PATH, "w") as f:
        json.dump(collision_data, f, indent=2)
    print(f"\n  Collision report: {COLLISION_REPORT_PATH}")

    # -- Final Summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE -- FINAL SUMMARY")
    print("=" * 60)
    print(f"  Dataset:")
    print(f"    Raw images on disk: {raw_stats['total_files_on_disk']}")
    print(f"    Valid images      : {raw_stats['total_valid_images']}")
    print(f"    Corrupt removed   : {raw_stats['corrupt_removed']}")
    print(f"    Landmark samples  : {dataset_stats['num_samples']}")
    print(f"  Model:")
    print(f"    Architecture      : Dense(256?128?64?32) + BatchNorm + Dropout")
    print(f"    Input features    : {dataset_stats['num_features']}")
    print(f"    Classes           : {dataset_stats['num_classes']}")
    print(f"  Training:")
    best_val = max(history.history["val_accuracy"])
    final_train = history.history["accuracy"][-1]
    print(f"    Final train acc   : {final_train:.2%}")
    print(f"    Best val acc      : {best_val:.2%}")
    print(f"  Test performance:")
    print(f"    Test accuracy     : {accuracy:.2%}")
    print(f"    Precision         : {precision:.2%}")
    print(f"    Recall            : {recall:.2%}")
    print(f"    F1 Score          : {f1:.2%}")
    print(f"  Top collision pairs:")
    for pair in ranked_pairs[:5]:
        print(f"    {pair['rank']}. {pair['pair']}: {pair['total_confusion_pct']:.1f}%")
    print(f"  Letters needing more data (HIGH priority):")
    high_priority = [r for r in recommendations if r["priority"] == "HIGH"]
    if high_priority:
        for rec in high_priority:
            print(f"    {rec['letter']}: {rec['accuracy']:.1f}% -- {rec['reason']}")
    else:
        print(f"    None -- all classes above 85%")
    print(f"  Artefacts:")
    print(f"    Model             : {KERAS_MODEL_PATH}")
    print(f"    Wrapper           : {MODEL_PATH}")
    print(f"    Label encoder     : {ENCODER_PATH}")
    print(f"    Scaler            : {SCALER_PATH}")
    print(f"    Confusion matrix  : {CM_PATH}")
    print(f"    Training curves   : {TRAINING_CURVES_PATH}")
    print(f"    Per-class acc     : {PER_CLASS_ACC_PATH}")
    print(f"    Collision chart   : {COLLISION_BAR_PATH}")
    print(f"    Collision report  : {COLLISION_REPORT_PATH}")
    print(f"    Dataset stats     : {DATASET_STATS_PATH}")
    print("=" * 60)

    return {
        "raw_stats": raw_stats,
        "dataset_stats": dataset_stats,
        "history": history,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
        "collisions": ranked_pairs,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sign Language A-Z -- Deep MLP Training Pipeline"
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Force re-extraction of landmarks from raw images (skips existing CSV)"
    )
    args = parser.parse_args()

    results = run_pipeline(reuse_csv=not args.force_extract)
