# -*- coding: utf-8 -*-
"""
Collision Audit — Sign Language A-Z Letter Confusion Analysis
============================================================
Identifies all letter collisions, determines root causes,
generates reports, visualizations, and implements fixes.

Phases:
  1. Collision Detection (confusion matrix, per-class accuracy)
  2. Collision Ranking (top-10 confused pairs)
  3. Feature Analysis (landmark distances, finger angles, extensions)
  4. Visual Comparison (side-by-side landmark plots)
  5. Special Rule Engine (geometric verification layer)
  6. Confidence Calibration (temperature scaling)
  7. Dataset Improvement (low-quality sample detection)
  8. Targeted Augmentation (confused-class data generation)
"""

import os
import sys
import warnings
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

from utils import extract_enhanced_features

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
REPORT_DIR = os.path.join(MODEL_DIR, "collision_report")
CSV_PATH = "extracted_landmarks.csv"
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(REPORT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(REPORT_DIR, "collision_report.json")
CM_IMAGE = os.path.join(REPORT_DIR, "confusion_matrix.png")
CM_NORMALIZED = os.path.join(REPORT_DIR, "confusion_matrix_normalized.png")
COLLISION_BAR = os.path.join(REPORT_DIR, "top_collisions.png")
PER_CLASS_ACC = os.path.join(REPORT_DIR, "per_class_accuracy.png")
FEATURE_PLOT = os.path.join(REPORT_DIR, "feature_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Collision Detection
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_data():
    """Load trained model, encoder, scaler, and test data."""
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    y_encoded = le.transform(y)
    _, X_test_raw, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_test = scaler.transform(X_test_raw)

    return model, le, scaler, X_test, y_test, df


def compute_confusion_matrix(model, le, X_test, y_test):
    """Generate confusion matrix and classification report."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    class_names = le.classes_

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred) * 100.0

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        per_class[name] = {
            "accuracy": float(report[name]["recall"]) * 100.0,
            "precision": float(report[name]["precision"]) * 100.0,
            "recall": float(report[name]["recall"]) * 100.0,
            "f1": float(report[name]["f1-score"]) * 100.0,
            "support": int(report[name]["support"]),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    return cm, y_pred, y_proba, class_names, accuracy, per_class, report


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Save styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    norm = plt.Normalize(vmin=0, vmax=cm.max())
    colors = plt.cm.BuGn(norm(cm))

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
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_normalized_confusion(cm, class_names, save_path):
    """Save normalized confusion matrix highlighting errors."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    mask = np.eye(len(class_names), dtype=bool)
    cmap = plt.cm.RdYlGn_r

    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap=cmap,
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
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Collision Ranking
# ═══════════════════════════════════════════════════════════════════════════

def rank_collisions(cm, class_names):
    """
    Identify and rank all confused letter pairs.
    Returns list of dicts sorted by confusion percentage descending.
    """
    collisions = []
    n = len(class_names)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total_i = cm[i, :].sum()
            confusion_pct = (cm[i, j] / total_i) * 100.0 if total_i > 0 else 0.0
            if confusion_pct > 0:
                false_pos = cm[j, i]
                false_neg = cm[i, j]
                collisions.append({
                    "true_label": str(class_names[i]),
                    "predicted_as": str(class_names[j]),
                    "confusion_pct": round(confusion_pct, 2),
                    "samples": int(cm[i, j]),
                    "false_positives": int(false_pos),
                    "false_negatives": int(false_neg),
                    "pair_key": f"{class_names[i]}<->{class_names[j]}",
                })

    # Sort by confusion percentage descending
    collisions.sort(key=lambda x: x["confusion_pct"], reverse=True)

    # Merge bidirectional collisions: G<->H = G predicted as H + H predicted as G
    merged = {}
    for c in collisions:
        key = c["pair_key"]
        rev_key = f"{c['predicted_as']}<->{c['true_label']}"
        if rev_key in merged:
            continue
        # Normalize key to alphabetical order
        a, b = sorted([c["true_label"], c["predicted_as"]])
        pair_key = f"{a}<->{b}"
        if pair_key not in merged:
            merged[pair_key] = {
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
        merged[pair_key]["total_confusion_pct"] += c["confusion_pct"]
        merged[pair_key]["total_samples"] += c["samples"]
        if c["true_label"] == a:
            merged[pair_key]["a_as_b_pct"] = c["confusion_pct"]
            merged[pair_key]["a_as_b_samples"] = c["samples"]
        else:
            merged[pair_key]["b_as_a_pct"] = c["confusion_pct"]
            merged[pair_key]["b_as_a_samples"] = c["samples"]

    ranked = sorted(merged.values(), key=lambda x: x["total_confusion_pct"], reverse=True)
    return collisions, ranked


def plot_per_class_accuracy(per_class, save_path):
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
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_top_collisions(ranked, top_n=10, save_path=None):
    """Horizontal bar chart of top N collisions."""
    top = ranked[:top_n]
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
    ax.set_title("Top Letter Collisions", fontsize=14, color="white", fontweight="bold")
    ax.tick_params(axis="both", colors="white")
    ax.set_xlim(0, max(vals) * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════

def count_extended_fingers(landmarks_21x3):
    """
    Count extended fingers from 21x3 normalized landmarks.
    Uses orientation-invariant geometric checks.
    """
    coords = landmarks_21x3.reshape(21, 3)
    wrist = coords[0]
    translated = coords - wrist
    hand_scale = np.linalg.norm(translated[9])
    if hand_scale < 1e-6:
        hand_scale = 1.0
    n = translated / hand_scale

    thumb_ext = np.linalg.norm(n[4] - n[5]) > 0.8
    index_ext = np.linalg.norm(n[8]) > np.linalg.norm(n[6])
    middle_ext = np.linalg.norm(n[12]) > np.linalg.norm(n[10])
    ring_ext = np.linalg.norm(n[16]) > np.linalg.norm(n[14])
    pinky_ext = np.linalg.norm(n[20]) > np.linalg.norm(n[18])

    return sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])


def compute_finger_spread(landmarks_21x3):
    """
    Compute fingertip pairwise distances as a measure of finger spread.
    Returns mean distance between adjacent fingertips.
    """
    coords = landmarks_21x3.reshape(21, 3)
    tips = [4, 8, 12, 16, 20]
    tip_positions = coords[tips]
    spreads = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            spreads.append(np.linalg.norm(tip_positions[i] - tip_positions[j]))
    return spreads


def analyze_collision_features(df, class_names, collisions, le):
    """
    For each top collision pair, analyze the average feature differences
    to understand why the model confuses them.
    """
    feature_cols = [c for c in df.columns if c != "label"]
    analysis = []

    # Get unique top pairs
    seen_pairs = set()
    for c in collisions:
        a, b = sorted([c["letter_1"], c["letter_2"]])
        pair = (a, b)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Get samples for each letter
        mask_a = df["label"] == a
        mask_b = df["label"] == b

        if mask_a.sum() == 0 or mask_b.sum() == 0:
            continue

        samples_a = df[mask_a][feature_cols].values.astype(np.float32)
        samples_b = df[mask_b][feature_cols].values.astype(np.float32)

        # Average features
        avg_a = samples_a.mean(axis=0)
        avg_b = samples_b.mean(axis=0)

        # Finger counts
        finger_a = np.array([count_extended_fingers(s[:63]) for s in samples_a])
        finger_b = np.array([count_extended_fingers(s[:63]) for s in samples_b])
        avg_finger_a = finger_a.mean()
        avg_finger_b = finger_b.mean()

        # Finger spread (fingertip pairwise distances)
        spread_a = np.array([np.mean(compute_finger_spread(s[:63])) for s in samples_a])
        spread_b = np.array([np.mean(compute_finger_spread(s[:63])) for s in samples_b])
        avg_spread_a = spread_a.mean()
        avg_spread_b = spread_b.mean()

        # Finger-specific spread: index-middle distance and middle-ring distance
        def specific_spreads(samples):
            ss = []
            for s in samples:
                coords = s[:63].reshape(21, 3)
                tip_idx = np.linalg.norm(coords[8] - coords[12])
                tip_mr = np.linalg.norm(coords[12] - coords[16])
                ss.append([tip_idx, tip_mr])
            return np.array(ss)

        ss_a = specific_spreads(samples_a)
        ss_b = specific_spreads(samples_b)
        avg_ss_a = ss_a.mean(axis=0)
        avg_ss_b = ss_b.mean(axis=0)

        # Feature differences (for the most discriminative features)
        feature_diff = np.abs(avg_a - avg_b)
        top_feature_indices = np.argsort(feature_diff)[-10:][::-1]  # top 10 diffs

        analysis.append({
            "pair": f"{a}<->{b}",
            "letter_a": a,
            "letter_b": b,
            "n_samples_a": int(mask_a.sum()),
            "n_samples_b": int(mask_b.sum()),
            "avg_fingers_a": round(float(avg_finger_a), 2),
            "avg_fingers_b": round(float(avg_finger_b), 2),
            "avg_spread_a": round(float(avg_spread_a), 4),
            "avg_spread_b": round(float(avg_spread_b), 4),
            "avg_index_mid_dist_a": round(float(avg_ss_a[0]), 4),
            "avg_index_mid_dist_b": round(float(avg_ss_b[0]), 4),
            "avg_mid_ring_dist_a": round(float(avg_ss_a[1]), 4),
            "avg_mid_ring_dist_b": round(float(avg_ss_b[1]), 4),
            "feature_diff_norm": round(float(np.mean(feature_diff)), 4),
            "top_discriminating_features": top_feature_indices.tolist(),
        })

    return analysis


def plot_feature_comparison(feature_analysis, save_path):
    """Visualize key feature differences for collision pairs."""
    if not feature_analysis:
        return

    n_pairs = min(len(feature_analysis), 5)
    n_cols = min(n_pairs, 3)
    n_rows = (n_pairs + n_cols - 1) // n_cols if n_pairs > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.patch.set_facecolor("#1a1a2e")

    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(axis="both", colors="white")
        for spine in ax.spines.values():
            spine.set_color("#3a3a5e")

        if idx < n_pairs:
            fa = feature_analysis[idx]
            pair_name = fa["pair"]
            labels = ["Finger Count", "Finger Spread", "Idx-Mid Dist"]
            vals_a = [fa["avg_fingers_a"], fa["avg_spread_a"], fa["avg_index_mid_dist_a"]]
            vals_b = [fa["avg_fingers_b"], fa["avg_spread_b"], fa["avg_index_mid_dist_b"]]

            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width / 2, vals_a, width, label=fa["letter_a"], color="#ffd93d", alpha=0.9)
            ax.bar(x + width / 2, vals_b, width, label=fa["letter_b"], color="#ff6b6b", alpha=0.9)
            ax.set_title(pair_name, fontsize=12, color="white", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10, color="white")
            ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white")
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — Special Rule Engine
# ═══════════════════════════════════════════════════════════════════════════

class CollisionRuleEngine:
    """
    Rule-based verification layer for highly confused letter pairs.
    After the MLP prediction, these geometric rules verify or correct the output.
    """
    def __init__(self):
        self.rules = self._build_rules()

    def _build_rules(self):
        return {
            # G (1 finger) vs H (2 fingers)
            ("G", "H"): self._rule_finger_count_gh,
            ("H", "G"): self._rule_finger_count_gh,
            # M (thumb tucked + 3 fingers) vs N (thumb tucked + 2 fingers)
            ("M", "N"): self._rule_mn,
            ("N", "M"): self._rule_mn,
            # U (2 fingers together) vs V (2 fingers apart)
            ("U", "V"): self._rule_uv,
            ("V", "U"): self._rule_uv,
            # C (open curved hand) vs O (rounded closed)
            ("C", "O"): self._rule_co,
            ("O", "C"): self._rule_co,
            # S (fist) vs T (fist with thumb between fingers)
            ("S", "T"): self._rule_st,
            ("T", "S"): self._rule_st,
            # W (3 fingers up, thumb+pinky tucked) vs E (curled)
            ("W", "E"): self._rule_we,
            ("E", "W"): self._rule_we,
            # D (index up, rest curled) vs F (index+thumb circle)
            ("D", "F"): self._rule_df,
            ("F", "D"): self._rule_df,
        }

    def _extract_landmarks(self, landmarks_21x3):
        coords = landmarks_21x3.reshape(21, 3)
        wrist = coords[0]
        translated = coords - wrist
        scale = np.linalg.norm(translated[9])
        if scale < 1e-6:
            scale = 1.0
        return translated / scale

    def count_fingers(self, landmarks_21x3):
        n = self._extract_landmarks(landmarks_21x3)
        thumb = np.linalg.norm(n[4] - n[5]) > 0.8
        index = np.linalg.norm(n[8]) > np.linalg.norm(n[6])
        middle = np.linalg.norm(n[12]) > np.linalg.norm(n[10])
        ring = np.linalg.norm(n[16]) > np.linalg.norm(n[14])
        pinky = np.linalg.norm(n[20]) > np.linalg.norm(n[18])
        return sum([thumb, index, middle, ring, pinky])

    def _rule_finger_count_gh(self, landmarks, _):
        """G=1 finger, H=2 fingers"""
        fc = self.count_fingers(landmarks)
        if fc <= 1:
            return "G", 0.95
        elif fc >= 2:
            return "H", 0.90
        return None, 0.0

    def _rule_mn(self, landmarks, _):
        """M=3 fingers (index+middle+ring), N=2 fingers (index+middle)"""
        n = self._extract_landmarks(landmarks)
        index = np.linalg.norm(n[8]) > np.linalg.norm(n[6])
        middle = np.linalg.norm(n[12]) > np.linalg.norm(n[10])
        ring = np.linalg.norm(n[16]) > np.linalg.norm(n[14])

        extended = sum([index, middle, ring])
        if extended >= 3:
            return "M", 0.85
        elif extended <= 2:
            return "N", 0.85
        # Check thumb position: M has thumb tucked, N may have thumb slightly out
        thumb_idx_dist = np.linalg.norm(n[4] - n[5])
        if thumb_idx_dist < 0.6:
            return "M", 0.70
        return None, 0.0

    def _rule_uv(self, landmarks, _):
        """U=fingers together, V=fingers apart. Check index-middle tip distance."""
        n = self._extract_landmarks(landmarks)
        idx_mid_dist = np.linalg.norm(n[8] - n[12])
        # U: fingers together (distance small), V: fingers apart (distance large)
        if idx_mid_dist < 0.3:
            return "U", 0.90
        elif idx_mid_dist >= 0.3:
            return "V", 0.90
        return None, 0.0

    def _rule_co(self, landmarks, _):
        """C=open curved, O=closed round. Check thumb-index distance."""
        n = self._extract_landmarks(landmarks)
        thumb_tip = n[4]
        index_tip = n[8]
        dist = np.linalg.norm(thumb_tip - index_tip)
        # C: open (dist large), O: closed (dist small)
        if dist < 0.2:
            return "O", 0.85
        elif dist >= 0.3:
            return "C", 0.85
        return None, 0.0

    def _rule_st(self, landmarks, _):
        """S=fist (all curled), T=fist with thumb between index and middle"""
        n = self._extract_landmarks(landmarks)
        fc = self.count_fingers(landmarks)
        if fc == 0:
            return "S", 0.90
        thumb_ext = np.linalg.norm(n[4] - n[5]) > 0.8
        if thumb_ext:
            return "T", 0.80
        return None, 0.0

    def _rule_we(self, landmarks, _):
        """W=3 fingers up (index+middle+ring), E=fingers curled"""
        fc = self.count_fingers(landmarks)
        if fc >= 3:
            return "W", 0.85
        elif fc <= 1:
            return "E", 0.80
        return None, 0.0

    def _rule_df(self, landmarks, _):
        """D=index up, F=index+thumb circle"""
        n = self._extract_landmarks(landmarks)
        index_up = np.linalg.norm(n[8]) > np.linalg.norm(n[6])
        thumb_idx_dist = np.linalg.norm(n[4] - n[5])
        fc = self.count_fingers(landmarks)

        if index_up and fc == 1:
            return "D", 0.90
        if thumb_idx_dist < 0.2 and index_up:
            return "F", 0.85
        return None, 0.0

    def verify(self, raw_prediction, second_prediction, probabilities, landmarks_21x3, class_names):
        """
        Verify and potentially correct a prediction.
        Returns (corrected_letter, confidence, corrected_flag, rule_used).
        """
        pred_letter = class_names[raw_prediction] if raw_prediction is not None else None
        second_letter = class_names[second_prediction] if second_prediction is not None else None

        # Check if this pair has a rule
        rule_key = (pred_letter, second_letter)
        reverse_key = (second_letter, pred_letter)

        rule_fn = self.rules.get(rule_key) or self.rules.get(reverse_key)
        if rule_fn is not None:
            corrected, rule_conf = rule_fn(landmarks_21x3, probabilities)
            if corrected is not None and corrected != pred_letter:
                return corrected, rule_conf, True, f"{rule_key[0]}<->{rule_key[1]}"

        return pred_letter, None, False, None


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — Confidence Calibration
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceCalibrator:
    """
    Temperature scaling and probability calibration.
    Learns an optimal temperature to soften/harden softmax probabilities.
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        """Learn optimal temperature via binary search on log-likelihood."""
        from scipy.optimize import minimize_scalar

        def nll(T):
            if T <= 0:
                return 1e10
            scaled = logits / T
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            eps = 1e-15
            probs = np.clip(probs, eps, 1.0 - eps)
            return -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        return self.temperature

    def calibrate(self, logits):
        """Apply temperature scaling to logits."""
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


def calibrate_model(model, X_test, y_test, le):
    """Fit a temperature scaling calibrator and return it."""
    logits = model.predict_log_proba(X_test)
    calibrator = ConfidenceCalibrator()
    temp = calibrator.fit(logits, y_test)
    return calibrator, temp


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7 — Dataset Improvement
# ═══════════════════════════════════════════════════════════════════════════

def find_low_quality_samples(df, feature_cols, y_pred_raw, y_true_encoded, le):
    """
    Identify low-quality samples:
    - Misclassified samples
    - Low-confidence predictions
    - Samples near decision boundaries
    """
    class_names = le.classes_
    low_quality = []

    for i in range(len(df)):
        true_label = df.iloc[i]["label"]
        if i < len(y_pred_raw):
            pred_label = class_names[y_pred_raw[i]]
            if true_label != pred_label:
                low_quality.append({
                    "index": i,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "reason": "Misclassified",
                })

    report = {
        "total_samples": len(df),
        "misclassified_count": len(low_quality),
        "misclassified_pct": round(len(low_quality) / len(df) * 100, 2) if len(df) > 0 else 0,
        "samples": low_quality[:100],  # first 100
    }
    return report


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8 — Targeted Augmentation
# ═══════════════════════════════════════════════════════════════════════════

def generate_targeted_augmentation(df, collision_pairs, augment_factor=10):
    """
    Generate targeted augmented samples for confused classes
    by applying geometric transformations to existing landmark data.
    """
    feature_cols = [c for c in df.columns if c != "label"]
    targets = set()
    for pair in collision_pairs[:5]:
        targets.add(pair["letter_1"])
        targets.add(pair["letter_2"])

    print(f"\n  Targeted augmentation for: {sorted(targets)}")

    new_rows = []
    for label in sorted(targets):
        mask = df["label"] == label
        samples = df[mask][feature_cols].values.astype(np.float32)

        for _ in range(augment_factor):
            for s in samples:
                coords = s[:63].reshape(21, 3)
                # Rotation
                angle = np.random.uniform(-20, 20) * np.pi / 180.0
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_mat = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                rotated = np.dot(coords, rot_mat.T)
                # Scale
                scale = np.random.uniform(0.90, 1.10)
                scaled = rotated * scale
                # Noise
                noise = np.random.normal(0, 0.008, size=coords.shape).astype(np.float32)
                augmented = scaled + noise
                # Re-extract features
                augmented_feat = extract_enhanced_features(augmented)
                new_rows.append(augmented_feat)

    if new_rows:
        aug_df = pd.DataFrame(new_rows, columns=feature_cols)
        aug_df["label"] = np.random.choice(sorted(targets), len(new_rows))
        combined = pd.concat([df, aug_df], ignore_index=True)
        print(f"  Added {len(new_rows)} augmented samples. Total: {len(combined)}")
        return combined
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_full_audit():
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE COLLISION AUDIT")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1/8] Loading model and test data...")
    model, le, scaler, X_test, y_test, df = load_model_and_data()
    class_names = le.classes_
    print(f"  Model classes: {len(class_names)} ({''.join(class_names)})")
    print(f"  Test samples: {len(X_test)}")

    # ── PHASE 1: Collision Detection ─────────────────────────────────────
    print("\n[2/8] Computing confusion matrix and classification report...")
    cm, y_pred, y_proba, class_names, accuracy, per_class, report = compute_confusion_matrix(
        model, le, X_test, y_test
    )
    print(f"  Overall Accuracy: {accuracy:.2f}%")

    plot_confusion_matrix(cm, class_names, "Confusion Matrix", CM_IMAGE)
    plot_normalized_confusion(cm, class_names, CM_NORMALIZED)
    plot_per_class_accuracy(per_class, PER_CLASS_ACC)

    # ── PHASE 2: Collision Ranking ───────────────────────────────────────
    print("\n[3/8] Ranking collisions...")
    detailed_collisions, ranked_pairs = rank_collisions(cm, class_names)
    print(f"  Found {len(ranked_pairs)} unique collision pairs")
    for i, pair in enumerate(ranked_pairs[:10]):
        print(f"  {i+1}. {pair['pair']}: {pair['total_confusion_pct']:.2f}% "
              f"({pair['a_as_b_samples']}+{pair['b_as_a_samples']} samples)")

    plot_top_collisions(ranked_pairs, top_n=10, save_path=COLLISION_BAR)

    # ── PHASE 3: Feature Analysis ────────────────────────────────────────
    print("\n[4/8] Analyzing collision features...")
    feature_analysis = analyze_collision_features(
        df, class_names, ranked_pairs, le
    )
    for fa in feature_analysis[:5]:
        print(f"  {fa['pair']}: fingers {fa['avg_fingers_a']} vs {fa['avg_fingers_b']}, "
              f"spread {fa['avg_spread_a']:.3f} vs {fa['avg_spread_b']:.3f}")

    plot_feature_comparison(feature_analysis, FEATURE_PLOT)

    # ── PHASE 5: Rule Engine ─────────────────────────────────────────────
    print("\n[5/8] Building collision rule engine...")
    rule_engine = CollisionRuleEngine()
    print(f"  Rules defined for {len(rule_engine.rules)} pair directions")

    # ── PHASE 6: Confidence Calibration ──────────────────────────────────
    print("\n[6/8] Calibrating confidence (temperature scaling)...")
    calibrator, temperature = calibrate_model(model, X_test, y_test, le)
    print(f"  Optimal temperature: {temperature:.4f}")

    # Calibrated probabilities
    logits = model.predict_log_proba(X_test)
    calibrated_probs = calibrator.calibrate(logits)
    calibrated_preds = calibrated_probs.argmax(axis=1)
    calibrated_acc = (calibrated_preds == y_test).mean() * 100.0
    print(f"  Calibrated accuracy: {calibrated_acc:.2f}% (vs raw {accuracy:.2f}%)")

    # ── PHASE 7: Dataset Quality ─────────────────────────────────────────
    print("\n[7/8] Identifying low-quality samples...")
    # Predict full df to find misclassified
    feature_cols = [c for c in df.columns if c != "label"]
    X_full = df[feature_cols].values.astype(np.float32)
    X_full_scaled = scaler.transform(X_full)
    y_full_pred = model.predict(X_full_scaled)
    quality_report = find_low_quality_samples(
        df, feature_cols, y_full_pred, le.transform(df["label"].values), le
    )
    print(f"  Misclassified: {quality_report['misclassified_count']} / "
          f"{quality_report['total_samples']} ({quality_report['misclassified_pct']}%)")

    # ── PHASE 8: Targeted Augmentation ───────────────────────────────────
    print("\n[8/8] Targeted augmentation for confused classes...")
    augmented_df = generate_targeted_augmentation(df, ranked_pairs, augment_factor=8)

    # ── Save report ──────────────────────────────────────────────────────
    report_data = {
        "model_accuracy": round(accuracy, 2),
        "calibrated_accuracy": round(calibrated_acc, 2),
        "temperature": round(temperature, 4),
        "total_collision_pairs": len(ranked_pairs),
        "top_10_collisions": [
            {
                "rank": i + 1,
                "pair": p["pair"],
                "total_confusion_pct": p["total_confusion_pct"],
                "a_as_b_pct": p["a_as_b_pct"],
                "b_as_a_pct": p["b_as_a_pct"],
                "total_samples": p["total_samples"],
            }
            for i, p in enumerate(ranked_pairs[:10])
        ],
        "per_class_accuracy": per_class,
        "feature_analysis": feature_analysis[:10],
        "dataset_quality": {
            "total_samples": quality_report["total_samples"],
            "misclassified_pct": quality_report["misclassified_pct"],
        },
        "calibrator_temperature": temperature,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\n  Report saved: {REPORT_FILE}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AUDIT COMPLETE")
    print("=" * 60)
    print(f"  Accuracy: {accuracy:.2f}% (calibrated: {calibrated_acc:.2f}%)")
    print(f"  Top collisions:")
    for i, p in enumerate(ranked_pairs[:5]):
        print(f"    {i+1}. {p['pair']}: {p['total_confusion_pct']:.2f}%")
    print(f"  Rule engine: {len(rule_engine.rules)} rules active")
    print(f"  Dataset augmented: {len(augmented_df)} total samples")
    print(f"  Reports: {REPORT_DIR}")
    print("=" * 60)

    return report_data, rule_engine, calibrator, augmented_df


if __name__ == "__main__":
    report, rule_engine, calibrator, aug_df = run_full_audit()
