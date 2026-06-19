"""
VisionSpeak — Real-Time Model Testing Utility (Phase 8)
=======================================================
Loads a trained model + scaler + encoder and runs live webcam prediction.
Reports per-frame: prediction, confidence, top-3 alternatives, FPS, stability.

Usage:
  python test_real_time.py                              # default: mlp_model.pkl
  python test_real_time.py --model models/mlp_model_v2.pkl --scaler models/scaler_v2.pkl
  python test_real_time.py --collisions                 # highlight collision risk pairs
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import argparse
import time
import collections

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import joblib
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import mediapipe as mp

from utils import extract_enhanced_features, extract_enhanced_features_v2

# Collision pairs we watch for
COLLISION_PAIRS = {
    ("G", "H"), ("M", "N"), ("U", "V"), ("C", "O"),
    ("S", "T"), ("W", "E"), ("D", "F"), ("A", "J"),
    ("A", "W"), ("A", "G"), ("G", "R"), ("H", "J"),
    ("I", "P"), ("I", "Q"), ("M", "Y"), ("P", "Q"), ("S", "Z"),
}

# Colour palette (BGR)
TEAL = (208, 194, 0)
GREEN = (100, 255, 100)
RED = (80, 80, 255)
YELLOW = (60, 220, 255)
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)
ORANGE = (30, 140, 255)
PURPLE = (220, 80, 220)
BG = (30, 20, 20)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Model Test")
    parser.add_argument("--model", default="models/mlp_model.pkl")
    parser.add_argument("--scaler", default="models/scaler.pkl")
    parser.add_argument("--encoder", default="models/label_encoder.pkl")
    parser.add_argument("--collisions", action="store_true", help="Highlight collision pairs")
    args = parser.parse_args()

    # ── Load artefacts ──────────────────────────────────────────────────
    print(f"  Loading model  : {args.model}")
    print(f"  Loading scaler : {args.scaler}")
    print(f"  Loading encoder: {args.encoder}")
    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    le = joblib.load(args.encoder)

    # Detect whether this is v1 (99 features) or v2 (136+ features)
    n_features = model.coefs_[0].shape[0]
    use_v2 = n_features > 100
    extract_fn = extract_enhanced_features_v2 if use_v2 else extract_enhanced_features
    print(f"  Feature set    : {'v2' if use_v2 else 'v1'} ({n_features} features)")

    # ── MediaPipe ───────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    fps = 0.0
    prev_time = time.time()
    pred_history = deque(maxlen=30)
    stable_pred = ""
    stable_count = 0
    frame_no = 0

    print("\n  [OK] Webcam open — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame_no += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pred_label = ""
        confidence = 0.0
        top3 = []
        collision_warning = False
        second_best = ""
        second_conf = 0.0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            features = extract_fn(hand).reshape(1, -1)
            scaled = scaler.transform(features)

            probs = model.predict_proba(scaled)[0]
            top_indices = np.argsort(probs)[::-1]
            pred_idx = top_indices[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            top3 = [(le.inverse_transform([i])[0], float(probs[i])) for i in top_indices[:3]]
            second_best = top3[1][0] if len(top3) > 1 else ""
            second_conf = top3[1][1] if len(top3) > 1 else 0.0

            pred_history.append(pred_label)
            if len(pred_history) == pred_history.maxlen:
                counts = collections.Counter(pred_history)
                stable_pred, stable_count = counts.most_common(1)[0]
            else:
                stable_pred = pred_label

            # Collision check
            if args.collisions and second_best:
                pair = (pred_label, second_best)
                rev_pair = (second_best, pred_label)
                if pair in COLLISION_PAIRS or rev_pair in COLLISION_PAIRS:
                    collision_warning = True

        # ── FPS ─────────────────────────────────────────────────────────
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── Draw overlay ────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), BG, -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        # Prediction panel
        panel_x, panel_y = w - 320, 20
        panel_w, panel_h = 290, 400
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (40, 30, 30), -1)
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), PURPLE, 1)

        y = panel_y + 40
        cv2.putText(frame, "REAL-TIME TEST", (panel_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PURPLE, 1)
        y += 10

        if pred_label:
            # Big prediction letter
            font_size = 3.5
            color = GREEN if confidence > 0.9 else (YELLOW if confidence > 0.7 else RED)
            cv2.putText(frame, pred_label, (panel_x + 30, y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 3)

            # Confidence
            cv2.putText(frame, f"{confidence * 100:.1f}%",
                        (panel_x + 30, y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Top-3 list
            y2 = y + 155
            cv2.putText(frame, "Top 3:", (panel_x + 15, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
            y2 += 25
            for rank, (lbl, prob) in enumerate(top3, 1):
                c = GREEN if rank == 1 else WHITE
                cv2.putText(frame, f"{rank}. {lbl}  {prob * 100:.1f}%",
                            (panel_x + 20, y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                y2 += 22

            # Second best
            if second_best:
                gap = confidence - second_conf
                gap_color = ORANGE if gap < 0.15 else GREEN
                cv2.putText(frame, f"Alt: {second_best} ({second_conf * 100:.0f}%)",
                            (panel_x + 15, y2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, gap_color, 1)
                cv2.putText(frame, f"Gap: {gap:.3f}",
                            (panel_x + 15, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, gap_color, 1)

            # Collision warning
            if collision_warning:
                cv2.putText(frame, "⚠ COLLISION RISK",
                            (panel_x + 15, y2 + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)

            # Stability
            cv2.putText(frame, f"Stable: {stable_pred} ({stable_count}/{pred_history.maxlen})",
                        (panel_x + 15, y2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

        else:
            cv2.putText(frame, "No Hand Detected",
                        (panel_x + 30, panel_y + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1)

        # FPS in top-left
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEAL, 1)

        # Model info
        info = f"Model: {os.path.basename(args.model)}"
        cv2.putText(frame, info, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

        cv2.imshow("VisionSpeak — Real-Time Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n  [OK] Test complete.\n")


if __name__ == "__main__":
    main()
