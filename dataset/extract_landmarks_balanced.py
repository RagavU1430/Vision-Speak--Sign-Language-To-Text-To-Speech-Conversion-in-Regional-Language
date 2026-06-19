"""
VisionSpeak — Balanced Landmark Extraction + Dataset Audit
==========================================================
Phase 1: Audits dataset (counts, corruption, hand detection rate)
Phase 2: Extracts exactly N landmarks per class, balanced across A-Z

Outputs:
  - landmarks_v1.csv  (balanced, 99 features per row)
  - Console report with per-class detection rates
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import hashlib
import time
import warnings
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
from utils import extract_enhanced_features, suppress_c_stderr

with suppress_c_stderr():
    import mediapipe as mp

DATASET_DIR = os.path.join("archive", "asl_alphabet_train", "asl_alphabet_train")
OUTPUT_CSV = os.path.join("dataset", "landmarks_v1.csv")
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
LABELS = [chr(i) for i in range(65, 91)]  # A-Z

SAMPLES_PER_CLASS = 1000


def get_md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_and_extract():
    print("=" * 60)
    print("  PHASE 1 — Dataset Audit")
    print("=" * 60)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    header = [f"f{i}" for i in range(99)] + ["label"]

    class_counts = {}
    detection_rates = {}
    corrupted = []
    duplicate_map = {}
    md5_registry = {}
    total_images = 0
    total_processed = 0
    total_hand_found = 0
    total_no_hand = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for label in sorted(LABELS):
            folder = os.path.join(DATASET_DIR, label)
            if not os.path.isdir(folder):
                print(f"  [WARN] '{label}' folder missing")
                class_counts[label] = 0
                detection_rates[label] = (0, 0, 0.0)
                continue

            files = sorted([
                fn for fn in os.listdir(folder)
                if fn.lower().endswith(VALID_EXTENSIONS)
            ])
            total_count = len(files)
            class_counts[label] = total_count

            sample_count = min(SAMPLES_PER_CLASS, total_count)
            selected = files[:sample_count]

            found = 0
            tried = 0

            for fn in tqdm(selected, desc=f"  {label} ({sample_count})", unit="img", leave=False):
                path = os.path.join(folder, fn)
                tried += 1

                # --- corruption + duplicate check (Phase 1) ---
                img = cv2.imread(path)
                if img is None:
                    corrupted.append(path)
                    continue

                fhash = get_md5(path)
                if fhash in md5_registry:
                    duplicate_map[path] = md5_registry[fhash]
                    continue
                md5_registry[fhash] = path

                # --- landmark extraction (Phase 2) ---
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    features = extract_enhanced_features(hand)
                    row = list(features) + [label]
                    writer.writerow(row)
                    found += 1
                    total_hand_found += 1
                else:
                    total_no_hand += 1

                total_processed += 1
                total_images += 1

            rate = (found / tried * 100) if tried > 0 else 0.0
            detection_rates[label] = (found, tried, rate)
            print(f"    {label}: {found}/{tried} detected ({rate:.1f}%)")

    hands.close()

    # ── PHASE 1 REPORT ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1 — AUDIT RESULTS")
    print("=" * 60)
    print(f"  Total images available   : {sum(class_counts.values())}")
    print(f"  Images examined          : {total_processed}")
    print(f"  Hands detected           : {total_hand_found}")
    print(f"  No hand found            : {total_no_hand}")
    print(f"  Corrupted images         : {len(corrupted)}")
    print(f"  Duplicate images         : {len(duplicate_map)}")
    print(f"  Output CSV               : {OUTPUT_CSV}")

    print(f"\n  -- Per-Class Detection --")
    print(f"  {'Class':>5} {'Avail':>7} {'Sampled':>8} {'Found':>6} {'Rate':>6}")
    for label in sorted(LABELS):
        f, t, r = detection_rates.get(label, (0, 0, 0.0))
        print(f"  {label:>5} {class_counts.get(label, 0):>7} {min(SAMPLES_PER_CLASS, class_counts.get(label, 0)):>8} {f:>6} {r:>5.1f}%")

    # ── Final count in CSV ────────────────────────────────────────────────
    with open(OUTPUT_CSV, "r") as f:
        lines = sum(1 for _ in f) - 1
    print(f"\n  Total samples in CSV     : {lines}")
    print(f"  Target                   : {len(LABELS) * SAMPLES_PER_CLASS}")
    print("=" * 60)
    return OUTPUT_CSV


if __name__ == "__main__":
    t0 = time.time()
    out = audit_and_extract()
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  →  {out}")
