"""
VisionSpeak CNN Sign Language — Dataset Auditor
==============================================
Scans A-Z dataset directories under archive/asl_alphabet_train/asl_alphabet_train.
Identifies duplicates, corrupted images, class counts, imbalances, and resolutions.
Saves details to models/dataset_audit_report.md.
"""

import os
import hashlib
import cv2
from collections import Counter
from tqdm import tqdm

# Paths
DATASET_DIR = os.path.join("archive", "asl_alphabet_train", "asl_alphabet_train")
REPORT_PATH = os.path.join("models", "dataset_audit_report.md")


def get_md5(file_path):
    """Computes MD5 hash of a file to detect duplicates."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None


def run_audit():
    print("\n" + "=" * 60)
    print("  RUNNING DATASET AUDIT ON ASL ALPHABET")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory '{DATASET_DIR}' not found.")
        return

    # Keep only A-Z classes
    classes = [chr(i) for i in range(65, 91)]
    class_counts = {}
    corrupted_images = []
    duplicate_images = {}
    md5_registry = {}
    resolutions = Counter()

    total_images_processed = 0

    print("Auditing classes A-Z...")
    for label in sorted(classes):
        class_path = os.path.join(DATASET_DIR, label)
        if not os.path.exists(class_path):
            print(f"[WARN] Class directory '{label}' is missing.")
            class_counts[label] = 0
            continue

        files = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
        class_counts[label] = len(files)

        for filename in tqdm(files, desc=f"Class {label}", leave=False):
            file_path = os.path.join(class_path, filename)
            total_images_processed += 1

            # 1. Check for duplicates
            file_hash = get_md5(file_path)
            if file_hash:
                if file_hash in md5_registry:
                    duplicate_images[file_path] = md5_registry[file_hash]
                else:
                    md5_registry[file_hash] = file_path

            # 2. Check for corruption & profile resolution
            try:
                img = cv2.imread(file_path)
                if img is None or img.size == 0:
                    corrupted_images.append((file_path, "Unable to read image matrix"))
                else:
                    h, w, c = img.shape
                    resolutions[(w, h)] += 1
            except Exception as e:
                corrupted_images.append((file_path, str(e)))

    # Compute statistics
    total_valid_unique = len(md5_registry)
    total_duplicates = len(duplicate_images)
    total_corrupted = len(corrupted_images)
    
    # Class imbalance calculations
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)
    max_count = class_counts[max_class]
    min_count = class_counts[min_class]
    imbalance_ratio = max_count / max_count if min_count == 0 else max_count / min_count

    # Generate Markdown Report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# VisionSpeak CNN Dataset Audit Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"* **Total Images Analyzed:** {total_images_processed}\n")
        f.write(f"* **Valid Unique Images:** {total_valid_unique}\n")
        f.write(f"* **Duplicate Files Found:** {total_duplicates}\n")
        f.write(f"* **Corrupted Files Found:** {total_corrupted}\n")
        f.write(f"* **Imbalance Ratio (Max/Min Class):** {imbalance_ratio:.2f} (Max: {max_class} with {max_count}, Min: {min_class} with {min_count})\n\n")

        f.write("## 2. Class Distribution Table\n")
        f.write("| Class | Image Count |\n")
        f.write("|-------|-------------|\n")
        for label, count in class_counts.items():
            f.write(f"| {label} | {count} |\n")
        f.write("\n")

        f.write("## 3. Resolution Profile\n")
        f.write("| Resolution (W x H) | Image Count |\n")
        f.write("|-------------------|-------------|\n")
        for res, count in resolutions.most_common():
            f.write(f"| {res[0]}x{res[1]} | {count} |\n")
        f.write("\n")

        f.write("## 4. Corrupted Images Log\n")
        if corrupted_images:
            f.write("| File Path | Error Reason |\n")
            f.write("|-----------|--------------|\n")
            for path, reason in corrupted_images[:50]:  # Limit log to top 50
                f.write(f"| `{path}` | {reason} |\n")
            if len(corrupted_images) > 50:
                f.write(f"\n*And {len(corrupted_images) - 50} more corrupted files.*")
        else:
            f.write("*No corrupted images found in dataset.*\n")
        f.write("\n")

        f.write("## 5. Duplicate Images Sample Log (Top 50)\n")
        if duplicate_images:
            f.write("| Duplicate Path | Original Path |\n")
            f.write("|----------------|---------------|\n")
            for dup, orig in list(duplicate_images.items())[:50]:
                f.write(f"| `{dup}` | `{orig}` |\n")
        else:
            f.write("*No duplicate images found in dataset.*\n")

    print("\n" + "=" * 60)
    print(f"[OK] Audit finished successfully.")
    print(f"     Report saved to: {REPORT_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_audit()
