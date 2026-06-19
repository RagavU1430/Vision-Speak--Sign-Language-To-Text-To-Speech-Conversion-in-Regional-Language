# File Movement & Import Change Report

## Moved Files

| # | Old Location | New Location | Import Changes |
|---|-------------|-------------|----------------|
| 1 | `supabase_client.py` | `database/supabase_client.py` | `from supabase_client import` → `from database.supabase_client import` in `predict_live.py` and `auth_gui.py` |
| 2 | `.sql/setup_emergency_db.sql` | `database/sql/setup_emergency_db.sql` | None |
| 3 | `collect_dataset.py` | `dataset/collect_dataset.py` | Added `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` |
| 4 | `extract_landmarks_balanced.py` | `dataset/extract_landmarks_balanced.py` | `OUTPUT_CSV` path changed to `dataset/landmarks_v1.csv` |
| 5 | `audit_dataset.py` | `dataset/audit_dataset.py` | Added `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` |
| 6 | `extracted_landmarks.csv` | `dataset/extracted_landmarks.csv` | `CSV_PATH` updated in `train_mlp.py`, `collision_audit.py` |
| 7 | `landmarks_v1.csv` | `dataset/landmarks_v1.csv` | Paths updated in `run_audit.py`, `optimize_mlp.py`, `extract_landmarks_balanced.py` |
| 8 | `train_mlp.py` | `training/train_mlp.py` | `CSV_PATH` updated to `dataset/extracted_landmarks.csv` |
| 9 | `collision_audit.py` | `training/collision_audit.py` | `CSV_PATH` updated to `dataset/extracted_landmarks.csv` |
| 10 | `optimize_mlp.py` | `training/optimize_mlp.py` | Argparse default updated to `dataset/landmarks_v1.csv` |
| 11 | `run_audit.py` | `training/run_audit.py` | `pd.read_csv("landmarks_v1.csv")` → `pd.read_csv("dataset/landmarks_v1.csv")` |
| 12 | `test_real_time.py` | `training/test_real_time.py` | Added `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` |
| 13 | `test_load.py` | `training/test_load.py` | Added `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` |
| 14 | `emotion_detection.py` | `emotion/emotion_detection.py` | `from emotion_detection import` → `from emotion.emotion_detection import` in `predict_live.py` |
| 15 | `auth_gui.py` | `frontend/auth_gui.py` | `from supabase_client import` → `from database.supabase_client import` |
| 16 | `hand_detection.py` | `recognition/hand_detection.py` | None |
| 17 | `utils.py` | `utils/__init__.py` | Renamed in-place to package init |
| 18 | `.env.example` | `config/.env.example` | None |
| 19 | `README.md` | `docs/README.md` | None |
| 20 | `artifacts/gh_landmarks_comparison.png` | `assets/artifacts/gh_landmarks_comparison.png` | None |
| 21 | `test_logs/` | `logs/test/` | None |
| 22 | `filename.py` | `archive/filename.py` | None |

## New Files (No Movement)

| File | Purpose |
|------|---------|
| `emergency/__init__.py` | New package placeholder |
| `speech/__init__.py` | New package placeholder |
| `translations/__init__.py` | New package placeholder |
| `utils/__init__.py` | `utils.py` renamed to `utils/__init__.py` |
| Database `__init__.py` files | Added to enable package imports |
| Dataset `__init__.py` files | Added to enable package imports |
| Training `__init__.py` files | Added to enable package imports |

## Import Changes Summary

| File | Old Import | New Import |
|------|-----------|------------|
| `predict_live.py` | `from emotion_detection import` | `from emotion.emotion_detection import` |
| `predict_live.py` | `from supabase_client import` | `from database.supabase_client import` |
| `frontend/auth_gui.py` | `from supabase_client import` | `from database.supabase_client import` |

## CSV Path Changes

| File | Old Path | New Path |
|------|----------|----------|
| `training/train_mlp.py` | `extracted_landmarks.csv` | `dataset/extracted_landmarks.csv` |
| `training/collision_audit.py` | `extracted_landmarks.csv` | `dataset/extracted_landmarks.csv` |
| `training/run_audit.py` | `landmarks_v1.csv` | `dataset/landmarks_v1.csv` |
| `training/optimize_mlp.py` | `landmarks_v1.csv` | `dataset/landmarks_v1.csv` |
| `dataset/extract_landmarks_balanced.py` | `landmarks_v1.csv` | `dataset/landmarks_v1.csv` |
