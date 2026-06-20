# VisionSpeak — Error Analysis Report

Generated: 2026-06-20

---

## CRITICAL ERRORS

### 1. `training/test_load.py` references non-existent CNN model

- **File:** `training/test_load.py:10`
- **Issue:** Imports `tensorflow` and tries to load `models/sign_cnn.h5`, which does not exist.
- **Impact:** Script crashes immediately on run.
- **Fix:** Delete the file or update to load `models/mlp_model.pkl` with joblib.

### 2. Missing `requirements.txt`

- **Issue:** No `requirements.txt` anywhere in the project. Dependencies cannot be installed with a single command.
- **Impact:** Users must manually discover and install every package.
- **Fix:** Generate `requirements.txt` from actual imports.

### 3. `.env.example` is incomplete

- **File:** `config/.env.example`
- **Issue:** Missing `EMERGENCY_USER_NAME` and `EMERGENCY_USER_AGE` fields that are present in the actual `.env`.
- **Impact:** Anyone copying the example to set up the project will miss required environment variables.
- **Fix:** Add the missing fields to `.env.example`.

---

## MODERATE ISSUES

### 4. Doc/code mismatch: DeepFace vs FER

- **Files:** `docs/README.md` (lines 58, 217, 253), `emotion/emotion_detection.py`
- **Issue:** Documentation states DeepFace is used; actual code uses the `fer` library (`fer.fer.FER`).
- **Impact:** Confusing for contributors; DeepFace and FER have different APIs and capabilities.
- **Fix:** Update docs to reference FER, or switch implementation to DeepFace.

### 5. `temp.mp3` collision in TTS

- **File:** `predict_live.py:1062`
- **Issue:** Tamil TTS writes to a hardcoded `temp.mp3` in the working directory. Multiple instances or crashes before cleanup may cause conflicts or stale files.
- **Fix:** Use `tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)`.

### 6. Double model inference

- **File:** `predict_live.py:3311,3315`
- **Issue:** `model.predict()` and `model.predict_log_proba()` are called sequentially on the same data, doubling computation.
- **Fix:** Call `model.predict_proba()` once and derive both prediction and log probabilities.

---

## MINOR ISSUES

### 7. Supabase health check queries non-existent table

- **File:** `database/supabase_client.py:92`
- **Issue:** Health check targets `_health_check` table, which does not exist. Exception handler silently swallows the 404.
- **Fix:** Use a simpler connectivity check (e.g., `client.auth.get_session()` or a ping).

### 8. `hand_detection.py` runs MediaPipe at import time

- **File:** `recognition/hand_detection.py:182`
- **Issue:** `hands = mp_hands.Hands(...)` executes at module level, causing import-time latency and potential errors if MediaPipe is not installed.
- **Fix:** Lazy-initialize inside a function or class.

### 9. Orphaned/legacy files

- `models/label_encoder_enhanced.pkl` — not referenced by any code
- `models/model_metadata.pkl` — not referenced by any code
- `archive/filename.py` — only contains `print("Hello, world!")`

### 10. Python 3.10+ required but not documented

- **File:** `predict_live.py:603,1333`
- **Issue:** Uses `str | None` and `list | None` union type syntax (PEP 604), which requires Python 3.10+.
- **Impact:** Crashes with `TypeError` on Python ≤3.9.
- **Fix:** Document Python 3.10+ requirement, or use `Optional[str]` / `Optional[list]` for backward compatibility.

---

## CODE QUALITY SUGGESTIONS

- `predict_live.py` is ~3800 lines — consider breaking into smaller modules.
- Heavy use of `print()` for logging — consider `logging` module.
- `emotion/emotion_detection.py` imports `from fer.fer import FER` — the standard import is `from fer import FER`; verify compatibility.
- `recognition/hand_detection.py` has a standalone `__main__` webcam loop; it is not a reusable module despite being in a package.

---

## SUMMARY

| Severity | Count |
|----------|-------|
| Critical | 3 |
| Moderate | 3 |
| Minor | 4 |
| Suggestion | 4 |
