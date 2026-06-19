# Potentially Unused Files — Analysis & Recommendations

This report identifies files that may no longer be in active use after the reorganization. **No files were deleted.** Each entry includes reasoning and a recommendation.

| File | Reason | Recommendation |
|------|--------|----------------|
| `models/label_encoder_enhanced.pkl` | Legacy CNN model encoder. Not referenced by any code in the project. No imports or pickle loads target this file. | Remove if the CNN pipeline is confirmed dead. Otherwise move to `archive/`. |
| `models/model_metadata.pkl` | Legacy metadata blob. Not referenced by any code. Appears to be from a prior training pipeline. | Remove or archive alongside the encoder above. |
| `models/model_comparison_report.md` | Informational report from a past model comparison. Not consumed by any script. | Keep for reference or move to `docs/`. |
| `archive/filename.py` | Contains only `print("Hello, world!")`. Clearly a placeholder/test file with no value. | Delete. |
| `training/test_load.py` | Loads a file at `models/sign_cnn.h5` which does not exist (no CNN model in the project). The script cannot execute as-is. | Either remove or update to reference an existing model, if the test is still needed. |
