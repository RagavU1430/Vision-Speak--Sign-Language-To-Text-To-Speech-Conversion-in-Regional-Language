# VisionSpeak — Sign Language Recognition Project
# Requires Python 3.10+

SIGN LANG PROJECT
│
├── .env
├── .gitignore
├── requirements.txt                 # Project dependencies
├── predict_live.py                  # Main entry point
├── profile.json                     # Runtime data
│
├── archive/
│   ├── asl_alphabet_train/
│   └── asl_alphabet_test/
│
├── assets/
│   └── artifacts/
│       └── gh_landmarks_comparison.png
│
├── config/
│   └── .env.example
│
├── database/
│   ├── __init__.py
│   ├── supabase_client.py
│   └── sql/
│       ├── setup_emergency_db.sql
│       ├── setup_emotion_column.sql
│       └── setup_user_profiles.sql
│
├── dataset/
│   ├── __init__.py
│   ├── collect_dataset.py
│   ├── extract_landmarks_balanced.py
│   ├── audit_dataset.py
│   ├── extracted_landmarks.csv
│   └── landmarks_v1.csv
│
├── docs/
│   └── README.md
│
├── emergency/
│   └── __init__.py
│
├── emotion/
│   ├── __init__.py
│   └── emotion_detection.py
│
├── frontend/
│   ├── __init__.py
│   └── auth_gui.py
│
├── logs/
│   └── test/
│
├── models/
│   ├── mlp_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── confusion_matrix.png
│   ├── model_comparison_report.md
│   └── collision_report/
│
├── recognition/
│   ├── __init__.py
│   └── hand_detection.py
│
├── speech/
│   └── __init__.py
│
├── training/
│   ├── __init__.py
│   ├── train_mlp.py
│   ├── collision_audit.py
│   ├── optimize_mlp.py
│   ├── run_audit.py
│   ├── test_real_time.py
│   └── test_load.py
│
├── translations/
│   └── __init__.py
│
└── utils/
    └── __init__.py
