# 🚀 VisionSpeak
### AI-Powered Multilingual Sign Language Communication & Emotion Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/MediaPipe-Hand_Tracking-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Keras%20/%20TensorFlow-Deep%20Learning-red?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Supabase-Database-success?style=for-the-badge&logo=supabase">
  <img src="https://img.shields.io/badge/FER-Facial_Emotion_AI-red?style=for-the-badge">
</p>

---

## 🌟 Overview

VisionSpeak is an intelligent real-time communication system designed to bridge the communication gap between hearing-impaired individuals and the general public.

The system recognizes hand gestures using AI, converts them into meaningful text, translates them into multiple languages, analyzes facial emotions, generates speech output, and stores interaction history in the cloud.

---

## 🎯 Problem Statement

Millions of hearing and speech-impaired individuals face communication challenges in everyday life.

VisionSpeak aims to provide:

✅ **Real-Time Sign Recognition** (optimized with Keras Deep MLP)

✅ **Emotion-Aware Communication** (real-time face analysis)

✅ **Multilingual Translation** (English / Tamil translation)

✅ **Speech Generation** (integrated regional text-to-speech)

✅ **Emergency Assistance Support** (keyword matching & auto-speak)

✅ **Cloud-Based Interaction History** (connected to Supabase)

---

# 🏗️ System Architecture

```text
Webcam
   │
   ├── MediaPipe Hand Tracking (21 landmarks)
   │           │
   │           ▼
   │      Landmark Extract & Norm
   │           │
   │           ▼
   │     Feature Engineering (99 normalized features)
   │           │
   │           ▼
   │     Deep Keras MLP Model (Dense 256->128->64->32)
   │           │
   │           ▼
   │     Confidence & Gap Filter (Gap >= 0.20)
   │           │
   │           ▼
   │     Majority Voting stabilization (Agreement >= 80%)
   │           │
   │           ▼
   │    Sign Recognition
   │
   └── FER (Facial Expression Recognition)
               │
               ▼
       Emotion Detection
               │
               ▼

      Combined Intelligence
               │
               ▼

      English/Tamil Translation
               │
               ▼

         Text To Speech
               │
               ▼

      Supabase Cloud Storage
```

---

# ✨ Key Features

## 🤟 Real-Time Sign Language Recognition
- **MediaPipe Hand landmarking**: Real-time 21 landmark extraction.
- **Enhanced Feature Mode**: Generates 99 coordinates, distances, angles, and ratios.
- **Deep MLP Classifier**: Deep Keras Sequential Neural Network model for sign classification.
- **Temporal Stabilization**: `MajorityVoteSystem` stabilizes predictions over a sliding 20-frame window.
- **Confidence-Gap Validation**: Pre-filters ambiguous frames by requiring `confidence >= 90%` and a top-1 vs top-2 probability gap of at least `20%`.

---

## 😀 Facial Emotion Detection
Using FER (Facial Expression Recognition):
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😐 Neutral

Emotion is displayed on the OpenCV HUD panel alongside recognized signs.

---

## 🌐 Multilingual Translation
Supported Languages:
- 🇬🇧 English
- 🇮🇳 Tamil (தமிழ்)

---

## 🔊 Intelligent Text-To-Speech
Converts recognized signs into speech:
- English Voice Output
- Tamil Voice Output
- Real-Time Playback

---

## 🚨 Emergency Assistance Mode
Emergency keywords are automatically detected and push priority database alerts:
- Examples: `HELP`, `EMERGENCY`, `DOCTOR`, `HOSPITAL`
- Triggers priority speech and cloud event logging.

---

## ☁️ Supabase Cloud Integration
Stores recognized text, translations, confidence scores, emotions, and timestamps for complete communication logging.

---

# 🧠 AI Models Used

## Hand Gesture Recognition (Upgraded)
- **Model**: Keras `Sequential` Neural Network
- **Architecture**:
  - `Input(99 features)`
  - `Dense(256, ReLU) -> BatchNormalization -> Dropout(0.3)`
  - `Dense(128, ReLU) -> BatchNormalization -> Dropout(0.3)`
  - `Dense(64, ReLU) -> BatchNormalization -> Dropout(0.2)`
  - `Dense(32, ReLU)`
  - `Dense(num_classes, Softmax)`
- **Training Configuration**:
  - `Adam` optimizer (learning_rate=0.001)
  - `EarlyStopping` (patience=10, restoring best weights)
  - `ReduceLROnPlateau` (factor=0.5, patience=5)
  - Horizontal bar plot of confused letter pairs

---

# 📂 Project Structure

```text
VisionSpeak/
│
├── models/
│   ├── mlp_model.keras       (Native Keras Sequential model)
│   ├── mlp_model.pkl         (Backward-compatible joblib wrapper)
│   ├── scaler.pkl            (Feature standard scaler)
│   ├── label_encoder.pkl     (Class label mapping)
│   ├── training_curves.png   (Accuracy/Loss plot)
│   ├── confusion_matrix.png  (Evaluation heatmap)
│   └── dataset_stats.json    (JSON stats of raw dataset)
│
├── dataset/
│   └── extracted_landmarks.csv  (Extracted landmark features)
│
├── training/
│   ├── train_mlp.py          (MLP Keras training pipeline)
│   └── run_audit.py          (Runs collision checks)
│
├── utils/
│   └── __init__.py           (Feature engineering & KerasMLPWrapper)
│
├── predict_live.py           (Live Webcam Application)
│
└── requirements.txt          (Project requirements)
```

---

# 📈 Performance

| Metric | Value |
|----------|---------|
| Recognition FPS | 25-30 FPS |
| Emotion Detection | Real-Time |
| Translation Speed | < 1 sec |
| Speech Response | Instant |
| Database Logging | Live |

---

# 👨‍💻 Developed By

**Ragav**

Artificial Intelligence & Data Science Student
