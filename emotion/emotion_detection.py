# -*- coding: utf-8 -*-
"""
Emotion Detection Module — VisionSpeak
=======================================
Detects facial emotions from webcam frames using the `fer` library
(FER2013-based CNN) and provides a thread-safe, non-blocking interface
for the main application loop.

Threading Strategy:
  - A background daemon thread runs emotion detection on every Nth frame
    (default: every 5th frame) to avoid impacting hand-tracking FPS.
  - The main thread calls detect(frame) every frame, but the background
    thread only processes the latest frame when it's ready.
  - get_emotion() returns the cached result instantly (lock-protected).

Error Handling:
  - If `fer` or TensorFlow fails to import, self.enabled = False and
    detect() / get_emotion() return safe defaults ("Neutral", 0.0).
  - The rest of the application continues running normally.

Usage:
    from emotion_detection import EmotionDetector

    detector = EmotionDetector(detect_every_n=5)
    if detector.enabled:
        detector.detect(frame)                          # submit frame (non-blocking)
        emotion, confidence = detector.get_emotion()    # read cached result
    detector.shutdown()                                 # cleanup on exit
"""

import threading
import time
from collections import deque, Counter

# ── Attempt to import the fer library ────────────────────────────────────────
# If fer or its TensorFlow backend is missing, we gracefully disable
# emotion detection rather than crashing the entire application.
try:
    import os
    import warnings

    # Suppress noisy TensorFlow / Keras logs during fer import
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")

    from fer.fer import FER
    FER_AVAILABLE = True
    print("[OK] fer library loaded successfully.")
    
    # Configure TensorFlow to use GPU if available
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] TensorFlow GPU support active. Found GPUs: {len(gpus)}")
        else:
            print("[INFO] TensorFlow GPU not detected. Running FER on CPU.")
    except Exception as tfe:
        print(f"[WARN] Could not configure TensorFlow GPU settings: {tfe}")
except ImportError as e:
    FER_AVAILABLE = False
    print(f"[WARN] fer library not available: {e}")
    print("       Emotion detection will be disabled.")
    print("       Install with: pip install fer")
except Exception as e:
    FER_AVAILABLE = False
    print(f"[WARN] fer library failed to initialize: {e}")
    print("       Emotion detection will be disabled.")


class EmotionDetector:
    """
    Thread-safe facial emotion detector.

    Runs the fer library's CNN classifier in a background thread to avoid
    blocking the main hand-tracking loop. Only processes every Nth frame
    to keep CPU/GPU usage minimal.

    Attributes:
        enabled (bool): True if fer loaded successfully and detector is active.

    Detected Emotions (6 categories):
        Happy, Sad, Angry, Surprise, Neutral, Fear
    """

    # ── Map fer's 7 emotion labels → our 6 categories (Title Case) ───────
    # "disgust" is merged into "Angry" because:
    #   1. Disgust is rarely the dominant detected emotion
    #   2. The facial expression is visually similar to anger
    EMOTION_MAP = {
        "angry": "Angry",
        "disgust": "Angry",     # merged: disgust → Angry
        "fear": "Fear",
        "happy": "Happy",
        "sad": "Sad",
        "surprise": "Surprise",
        "neutral": "Neutral",
    }

    def __init__(self, detect_every_n=5):
        """
        Initialize the emotion detector.

        Args:
            detect_every_n (int): Only process every Nth frame submitted
                                  via detect(). Default 5 means at 30 FPS,
                                  emotion updates ~6 times per second.
        """
        self.enabled = False
        self._detect_every_n = detect_every_n
        self._frame_counter = 0

        # Cached result — returned by get_emotion() on every call
        self._current_emotion = "Neutral"
        self._current_confidence = 0.0
        self._result_lock = threading.Lock()

        # Frame submission — main thread writes, worker thread reads
        self._pending_frame = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()  # signals worker that a new frame is ready

        # Shutdown signal
        self._shutdown_event = threading.Event()

        # ── Temporal Smoothing and Stability Parameters ──────────────────────
        self._history_size = 20
        self._emotion_history = deque(maxlen=self._history_size)
        # Pre-populate history with "Neutral" to ensure smooth initialization
        for _ in range(self._history_size):
            self._emotion_history.append("Neutral")

        self._stable_emotion = "Neutral"
        self._smoothed_confidence = 0.0
        self._last_emotion_change_time = time.time()
        self._ema_alpha = 0.3
        self._min_hold_time = 1.0  # seconds

        # ── Try to create the fer detector ────────────────────────────────
        if not FER_AVAILABLE:
            print("[EMOTION] Detector disabled (fer not available).")
            return

        try:
            # Use default Haar Cascade face detection (mtcnn=False)
            # This is faster than MTCNN and sufficient for webcam use
            self._detector = FER(mtcnn=False)
            self.enabled = True
            print("[OK] Emotion detector initialized (Haar Cascade + FER2013 CNN).")
        except Exception as e:
            print(f"[WARN] Failed to create FER detector: {e}")
            print("       Emotion detection will be disabled.")
            return

        # ── Launch background worker thread ───────────────────────────────
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="EmotionWorkerThread",
            daemon=True,  # auto-terminates when main program exits
        )
        self._worker_thread.start()

    def _worker_loop(self):
        """
        Background worker: waits for new frames and runs emotion detection.

        This loop runs continuously until shutdown() is called. It blocks
        on self._frame_event, so it uses zero CPU when no new frames are
        submitted. When a frame arrives, it runs fer's detect_emotions()
        and caches the result.
        """
        while not self._shutdown_event.is_set():
            # Wait for a new frame (with timeout to check shutdown periodically)
            signaled = self._frame_event.wait(timeout=0.5)
            if not signaled:
                continue  # timeout — loop back and check shutdown flag

            # Clear the event flag and grab the latest frame
            self._frame_event.clear()
            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            # ── Run emotion detection ─────────────────────────────────────
            try:
                results = self._detector.detect_emotions(frame)

                if results and len(results) > 0:
                    # Take the first detected face (closest/largest)
                    emotions_dict = results[0]["emotions"]

                    # ── Merge disgust into angry before finding top emotion ──
                    merged = {}
                    for fer_label, score in emotions_dict.items():
                        our_label = self.EMOTION_MAP.get(fer_label, "Neutral")
                        merged[our_label] = merged.get(our_label, 0.0) + score

                    # Find the emotion with the highest merged score
                    top_emotion = max(merged, key=merged.get)
                    top_confidence = merged[top_emotion]

                    # ── Requirement 2: Confidence-based filtering ──
                    # Ignore predictions below 50% confidence (treat as Neutral)
                    if top_confidence < 0.50:
                        top_emotion = "Neutral"

                    # ── Requirement 3: Smooth confidence with EMA (alpha = 0.3) ──
                    self._smoothed_confidence = (self._ema_alpha * top_confidence) + ((1.0 - self._ema_alpha) * self._smoothed_confidence)

                    # ── Requirement 1: Add to rolling history ──
                    self._emotion_history.append(top_emotion)
                else:
                    # No face detected — keep previous result but decay confidence
                    # This prevents the HUD from flickering when the face is
                    # temporarily occluded (e.g., hand covering face while signing)
                    self._smoothed_confidence *= 0.85  # gradual decay
                    if self._smoothed_confidence < 0.05:
                        self._smoothed_confidence = 0.0
                    
                    # Add Neutral to history to gradually return to baseline state
                    self._emotion_history.append("Neutral")

                # ── Step 5: Majority Voting & Hold Time ──
                counter = Counter(self._emotion_history)
                majority_emotion, count = counter.most_common(1)[0]
                majority_ratio = count / len(self._emotion_history)

                # Only change displayed stable emotion if majority exceeds 60%,
                # it is a new emotion, and the minimum hold time of 1 second has passed.
                now = time.time()
                if (majority_ratio > 0.60
                        and majority_emotion != self._stable_emotion
                        and (now - self._last_emotion_change_time) >= self._min_hold_time):
                    self._stable_emotion = majority_emotion
                    self._last_emotion_change_time = now
                    print(f"[EMOTION] Stable emotion changed to: {self._stable_emotion} ({majority_ratio*100:.1f}% majority)")

                # Update cached result (thread-safe)
                with self._result_lock:
                    self._current_emotion = self._stable_emotion
                    self._current_confidence = round(self._smoothed_confidence, 3)

            except Exception as e:
                # Never crash the background thread — log and continue
                print(f"[EMOTION] Detection error (non-fatal): {e}")

    def detect(self, frame):
        """
        Submit a frame for emotion detection (called from main loop).

        This method is non-blocking. It increments a frame counter and
        only submits every Nth frame to the background worker thread.
        The actual detection runs asynchronously.

        Args:
            frame: BGR numpy array from OpenCV (the webcam frame).
        """
        if not self.enabled:
            return

        # Only process every Nth frame to maintain FPS
        self._frame_counter += 1
        if self._frame_counter % self._detect_every_n != 0:
            return

        # Submit a copy of the frame to the worker thread
        # (copy is important — the main thread will modify the original frame
        #  for drawing, and we don't want the worker to read corrupted data)
        with self._frame_lock:
            self._pending_frame = frame.copy()

        # Signal the worker that a new frame is ready
        self._frame_event.set()

    def get_emotion(self):
        """
        Get the current detected emotion (called from main loop).

        Returns instantly with the cached result from the last successful
        detection. Thread-safe.

        Returns:
            tuple: (emotion_label: str, confidence: float)
                   e.g. ("Happy", 0.85) or ("Neutral", 0.0) if no face
        """
        with self._result_lock:
            return self._current_emotion, self._current_confidence

    def shutdown(self):
        """
        Signal the background worker thread to stop and wait for it to finish.
        Call this during application cleanup (before sys.exit).
        """
        if not self.enabled:
            return

        self._shutdown_event.set()
        self._frame_event.set()  # unblock the worker if it's waiting
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        print("[OK] Emotion detector shut down.")
