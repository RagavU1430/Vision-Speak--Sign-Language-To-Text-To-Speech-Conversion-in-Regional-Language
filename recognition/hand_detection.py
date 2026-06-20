import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import time
from utils import suppress_c_stderr

os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np

# Lazy-init MediaPipe so importing the module doesn't trigger CPU-heavy init
_mp_hands = None
_hands = None
_mp_draw = None
_landmark_spec = None
_connection_spec = None
_smoother = None


def _init_mediapipe():
    global _mp_hands, _hands, _mp_draw, _landmark_spec, _connection_spec, _smoother
    if _hands is not None:
        return

    with suppress_c_stderr():
        import mediapipe as mp

        warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

        _mp_hands = mp.solutions.hands
        _hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _hands.process(dummy_frame)

        _mp_draw = mp.solutions.drawing_utils

    _landmark_spec = _mp_draw.DrawingSpec(color=(255, 255, 0), thickness=-1, circle_radius=5)
    _connection_spec = _mp_draw.DrawingSpec(color=(180, 105, 255), thickness=3, circle_radius=2)
    _smoother = HandSmoother()


class OneEuroFilter:
    def __init__(self, min_cutoff=2.0, beta=0.1, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filt = None
        self.dx_filt = None
        self.t_prev = None

    def __call__(self, t, x):
        if self.x_filt is None:
            self.x_filt = x.copy()
            self.dx_filt = np.zeros_like(x, dtype=np.float32)
            self.t_prev = t
            return self.x_filt

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_filt

        dx = (x - self.x_filt) / dt
        d_alpha = self._alpha(dt, self.d_cutoff)
        self.dx_filt = d_alpha * dx + (1.0 - d_alpha) * self.dx_filt

        cutoff = self.min_cutoff + self.beta * np.abs(self.dx_filt)
        alpha = self._alpha(dt, cutoff)
        self.x_filt = alpha * x + (1.0 - alpha) * self.x_filt
        self.t_prev = t
        return self.x_filt

    def _alpha(self, dt, cutoff):
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)


class HandSmoother:
    def __init__(self, min_cutoff=2.0, beta=0.1):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.tracked_hands = []

    def smooth(self, current_hands):
        now = time.time()
        if not current_hands:
            self.tracked_hands = []
            return current_hands

        new_tracked_hands = []
        for curr_hand in current_hands:
            curr_wrist = curr_hand.landmark[0]
            best_match_idx = -1
            min_dist = 0.15

            for idx, tracked in enumerate(self.tracked_hands):
                prev_wrist_x, prev_wrist_y, _ = tracked["last_wrist"]
                dist = ((curr_wrist.x - prev_wrist_x)**2 + (curr_wrist.y - prev_wrist_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx

            curr_coords = np.array([[lm.x, lm.y, lm.z] for lm in curr_hand.landmark], dtype=np.float32)

            if best_match_idx != -1:
                tracked = self.tracked_hands[best_match_idx]
                filt = tracked["filter"]
                smoothed = filt(now, curr_coords)
            else:
                filt = OneEuroFilter(min_cutoff=self.min_cutoff, beta=self.beta)
                smoothed = filt(now, curr_coords)

            for i, lm in enumerate(curr_hand.landmark):
                lm.x = float(smoothed[i, 0])
                lm.y = float(smoothed[i, 1])
                lm.z = float(smoothed[i, 2])

            new_tracked_hands.append({
                "filter": filt,
                "last_wrist": (curr_hand.landmark[0].x, curr_hand.landmark[0].y, curr_hand.landmark[0].z)
            })

        self.tracked_hands = new_tracked_hands
        return current_hands


def get_hands_instance():
    _init_mediapipe()
    return _hands


def get_drawing_utils():
    _init_mediapipe()
    return _mp_draw, _landmark_spec, _connection_spec


def get_smoother():
    _init_mediapipe()
    return _smoother


def draw_hud(frame, fps, hand_count):
    overlay = frame.copy()

    x, y, w, h = 20, 20, 260, 105

    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)

    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "HUD: HAND MONITOR v1.0", (x + 12, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, "FPS:", (x + 12, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(fps)}", (x + 65, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, "Hands:", (x + 12, y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    hand_color = (180, 105, 255) if hand_count > 0 else (100, 100, 255)
    cv2.putText(frame, f"{hand_count}", (x + 80, y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2, cv2.LINE_AA)


def main():
    _init_mediapipe()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting hand detection. Press ESC to exit.")

    frame_count = 0
    fps_start_time = time.time()
    display_fps = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = _hands.process(rgb)

        hand_count = 0
        if result.multi_hand_landmarks:
            hand_count = len(result.multi_hand_landmarks)
            result.multi_hand_landmarks = _smoother.smooth(result.multi_hand_landmarks)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                _mp_draw.draw_landmarks(
                    frame,
                    hand,
                    _mp_hands.HAND_CONNECTIONS,
                    _landmark_spec,
                    _connection_spec
                )

        frame_count += 1
        c_time = time.time()
        elapsed_time = c_time - fps_start_time
        if elapsed_time >= 0.5:
            display_fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = c_time

        draw_hud(frame, display_fps, hand_count)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
