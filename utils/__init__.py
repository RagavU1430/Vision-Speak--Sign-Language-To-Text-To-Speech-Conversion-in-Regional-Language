import os
import sys
import contextlib
import warnings

# ==================== SUPPRESS TF/MP C++ NOISE ====================
# Must be set before any TensorFlow or MediaPipe import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'


# Context manager to temporarily redirect low-level OS stderr (file descriptor 2) to devnull.
# This successfully suppresses C++ standard error prints from MediaPipe and TensorFlow.
@contextlib.contextmanager
def suppress_c_stderr():
    """Redirect file descriptor 2 (stderr) to devnull to silence low-level C++ logging."""
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError):
        yield
        return
    saved_stderr_fd = os.dup(stderr_fd)
    devnull = open(os.devnull, 'w')
    try:
        os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        devnull.close()


# Import heavy dependencies inside suppress context so C++ noise is silenced
with suppress_c_stderr():
    import numpy as np
    import cv2
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


class SkeletonGenerator:
    """
    Converts hand landmarks into a normalized 400×400 skeleton canvas that
    visually matches the training dataset format exactly:
      - White (255, 255, 255) background
      - Green (0, 255, 0) landmark dots with circle_radius=4
      - Green (0, 200, 0) connections with thickness=3
      - Hand centered and scaled to occupy ~75% of canvas

    The bounding box normalization ensures consistent CNN input regardless of
    where the hand appears in the webcam frame or how close/far it is.
    """

    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Enhanced drawing specs — thicker lines and larger dots for visual clarity
        # These match the training dataset's visual style exactly
        self.landmark_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=-1, circle_radius=4
        )
        self.connection_spec = self.mp_draw.DrawingSpec(
            color=(0, 200, 0), thickness=3, circle_radius=2
        )

    def generate(self, landmarks_list, frame_w, frame_h):
        """
        Transforms landmarks (list of (x, y, z) tuples) into a normalized skeleton canvas.

        Pipeline:
          1. Convert normalized coords → pixel coords using frame dimensions
          2. Compute tight bounding box around all 21 landmarks
          3. Add padding (20px) to prevent edge clipping
          4. Calculate uniform scale factor to fit hand into 300px (75% of 400)
          5. Translate hand center to canvas center (200, 200)
          6. Draw skeleton with dataset-matching green style

        Returns: numpy.ndarray (400, 400, 3) uint8 BGR image
        """
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Step 1: Bounding box in webcam pixel coordinates
        px_x = [pt[0] * frame_w for pt in landmarks_list]
        px_y = [pt[1] * frame_h for pt in landmarks_list]

        min_x, max_x = min(px_x), max(px_x)
        min_y, max_y = min(px_y), max(px_y)

        hand_w = max_x - min_x
        hand_h = max_y - min_y

        # Step 2: Padding for aspect ratio preservation
        padding = 20
        padded_w = hand_w + 2 * padding
        padded_h = hand_h + 2 * padding

        # Step 3: Uniform scale to occupy 75% of canvas (300 / max_padded_dimension)
        target_size = 300.0
        max_dim = max(padded_w, padded_h)
        scale = target_size / max_dim if max_dim > 0 else 1.0

        # Step 4: Hand center in pixel space
        center_x = min_x + hand_w / 2.0
        center_y = min_y + hand_h / 2.0

        # Step 5: Project each landmark onto canvas coordinates
        normalized_landmarks = landmark_pb2.NormalizedLandmarkList()
        for pt in landmarks_list:
            canvas_x = (pt[0] * frame_w - center_x) * scale + (self.width / 2.0)
            canvas_y = (pt[1] * frame_h - center_y) * scale + (self.height / 2.0)
            canvas_z = pt[2] * frame_w * scale

            new_lm = normalized_landmarks.landmark.add()
            new_lm.x = canvas_x / float(self.width)
            new_lm.y = canvas_y / float(self.height)
            new_lm.z = canvas_z / float(self.width)

        # Step 6: Draw skeleton
        self.mp_draw.draw_landmarks(
            canvas,
            normalized_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.landmark_spec,
            self.connection_spec,
        )
        return canvas


def extract_enhanced_features(hand_landmarks) -> np.ndarray:
    """
    Extracts 99 enhanced features from hand landmarks:
      1. Translates all landmarks relative to the wrist (landmark 0).
      2. Normalizes coordinates by dividing by wrist-to-middle-MCP (landmark 9) distance.
      3. Calculates:
         - Finger lengths (5 features)
         - Joint angles (14 angles)
         - Fingertip distances to wrist (5 features)
         - Pairwise fingertip distances (10 features)
         - Palm width (1 feature)
         - Palm height (1 feature)
         - Normalized coordinate vector (63 features)
    Returns:
      A flat 1D numpy array of size 99.
    """
    # Handle both MediaPipe landmark objects and raw arrays/lists of coords
    if hasattr(hand_landmarks, 'landmark'):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    else:
        coords = np.array(hand_landmarks, dtype=np.float32).reshape(21, 3)

    # 1. Translation: wrist is landmark 0
    wrist = coords[0]
    translated = coords - wrist  # shape: (21, 3)

    # 2. Scale normalization
    # Distance between wrist (0) and middle MCP (9)
    hand_scale = np.linalg.norm(translated[9])
    if hand_scale < 1e-6:
        hand_scale = 1.0
    normalized = translated / hand_scale

    # 3. Calculate Finger Lengths
    # Cumulative distances along the joints of each finger
    # Wrist (0), Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    def finger_len(indices):
        return sum(np.linalg.norm(normalized[indices[i]] - normalized[indices[i+1]]) for i in range(len(indices)-1))

    lengths = [
        finger_len([0, 1, 2, 3, 4]),      # Thumb
        finger_len([0, 5, 6, 7, 8]),      # Index
        finger_len([0, 9, 10, 11, 12]),   # Middle
        finger_len([0, 13, 14, 15, 16]),  # Ring
        finger_len([0, 17, 18, 19, 20]),  # Pinky
    ]

    # 4. Joint angles
    # For three points A, B, C, angle at B is between BA and BC.
    # Angle is arccos of dot product of normalized BA and BC vectors.
    def joint_angle(idx_a, idx_b, idx_c):
        v1 = normalized[idx_a] - normalized[idx_b]
        v2 = normalized[idx_c] - normalized[idx_b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    angles = [
        # Thumb
        joint_angle(1, 2, 3), joint_angle(2, 3, 4),
        # Index
        joint_angle(5, 6, 7), joint_angle(6, 7, 8),
        # Middle
        joint_angle(9, 10, 11), joint_angle(10, 11, 12),
        # Ring
        joint_angle(13, 14, 15), joint_angle(14, 15, 16),
        # Pinky
        joint_angle(17, 18, 19), joint_angle(18, 19, 20),
        # MCP angles (angle between wrist direction and first segment)
        joint_angle(0, 5, 6),
        joint_angle(0, 9, 10),
        joint_angle(0, 13, 14),
        joint_angle(0, 17, 18)
    ]

    # 5. Fingertip distances to wrist
    # Tips: 4, 8, 12, 16, 20
    tips = [4, 8, 12, 16, 20]
    tip_to_wrist = [np.linalg.norm(normalized[tip]) for tip in tips]

    # 6. Pairwise fingertip distances (10 combinations)
    tip_to_tip = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            tip_to_tip.append(np.linalg.norm(normalized[tips[i]] - normalized[tips[j]]))

    # 7. Palm width and height
    palm_width = np.linalg.norm(normalized[5] - normalized[17])
    palm_height = np.linalg.norm(normalized[0] - normalized[9])  # always 1.0 due to scale division

    # Concatenate all features into a single flat vector
    features = np.concatenate([
        normalized.flatten(),       # 63 features
        lengths,                    # 5 features
        angles,                     # 14 features
        tip_to_wrist,               # 5 features
        tip_to_tip,                 # 10 features
        [palm_width, palm_height]   # 2 features
    ])  # Total 99 features

    return features


def extract_raw_normalized_landmarks(hand_landmarks) -> np.ndarray:
    """
    Extracts 63 raw normalized features from hand landmarks:
      1. Translates all landmarks relative to the wrist (landmark 0).
      2. Normalizes coordinates by dividing by wrist-to-middle-MCP (landmark 9) distance.
    Returns:
      A flat 1D numpy array of size 63.
    """
    # Handle both MediaPipe landmark objects and raw arrays/lists of coords
    if hasattr(hand_landmarks, 'landmark'):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    else:
        coords = np.array(hand_landmarks, dtype=np.float32).reshape(21, 3)

    # 1. Translation: wrist is landmark 0
    wrist = coords[0]
    translated = coords - wrist  # shape: (21, 3)

    # 2. Scale normalization
    # Distance between wrist (0) and middle MCP (9)
    hand_scale = np.linalg.norm(translated[9])
    if hand_scale < 1e-6:
        hand_scale = 1.0
    normalized = translated / hand_scale

    return normalized.flatten()


def extract_enhanced_features_v2(hand_landmarks) -> np.ndarray:
    """
    Extracts ~140 enhanced features — superset of v1 (99) plus geometric features:
      - 99 v1 features (normalized coords, finger lengths, joint angles, etc.)
      - Finger extension states (5 binary)
      - Thumb opposition distance (1)
      - Palm normal vector (3)
      - Finger-tip pairwise spread (max spread between adjacent tips, 4)
      - Finger curl angles (additional joint angles: MCP-PIP, PIP-DIP for each finger, 10)
      - Finger cross-ratios (length ratios: index/middle, ring/middle, etc., 4)
      - Hand openness (mean tip-to-tip distance, 1)
      - Thumb-index web angle (1)
      - Wrist-to-palm-center direction (3)
      - Finger-base spread (distances between MCPs, 6)
      - Total: 99 + 38 = 137
    """
    # Start with all v1 features
    base = extract_enhanced_features(hand_landmarks)

    # Get the normalized coordinates for additional geometric computation
    if hasattr(hand_landmarks, 'landmark'):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    else:
        coords = np.array(hand_landmarks, dtype=np.float32).reshape(21, 3)

    wrist = coords[0]
    translated = coords - wrist
    hand_scale = np.linalg.norm(translated[9])
    if hand_scale < 1e-6:
        hand_scale = 1.0
    n = translated / hand_scale

    # Finger extension states: tip further from wrist than MCP
    mcp_indices = [1, 5, 9, 13, 17]   # thumb, index, middle, ring, pinky MCPs
    tip_indices = [4, 8, 12, 16, 20]
    def is_extended(mcp_idx, tip_idx):
        mcp_dist = np.linalg.norm(n[mcp_idx])
        tip_dist = np.linalg.norm(n[tip_idx])
        return 1.0 if tip_dist > mcp_dist * 1.3 else 0.0
    ext_states = [is_extended(m, t) for m, t in zip(mcp_indices, tip_indices)]

    # Thumb opposition: distance between thumb tip (4) and pinky MCP (17)
    thumb_opposition = np.linalg.norm(n[4] - n[17])

    # Palm normal: cross product of vectors from wrist to index MCP and wrist to pinky MCP
    v1 = n[5] - n[0]
    v2 = n[17] - n[0]
    palm_normal = np.cross(v1, v2)
    norm = np.linalg.norm(palm_normal)
    if norm > 1e-6:
        palm_normal = palm_normal / norm

    # Finger-tip pairwise spread between adjacent fingertips
    adjacent_tip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    finger_spread = [np.linalg.norm(n[a] - n[b]) for a, b in adjacent_tip_pairs]

    # Finger curl angles: MCP-PIP-DIP for each finger
    def curl_angle(mcp, pip, dip):
        """Angle at PIP joint between MCP→PIP and PIP→DIP vectors."""
        v1 = n[mcp] - n[pip]
        v2 = n[dip] - n[pip]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_a = np.dot(v1, v2) / (n1 * n2)
        return np.arccos(np.clip(cos_a, -1.0, 1.0))

    # Finger joints: (MCP, PIP, DIP, TIP) for each finger
    finger_joints = [
        (1, 2, 3, 4),     # thumb
        (5, 6, 7, 8),     # index
        (9, 10, 11, 12),  # middle
        (13, 14, 15, 16), # ring
        (17, 18, 19, 20), # pinky
    ]
    curl_angles = []
    for mcp, pip, dip, tip in finger_joints:
        curl_angles.append(curl_angle(mcp, pip, dip))

    # Finger cross-ratios
    def finger_length(mcp, tip):
        return np.linalg.norm(n[tip] - n[mcp])
    lengths = [
        finger_length(1, 4),    # thumb
        finger_length(5, 8),    # index
        finger_length(9, 12),   # middle
        finger_length(13, 16),  # ring
        finger_length(17, 20),  # pinky
    ]
    ratios = [
        lengths[0] / max(lengths[1], 1e-6),  # thumb/index
        lengths[1] / max(lengths[2], 1e-6),  # index/middle
        lengths[3] / max(lengths[2], 1e-6),  # ring/middle
        lengths[4] / max(lengths[2], 1e-6),  # pinky/middle
    ]

    # Hand openness: mean pairwise tip-to-tip distance
    tips = [4, 8, 12, 16, 20]
    tip_dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            tip_dists.append(np.linalg.norm(n[tips[i]] - n[tips[j]]))
    hand_openness = np.mean(tip_dists) if tip_dists else 0.0

    # Thumb-index web angle: angle between thumb MCP→tip and index MCP→tip
    def vector_angle(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    thumb_vec = n[4] - n[1]
    index_vec = n[8] - n[5]
    thumb_index_angle = vector_angle(thumb_vec, index_vec)

    # Wrist-to-palm-center direction
    palm_center = np.mean([n[5], n[9], n[13], n[17]], axis=0)
    wrist_to_palm = palm_center - n[0]
    wp_norm = np.linalg.norm(wrist_to_palm)
    if wp_norm > 1e-6:
        wrist_to_palm = wrist_to_palm / wp_norm

    # Finger-base spread: pairwise distances between MCPs of adjacent fingers
    mcps = [1, 5, 9, 13, 17]
    base_spread = []
    for i in range(len(mcps)):
        for j in range(i + 1, len(mcps)):
            base_spread.append(np.linalg.norm(n[mcps[i]] - n[mcps[j]]))

    v2_features = np.concatenate([
        ext_states,            # 5
        [thumb_opposition],    # 1
        palm_normal,           # 3
        finger_spread,         # 4
        curl_angles,           # 5
        ratios,                # 4
        [hand_openness],       # 1
        [thumb_index_angle],   # 1
        wrist_to_palm,         # 3
        base_spread,           # 10 (all C(5,2) pairs)
    ])  # = 37

    return np.concatenate([base, v2_features])  # 99 + 37 = 136 features


class KerasMLPWrapper:
    """
    Wraps a Keras model to expose scikit-learn–compatible predict/predict_proba
    methods. This allows predict_live.py to call model.predict_proba() without
    any code changes to the prediction pipeline.
    """

    def __init__(self, keras_model_path: str):
        self.keras_model_path = keras_model_path
        self._model = None  # Lazy-loaded to avoid import at unpickle time

    def _load(self):
        """Lazy-load the Keras model on first use."""
        if self._model is None:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.keras_model_path)
        return self._model

    def predict_proba(self, X):
        """Return class probabilities — shape (n_samples, n_classes)."""
        model = self._load()
        return model.predict(X, verbose=0)

    def predict(self, X):
        """Return predicted class indices — shape (n_samples,)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    @property
    def n_features_in_(self):
        """Return expected input feature count for compatibility checks."""
        model = self._load()
        return model.input_shape[-1]


