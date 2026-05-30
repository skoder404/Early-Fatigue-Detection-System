import cv2
import math
import time
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# CONFIG
# =========================================================
FACE_LANDMARKER_MODEL = "face_landmarker.task"

ML_MODEL_PATH = "fatigue_lightgbm_model_final.pkl"
FEATURE_COLUMNS_PATH = "fatigue_lightgbm_feature_columns_final.pkl"
THRESHOLD_PATH = "fatigue_lightgbm_final_threshold.pkl"

CAMERA_INDEX = 0
WINDOW_NAME = "Fatigue Detection Dashboard - Hybrid 3 State"

CAM_W = 960
CAM_H = 720
PANEL_W = 470
WINDOW_W = CAM_W + PANEL_W
WINDOW_H = CAM_H

# Slightly slower than previous fast version
SUMMARY_INTERVAL = 8.0
MIN_OBSERVATION_BEFORE_DECISION = 4.0

MAX_FACES = 1
DRAW_LANDMARKS = True

# Your working thresholds
EAR_CLOSED_THRESHOLD = 0.23
MAR_YAWN_THRESHOLD = 0.65

BLINK_CONSEC_FRAMES = 2
YAWN_CONSEC_FRAMES = 8

HEAD_NOD_DROP_THRESHOLD = 8.0
HEAD_NOD_CONSEC_FRAMES = 3

# Immediate fatigue override
CONTINUOUS_EYE_CLOSURE_FATIGUE_SEC = 5.0

# Final fatigue score ranges
ALERT_MAX = 35
MILD_FATIGUE_MAX = 65

# Model is only support
MODEL_MILD_THRESHOLD = 0.60
MODEL_FATIGUE_THRESHOLD = 0.72

# Stabilization
CONSEC_WINDOWS_FOR_MILD = 1
CONSEC_WINDOWS_FOR_FATIGUE = 1
CONSEC_WINDOWS_FOR_ALERT = 1

# =========================================================
# LANDMARK INDICES
# =========================================================
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 81, 13, 311, 291, 402, 14, 178]

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
FOREHEAD_IDX = 10
CHIN_IDX = 152

# =========================================================
# HELPERS
# =========================================================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32))

def calculate_ear(eye_points):
    a = euclidean(eye_points[1], eye_points[5])
    b = euclidean(eye_points[2], eye_points[4])
    c = euclidean(eye_points[0], eye_points[3])
    if c == 0:
        return 0.0
    return float((a + b) / (2.0 * c))

def calculate_mar(mouth_points):
    a = euclidean(mouth_points[1], mouth_points[7])
    b = euclidean(mouth_points[2], mouth_points[6])
    c = euclidean(mouth_points[3], mouth_points[5])
    d = euclidean(mouth_points[0], mouth_points[4])
    if d == 0:
        return 0.0
    return float((a + b + c) / (2.0 * d))

def calculate_roll_tilt(left_eye_outer, right_eye_outer):
    dx = right_eye_outer[0] - left_eye_outer[0]
    dy = right_eye_outer[1] - left_eye_outer[1]
    return float(math.degrees(math.atan2(dy, dx)))

def calculate_pitch(chin_pt, forehead_pt):
    dx = chin_pt[0] - forehead_pt[0]
    dy = chin_pt[1] - forehead_pt[1]
    if abs(dy) < 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(dx, dy)))

def polygon_mask(shape, points):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def extract_patch_with_mask(frame, points, padding=4):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)
    x_max = min(max(xs) + padding, frame.shape[1] - 1)
    y_max = min(max(ys) + padding, frame.shape[0] - 1)

    if x_max <= x_min or y_max <= y_min:
        return None, None

    patch = frame[y_min:y_max, x_min:x_max].copy()
    shifted = [(x - x_min, y - y_min) for (x, y) in points]
    mask = polygon_mask(patch.shape, shifted)
    return patch, mask

def calculate_eye_redness(eye_patch, mask):
    if eye_patch is None or mask is None or eye_patch.size == 0:
        return 0.0

    b, g, r = cv2.split(eye_patch)
    valid = mask > 0
    if np.sum(valid) == 0:
        return 0.0

    r_i = r.astype(np.int32)
    g_i = g.astype(np.int32)
    b_i = b.astype(np.int32)

    red_pixels = ((r_i > g_i + 20) & (r_i > b_i + 20) & valid)
    return float(np.sum(red_pixels) / np.sum(valid))

def draw_text(img, text, x, y, color=(255, 255, 255), scale=0.75, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_points(frame, points, color):
    for p in points:
        cv2.circle(frame, p, 2, color, -1)

def get_state_color(state):
    if state == "Alert":
        return (0, 255, 0)
    if state == "Mild Fatigue":
        return (0, 165, 255)
    if state == "Fatigued":
        return (0, 0, 255)
    return (255, 255, 255)

def safe_stats(values):
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float32)
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

def normalized_score(value, low, high, inverse=False):
    if high == low:
        return 0.0

    if inverse:
        score = (high - value) / (high - low)
    else:
        score = (value - low) / (high - low)

    score = max(0.0, min(1.0, score))
    return float(score * 100.0)

def compute_rule_fatigue_score_exact(avg_ear, avg_mar, avg_tilt, avg_redness, perclos, blink_rate, yawn_rate, nod_rate):
    ear_score = normalized_score(avg_ear, low=0.18, high=0.32, inverse=True)
    mar_score = normalized_score(avg_mar, low=0.35, high=0.85, inverse=False)
    tilt_score = normalized_score(abs(avg_tilt), low=5.0, high=25.0, inverse=False)
    redness_score = normalized_score(avg_redness, low=0.02, high=0.18, inverse=False)
    perclos_score = normalized_score(perclos, low=10.0, high=50.0, inverse=False)
    blink_score = normalized_score(blink_rate, low=8.0, high=35.0, inverse=False)
    yawn_score = normalized_score(yawn_rate, low=0.0, high=6.0, inverse=False)
    nod_score = normalized_score(nod_rate, low=0.0, high=5.0, inverse=False)

    final_score = (
        0.20 * ear_score +
        0.10 * mar_score +
        0.08 * tilt_score +
        0.08 * redness_score +
        0.24 * perclos_score +
        0.10 * blink_score +
        0.12 * yawn_score +
        0.08 * nod_score
    )

    if final_score < ALERT_MAX:
        state = "Alert"
    elif final_score < MILD_FATIGUE_MAX:
        state = "Mild Fatigue"
    else:
        state = "Fatigued"

    return round(final_score, 2), state

def compute_display_probability(rule_score, raw_model_prob, state):
    rule_prob = max(0.0, min(1.0, rule_score / 100.0))
    blended = 0.80 * rule_prob + 0.20 * raw_model_prob

    if state == "Alert":
        blended *= 0.45
        blended = min(blended, 0.35)
    elif state == "Mild Fatigue":
        blended *= 0.78
        blended = min(blended, 0.72)
    else:
        blended *= 1.00
        blended = min(blended, 0.98)

    return round(max(0.10, min(1.0, blended)), 3)

# =========================================================
# MEDIAPIPE
# =========================================================
def create_face_landmarker(model_path: str, max_faces: int = 1):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=max_faces,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)

def detect_face_landmarks(frame, landmarker, timestamp_ms):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, int(timestamp_ms))

def extract_pixel_landmarks(face_landmarks, frame_w, frame_h):
    points = []
    for lm in face_landmarks:
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append((x, y))
    return points

def get_points_by_index(all_points, indices):
    return [all_points[i] for i in indices]

# =========================================================
# MODEL
# =========================================================
class FatigueMLPredictor:
    def __init__(self, model_path, feature_columns_path, threshold_path):
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(feature_columns_path)
        self.saved_threshold = float(joblib.load(threshold_path))

    def predict(self, feature_dict):
        x = pd.DataFrame([feature_dict])[self.feature_columns]
        prob_drowsy = float(self.model.predict_proba(x)[0][1])
        return prob_drowsy

# =========================================================
# TRACKER
# =========================================================
class Hybrid3StateTracker:
    def __init__(self):
        self.summary_start_time = time.time()
        self.system_start_time = time.time()

        self.ear_buffer = []
        self.mar_buffer = []
        self.tilt_buffer = []
        self.redness_buffer = []

        self.total_frames_window = 0
        self.valid_face_frames_window = 0
        self.closed_eye_frames_window = 0

        self.blink_count_window = 0
        self.blink_frame_counter = 0

        self.yawn_count_window = 0
        self.yawn_frame_counter = 0

        self.nod_count_window = 0
        self.pitch_history = []
        self.nod_frame_counter = 0

        self.long_eye_closure_events = 0

        self.latest_avg_ear = 0.0
        self.latest_avg_mar = 0.0
        self.latest_avg_tilt = 0.0
        self.latest_avg_redness = 0.0
        self.latest_perclos = 0.0
        self.latest_blink_rate = 0.0
        self.latest_yawn_rate = 0.0
        self.latest_nod_rate = 0.0
        self.latest_fatigue_score = 0.0

        self.latest_state = "Alert"
        self.latest_raw_model_probability = 0.0
        self.latest_display_probability = 0.10

        self.alert_streak = 0
        self.mild_streak = 0
        self.fatigue_streak = 0

        self.current_eye_closed_start = None
        self.continuous_eye_closure_sec = 0.0
        self.force_immediate_fatigue = False

    def total_observation_time(self):
        return time.time() - self.system_start_time

    def update_frame_metrics(self, ear, mar, tilt, redness, pitch):
        now = time.time()

        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)
        self.tilt_buffer.append(tilt)
        self.redness_buffer.append(redness)

        self.total_frames_window += 1
        self.valid_face_frames_window += 1

        if ear < EAR_CLOSED_THRESHOLD:
            self.closed_eye_frames_window += 1

            if self.current_eye_closed_start is None:
                self.current_eye_closed_start = now

            self.continuous_eye_closure_sec = now - self.current_eye_closed_start

            if self.continuous_eye_closure_sec >= CONTINUOUS_EYE_CLOSURE_FATIGUE_SEC:
                self.force_immediate_fatigue = True

        else:
            self.current_eye_closed_start = None
            self.continuous_eye_closure_sec = 0.0

        if ear < EAR_CLOSED_THRESHOLD:
            self.blink_frame_counter += 1
        else:
            if self.blink_frame_counter >= BLINK_CONSEC_FRAMES:
                self.blink_count_window += 1
            if self.blink_frame_counter >= 12:
                self.long_eye_closure_events += 1
            self.blink_frame_counter = 0

        if mar > MAR_YAWN_THRESHOLD:
            self.yawn_frame_counter += 1
        else:
            if self.yawn_frame_counter >= YAWN_CONSEC_FRAMES:
                self.yawn_count_window += 1
            self.yawn_frame_counter = 0

        self.pitch_history.append(pitch)
        if len(self.pitch_history) > 10:
            self.pitch_history.pop(0)

        if len(self.pitch_history) >= 5:
            pitch_range = max(self.pitch_history) - min(self.pitch_history)
            if pitch_range > HEAD_NOD_DROP_THRESHOLD:
                self.nod_frame_counter += 1
            else:
                if self.nod_frame_counter >= HEAD_NOD_CONSEC_FRAMES:
                    self.nod_count_window += 1
                self.nod_frame_counter = 0

    def update_no_face(self):
        self.total_frames_window += 1

    def is_time_to_summarize(self):
        return (time.time() - self.summary_start_time) >= SUMMARY_INTERVAL

    def build_feature_dict(self, fps_value):
        elapsed = max(time.time() - self.summary_start_time, 1e-6)

        self.latest_avg_ear, _, _, _ = safe_stats(self.ear_buffer)
        self.latest_avg_mar, _, _, _ = safe_stats(self.mar_buffer)
        self.latest_avg_tilt, _, _, _ = safe_stats(self.tilt_buffer)
        self.latest_avg_redness, _, _, _ = safe_stats(self.redness_buffer)

        ear_mean, ear_std, ear_min, ear_max = safe_stats(self.ear_buffer)
        mar_mean, mar_std, mar_min, mar_max = safe_stats(self.mar_buffer)
        tilt_mean, tilt_std, tilt_min, tilt_max = safe_stats(self.tilt_buffer)
        red_mean, red_std, red_min, red_max = safe_stats(self.redness_buffer)

        self.latest_perclos = (self.closed_eye_frames_window / max(self.total_frames_window, 1)) * 100.0
        self.latest_blink_rate = (self.blink_count_window / elapsed) * 60.0
        self.latest_yawn_rate = (self.yawn_count_window / elapsed) * 60.0
        self.latest_nod_rate = (self.nod_count_window / elapsed) * 60.0
        valid_face_ratio = self.valid_face_frames_window / max(self.total_frames_window, 1)

        return {
            "ear_mean": ear_mean,
            "ear_std": ear_std,
            "ear_min": ear_min,
            "ear_max": ear_max,
            "perclos": self.latest_perclos,
            "blink_count": float(self.blink_count_window),
            "blink_rate_per_min": self.latest_blink_rate,
            "mar_mean": mar_mean,
            "mar_std": mar_std,
            "mar_min": mar_min,
            "mar_max": mar_max,
            "yawn_count": float(self.yawn_count_window),
            "yawn_rate_per_min": self.latest_yawn_rate,
            "head_tilt_mean": tilt_mean,
            "head_tilt_std": tilt_std,
            "head_tilt_min": tilt_min,
            "head_tilt_max": tilt_max,
            "eye_redness_mean": red_mean,
            "eye_redness_std": red_std,
            "eye_redness_min": red_min,
            "eye_redness_max": red_max,
            "duration_sec": float(elapsed),
            "fps": float(fps_value),
            "valid_face_frames": float(self.valid_face_frames_window),
            "processed_frames": float(self.total_frames_window),
            "valid_face_ratio": float(valid_face_ratio),
        }

    def compute_rule_score(self):
        self.latest_fatigue_score, state = compute_rule_fatigue_score_exact(
            avg_ear=self.latest_avg_ear,
            avg_mar=self.latest_avg_mar,
            avg_tilt=self.latest_avg_tilt,
            avg_redness=self.latest_avg_redness,
            perclos=self.latest_perclos,
            blink_rate=self.latest_blink_rate,
            yawn_rate=self.latest_yawn_rate,
            nod_rate=self.latest_nod_rate,
        )
        return self.latest_fatigue_score, state

    def combine_rule_and_model(self, rule_score, rule_state, model_prob):
        if self.force_immediate_fatigue:
            return "Fatigued"

        if self.total_observation_time() < MIN_OBSERVATION_BEFORE_DECISION:
            return "Alert"

        proposed_state = rule_state

        if proposed_state == "Alert":
            if model_prob >= MODEL_MILD_THRESHOLD and rule_score >= 25:
                proposed_state = "Mild Fatigue"
        elif proposed_state == "Mild Fatigue":
            if model_prob >= MODEL_FATIGUE_THRESHOLD and rule_score >= 55:
                proposed_state = "Fatigued"

        if proposed_state == "Alert":
            self.alert_streak += 1
            self.mild_streak = 0
            self.fatigue_streak = 0
        elif proposed_state == "Mild Fatigue":
            self.mild_streak += 1
            self.alert_streak = 0
            self.fatigue_streak = 0
        else:
            self.fatigue_streak += 1
            self.alert_streak = 0
            self.mild_streak = 0

        if self.fatigue_streak >= CONSEC_WINDOWS_FOR_FATIGUE:
            return "Fatigued"
        if self.mild_streak >= CONSEC_WINDOWS_FOR_MILD:
            return "Mild Fatigue"
        return "Alert"

    def reset_window(self):
        self.summary_start_time = time.time()

        self.ear_buffer.clear()
        self.mar_buffer.clear()
        self.tilt_buffer.clear()
        self.redness_buffer.clear()

        self.total_frames_window = 0
        self.valid_face_frames_window = 0
        self.closed_eye_frames_window = 0

        self.blink_count_window = 0
        self.blink_frame_counter = 0

        self.yawn_count_window = 0
        self.yawn_frame_counter = 0

        self.nod_count_window = 0
        self.pitch_history.clear()
        self.nod_frame_counter = 0

        self.long_eye_closure_events = 0

# =========================================================
# UI
# =========================================================
def build_dashboard_panel(
    width,
    height,
    live_ear,
    live_mar,
    live_tilt,
    live_redness,
    avg_ear,
    avg_mar,
    avg_tilt,
    avg_redness,
    perclos,
    blink_rate,
    yawn_rate,
    nod_rate,
    fatigue_score,
    display_prob,
    raw_model_prob,
    state,
    continuous_eye_closure_sec,
):
    panel = np.full((height, width, 3), 35, dtype=np.uint8)

    draw_text(panel, "FATIGUE DASHBOARD", 20, 38, (0, 255, 255), 0.85, 2)
    cv2.line(panel, (20, 52), (width - 20, 52), (80, 80, 80), 2)

    y = 88
    gap = 30

    draw_text(panel, "Live Metrics", 20, y, (255, 220, 120), 0.75, 2)
    y += gap
    draw_text(panel, f"EAR        : {live_ear:.3f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"MAR        : {live_mar:.3f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Head Tilt  : {live_tilt:.2f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Redness    : {live_redness:.3f}", 25, y, scale=0.68)

    y += 20
    draw_text(panel, "8-Second Summary", 20, y, (255, 220, 120), 0.75, 2)
    y += gap
    draw_text(panel, f"Avg EAR    : {avg_ear:.3f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Avg MAR    : {avg_mar:.3f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Avg Tilt   : {avg_tilt:.2f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Avg Redness: {avg_redness:.3f}", 25, y, scale=0.68)

    y += 20
    draw_text(panel, "Fatigue Indicators", 20, y, (255, 220, 120), 0.75, 2)
    y += gap
    draw_text(panel, f"PERCLOS    : {perclos:.2f}%", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Blink/min  : {blink_rate:.2f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Yawn/min   : {yawn_rate:.2f}", 25, y, scale=0.68)
    y += gap
    draw_text(panel, f"Nod/min    : {nod_rate:.2f}", 25, y, scale=0.68)

    y += 20
    draw_text(panel, f"Eyes Closed: {continuous_eye_closure_sec:.1f}s", 25, y, (255, 255, 255), 0.66, 1)

    y += 24
    draw_text(panel, "Overall Result", 20, y, (255, 220, 120), 0.78, 2)

    y += 40
    draw_text(panel, f"Fatigue Score : {fatigue_score:.2f}", 25, y, (0, 255, 255), 0.82, 2)

    y += 36
    draw_text(panel, f"Display Prob  : {display_prob:.3f}", 25, y, (255, 255, 255), 0.72, 2)

    y += 30
    draw_text(panel, f"Raw Model Prob: {raw_model_prob:.3f}", 25, y, (170, 170, 170), 0.58, 1)

    y += 52
    draw_text(panel, f"State: {state}", 18, y, get_state_color(state), 1.05, 3)

    draw_text(panel, "ESC - Exit", 20, height - 10, (170, 170, 170), 0.48, 1)
    return panel

# =========================================================
# MAIN
# =========================================================
def main():
    landmarker = create_face_landmarker(FACE_LANDMARKER_MODEL, MAX_FACES)
    predictor = FatigueMLPredictor(
        ML_MODEL_PATH,
        FEATURE_COLUMNS_PATH,
        THRESHOLD_PATH
    )
    tracker = Hybrid3StateTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    current_ear = 0.0
    current_mar = 0.0
    current_tilt = 0.0
    current_redness = 0.0

    prev_time = time.time()
    fps_history = deque(maxlen=30)
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        now = time.time()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        fps_history.append(1.0 / dt)
        fps_value = float(np.mean(fps_history)) if fps_history else 0.0

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        timestamp_ms += int(dt * 1000) if dt > 0 else 33
        result = detect_face_landmarks(frame, landmarker, timestamp_ms)

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            all_points = extract_pixel_landmarks(face_landmarks, frame_w, frame_h)

            left_eye = get_points_by_index(all_points, LEFT_EYE_IDX)
            right_eye = get_points_by_index(all_points, RIGHT_EYE_IDX)
            mouth = get_points_by_index(all_points, MOUTH_IDX)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0

            current_mar = calculate_mar(mouth)
            current_tilt = calculate_roll_tilt(all_points[LEFT_EYE_OUTER], all_points[RIGHT_EYE_OUTER])
            current_pitch = calculate_pitch(all_points[CHIN_IDX], all_points[FOREHEAD_IDX])

            left_patch, left_mask = extract_patch_with_mask(frame, left_eye)
            right_patch, right_mask = extract_patch_with_mask(frame, right_eye)
            left_redness = calculate_eye_redness(left_patch, left_mask)
            right_redness = calculate_eye_redness(right_patch, right_mask)
            current_redness = (left_redness + right_redness) / 2.0

            tracker.update_frame_metrics(
                ear=current_ear,
                mar=current_mar,
                tilt=current_tilt,
                redness=current_redness,
                pitch=current_pitch,
            )

            if tracker.force_immediate_fatigue:
                tracker.latest_state = "Fatigued"
                tracker.latest_display_probability = 0.95
                tracker.latest_raw_model_probability = max(tracker.latest_raw_model_probability, 0.80)

            if DRAW_LANDMARKS:
                draw_points(frame, left_eye, (0, 255, 0))
                draw_points(frame, right_eye, (0, 255, 0))
                draw_points(frame, mouth, (255, 0, 0))
                cv2.line(frame, all_points[LEFT_EYE_OUTER], all_points[RIGHT_EYE_OUTER], (0, 255, 255), 2)

            draw_text(frame, f"Live EAR: {current_ear:.3f}", 20, 35, (0, 255, 0), 0.7, 2)
            draw_text(frame, f"Live MAR: {current_mar:.3f}", 20, 70, (0, 255, 0), 0.7, 2)
            draw_text(frame, f"Live Tilt: {current_tilt:.2f}", 20, 105, (0, 255, 0), 0.7, 2)

        else:
            tracker.update_no_face()
            draw_text(frame, "No face detected", 20, 40, (0, 0, 255), 0.9, 2)

        if tracker.is_time_to_summarize():
            feature_dict = tracker.build_feature_dict(fps_value=fps_value)
            raw_model_prob = predictor.predict(feature_dict)
            rule_score, rule_state = tracker.compute_rule_score()

            tracker.latest_state = tracker.combine_rule_and_model(rule_score, rule_state, raw_model_prob)
            tracker.latest_raw_model_probability = raw_model_prob
            tracker.latest_display_probability = compute_display_probability(
                rule_score=rule_score,
                raw_model_prob=raw_model_prob,
                state=tracker.latest_state
            )

            print("\n================ 8-SECOND HYBRID SUMMARY ================")
            print(f"Avg EAR            : {tracker.latest_avg_ear:.4f}")
            print(f"Avg MAR            : {tracker.latest_avg_mar:.4f}")
            print(f"Avg Head Tilt      : {tracker.latest_avg_tilt:.2f}")
            print(f"Avg Eye Redness    : {tracker.latest_avg_redness:.4f}")
            print(f"PERCLOS            : {tracker.latest_perclos:.2f}%")
            print(f"Blink Rate/min     : {tracker.latest_blink_rate:.2f}")
            print(f"Yawn Rate/min      : {tracker.latest_yawn_rate:.2f}")
            print(f"Nod Rate/min       : {tracker.latest_nod_rate:.2f}")
            print(f"Fatigue Score      : {tracker.latest_fatigue_score:.2f}")
            print(f"Rule State         : {rule_state}")
            print(f"Raw Model Prob     : {tracker.latest_raw_model_probability:.4f}")
            print(f"Display Probability: {tracker.latest_display_probability:.4f}")
            print(f"Continuous Eye Clos: {tracker.continuous_eye_closure_sec:.2f}s")
            print(f"Final State        : {tracker.latest_state}")
            print("=========================================================")

            tracker.reset_window()

        camera_view = cv2.resize(frame, (CAM_W, CAM_H))
        dashboard = build_dashboard_panel(
            width=PANEL_W,
            height=CAM_H,
            live_ear=current_ear,
            live_mar=current_mar,
            live_tilt=current_tilt,
            live_redness=current_redness,
            avg_ear=tracker.latest_avg_ear,
            avg_mar=tracker.latest_avg_mar,
            avg_tilt=tracker.latest_avg_tilt,
            avg_redness=tracker.latest_avg_redness,
            perclos=tracker.latest_perclos,
            blink_rate=tracker.latest_blink_rate,
            yawn_rate=tracker.latest_yawn_rate,
            nod_rate=tracker.latest_nod_rate,
            fatigue_score=tracker.latest_fatigue_score,
            display_prob=tracker.latest_display_probability,
            raw_model_prob=tracker.latest_raw_model_probability,
            state=tracker.latest_state,
            continuous_eye_closure_sec=tracker.continuous_eye_closure_sec,
        )

        combined = np.hstack((camera_view, dashboard))
        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()