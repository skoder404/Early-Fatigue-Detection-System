import cv2
import math
import numpy as np


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


def calculate_pitch(chin_point, forehead_point):
    dy = chin_point[1] - forehead_point[1]
    dx = chin_point[0] - forehead_point[0]
    return float(math.degrees(math.atan2(dy, max(abs(dx), 1))))


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

    red_pixels = (
        (r_i > g_i + 20) &
        (r_i > b_i + 20) &
        valid
    )

    return float(np.sum(red_pixels) / np.sum(valid))


def normalized_score(value, low, high, inverse=False):
    if high == low:
        return 0.0

    if inverse:
        score = (high - value) / (high - low)
    else:
        score = (value - low) / (high - low)

    score = max(0.0, min(1.0, score))
    return float(score * 100.0)


def compute_fatigue_score(avg_ear, avg_mar, avg_tilt, avg_redness, perclos, blink_rate, yawn_rate, nod_rate):
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

    if final_score < 35:
        state = "Alert"
    elif final_score < 65:
        state = "Mild Fatigue"
    else:
        state = "Fatigued"

    return round(final_score, 2), state