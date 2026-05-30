import cv2
import numpy as np


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
    return (0, 0, 255)


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
    continuous_eye_closure_sec,
    fatigue_score,
    state,
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

    y += 52
    draw_text(panel, f"State: {state}", 18, y, get_state_color(state), 1.05, 3)

    draw_text(panel, "ESC - Exit", 20, height - 10, (170, 170, 170), 0.48, 1)
    return panel