import cv2
import time
import numpy as np
from collections import deque
from tkinter import Tk, messagebox

from config import (
    MILD_ALARM_SOUND_PATH,
    FATIGUE_ALARM_SOUND_PATH,
    STOP_ALARM_KEY,
    EMERGENCY_COUNTDOWN_SEC,
    EMERGENCY_CONTACT_NAME,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_FROM,
    EMERGENCY_WHATSAPP_NUMBERS,
    FACE_LANDMARKER_MODEL,
    ML_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    THRESHOLD_PATH,
    CAMERA_INDEX,
    WINDOW_NAME,
    MAX_FACES,
    DRAW_LANDMARKS,
    CAM_W,
    CAM_H,
    PANEL_W,
    WINDOW_W,
    WINDOW_H,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_IDX,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    FOREHEAD_IDX,
    CHIN_IDX,
)
from alarm_manager import AlarmManager
from detector import create_face_landmarker, detect_face_landmarks, extract_pixel_landmarks, get_points_by_index
from features import (
    calculate_ear,
    calculate_mar,
    calculate_roll_tilt,
    calculate_pitch,
    extract_patch_with_mask,
    calculate_eye_redness,
)
from predictor import FatigueMLPredictor
from tracker import Hybrid3StateTracker
from ui import draw_text, draw_points, build_dashboard_panel
from whatsapp_notifier import send_whatsapp_alert
from location_helper import get_current_location_text


def show_emergency_popup(msg):
    root = Tk()
    root.withdraw()
    messagebox.showwarning("Emergency Alert", msg)
    root.destroy()


def main():
    landmarker = create_face_landmarker(FACE_LANDMARKER_MODEL, MAX_FACES)
    predictor = FatigueMLPredictor(
        ML_MODEL_PATH,
        FEATURE_COLUMNS_PATH,
        THRESHOLD_PATH
    )
    tracker = Hybrid3StateTracker()

    mild_alarm = AlarmManager(MILD_ALARM_SOUND_PATH)
    fatigue_alarm = AlarmManager(FATIGUE_ALARM_SOUND_PATH)

    mild_alarm_active = False
    fatigue_alarm_active = False

    fatigued_start_time = None
    emergency_sent = False

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

    prev_time = 0.0
    fps_history = deque(maxlen=30)
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        now = cv2.getTickCount() / cv2.getTickFrequency()
        if prev_time == 0.0:
            dt = 1 / 30
        else:
            dt = max(now - prev_time, 1e-6)
        prev_time = now

        fps_history.append(1.0 / dt)
        fps_value = float(np.mean(fps_history)) if fps_history else 0.0

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        # Pause detection only for fatigued alarm
        if fatigue_alarm_active:
            draw_text(frame, "FATIGUE DETECTED!", 20, 160, (0, 0, 255), 0.9, 3)
            draw_text(frame, "Press S to stop alarm and restart", 20, 200, (0, 255, 255), 0.7, 2)

            if fatigued_start_time is not None:
                elapsed = time.time() - fatigued_start_time
                remaining = max(0, int(EMERGENCY_COUNTDOWN_SEC - elapsed))
                draw_text(frame, f"Emergency escalation in: {remaining}s", 20, 240, (0, 0, 255), 0.7, 2)

                if elapsed >= EMERGENCY_COUNTDOWN_SEC and not emergency_sent:
                    loc = get_current_location_text()
                    message_text = ("Emergency Warning!\n"
                            "Driver remained in fatigued state and did not stop the alarm for 60 seconds.\n"
                            f"Location: {loc['city']}, {loc['region']}, {loc['country']}\n"
                            f"Coordinates: {loc['coordinates']}\n"
                            f"Map: {loc['map_link']}"
                        )

                    try:
                        send_whatsapp_alert(
                            TWILIO_ACCOUNT_SID,
                            TWILIO_AUTH_TOKEN,
                            TWILIO_WHATSAPP_FROM,
                            EMERGENCY_WHATSAPP_NUMBERS,
                            message_text
                        )
                        print("Emergency WhatsApp alert sent.")
                    except Exception as e:
                        print("Failed to send WhatsApp alert:", e)

                    popup_text = (
                            "Emergency escalation triggered.\n\n"
                            "Contacting / Sending message to family member(s)...\n\n"
                            f"Location: {loc['city']}, {loc['region']}, {loc['country']}\n"
                            f"Coordinates: {loc['coordinates']}\n"
                            f"Map: {loc['map_link']}\n\n"
                            "Placeholder for future emergency service integration: 100 / 108"
                        )
                    show_emergency_popup(popup_text)

                    # Future integration:
                    # Contact police: 100
                    # Contact ambulance: 108

                    emergency_sent = True

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

            key = cv2.waitKey(1) & 0xFF
            if key == STOP_ALARM_KEY:
                fatigue_alarm.stop_alarm()
                fatigue_alarm_active = False

                if mild_alarm_active:
                    mild_alarm.stop_alarm()
                    mild_alarm_active = False

                fatigued_start_time = None
                emergency_sent = False
                tracker.full_reset_after_alarm()

            elif key == 27:
                break

            continue

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
            tracker.update_summary(raw_model_prob)

            if tracker.latest_state == "Mild Fatigue":
                if not mild_alarm_active and not fatigue_alarm_active:
                    mild_alarm.start_alarm()
                    mild_alarm_active = True

            elif tracker.latest_state == "Fatigued":
                if mild_alarm_active:
                    mild_alarm.stop_alarm()
                    mild_alarm_active = False

                if not fatigue_alarm_active:
                    fatigue_alarm.start_alarm()
                    fatigue_alarm_active = True
                    fatigued_start_time = time.time()
                    emergency_sent = False

            elif tracker.latest_state == "Alert":
                if mild_alarm_active:
                    mild_alarm.stop_alarm()
                    mild_alarm_active = False

            print("\n================ 8-SECOND HYBRID SUMMARY ================")
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

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    mild_alarm.stop_alarm()
    fatigue_alarm.stop_alarm()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()