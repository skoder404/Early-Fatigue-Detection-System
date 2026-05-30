import time
from config import (
    SUMMARY_INTERVAL,
    MIN_OBSERVATION_BEFORE_DECISION,
    EAR_CLOSED_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    BLINK_CONSEC_FRAMES,
    YAWN_CONSEC_FRAMES,
    HEAD_NOD_DROP_THRESHOLD,
    HEAD_NOD_CONSEC_FRAMES,
    CONTINUOUS_EYE_CLOSURE_FATIGUE_SEC,
    MODEL_MILD_THRESHOLD,
    MODEL_FATIGUE_THRESHOLD,
    CONSEC_WINDOWS_FOR_MILD,
    CONSEC_WINDOWS_FOR_FATIGUE,
    CONSEC_WINDOWS_FOR_ALERT,
)
from features import safe_stats, compute_rule_fatigue_score_exact, compute_display_probability


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
        if self.alert_streak >= CONSEC_WINDOWS_FOR_ALERT:
            return "Alert"
        return self.latest_state

    def update_summary(self, raw_model_prob):
        rule_score, rule_state = self.compute_rule_score()
        self.latest_state = self.combine_rule_and_model(rule_score, rule_state, raw_model_prob)
        self.latest_raw_model_probability = raw_model_prob
        self.latest_display_probability = compute_display_probability(
            rule_score=rule_score,
            raw_model_prob=raw_model_prob,
            state=self.latest_state
        )

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

    def full_reset_after_alarm(self):
        self.summary_start_time = time.time()
        self.system_start_time = time.time()

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