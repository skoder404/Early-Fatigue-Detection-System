import csv
import os
import time
import numpy as np

from config import (
    SUMMARY_INTERVAL,
    EAR_CLOSED_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    BLINK_CONSEC_FRAMES,
    YAWN_CONSEC_FRAMES,
    HEAD_NOD_DROP_THRESHOLD,
    HEAD_NOD_CONSEC_FRAMES,
    CONTINUOUS_EYE_CLOSURE_FATIGUE_SEC,
    SAVE_TO_CSV,
    CSV_FILE,
)
from features import compute_fatigue_score


class FatigueTracker:
    def __init__(self):
        self.summary_start_time = time.time()

        self.ear_buffer = []
        self.mar_buffer = []
        self.tilt_buffer = []
        self.redness_buffer = []

        self.total_frames_window = 0
        self.closed_eye_frames_window = 0

        self.blink_count_window = 0
        self.blink_frame_counter = 0

        self.yawn_count_window = 0
        self.yawn_frame_counter = 0

        self.nod_count_window = 0
        self.pitch_history = []
        self.nod_frame_counter = 0

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

        self.current_eye_closed_start = None
        self.continuous_eye_closure_sec = 0.0
        self.force_immediate_fatigue = False

        if SAVE_TO_CSV:
            self._init_csv()

    def _init_csv(self):
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "avg_ear",
                    "avg_mar",
                    "avg_tilt",
                    "avg_redness",
                    "perclos",
                    "blink_count",
                    "blink_rate_per_min",
                    "yawn_count",
                    "yawn_rate_per_min",
                    "nod_count",
                    "nod_rate_per_min",
                    "continuous_eye_closure_sec",
                    "fatigue_score",
                    "state",
                ])

    def _append_csv(self):
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(self.latest_avg_ear, 4),
                round(self.latest_avg_mar, 4),
                round(self.latest_avg_tilt, 2),
                round(self.latest_avg_redness, 4),
                round(self.latest_perclos, 2),
                self.blink_count_window,
                round(self.latest_blink_rate, 2),
                self.yawn_count_window,
                round(self.latest_yawn_rate, 2),
                self.nod_count_window,
                round(self.latest_nod_rate, 2),
                round(self.continuous_eye_closure_sec, 2),
                round(self.latest_fatigue_score, 2),
                self.latest_state,
            ])

    def update_frame_metrics(self, ear, mar, tilt, redness, pitch):
        now = time.time()

        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)
        self.tilt_buffer.append(tilt)
        self.redness_buffer.append(redness)

        self.total_frames_window += 1

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

    def update_summary_if_needed(self):
        now = time.time()
        elapsed = now - self.summary_start_time

        if elapsed < SUMMARY_INTERVAL:
            return False

        if self.total_frames_window > 0:
            self.latest_avg_ear = float(np.mean(self.ear_buffer)) if self.ear_buffer else 0.0
            self.latest_avg_mar = float(np.mean(self.mar_buffer)) if self.mar_buffer else 0.0
            self.latest_avg_tilt = float(np.mean(self.tilt_buffer)) if self.tilt_buffer else 0.0
            self.latest_avg_redness = float(np.mean(self.redness_buffer)) if self.redness_buffer else 0.0

            self.latest_perclos = (self.closed_eye_frames_window / self.total_frames_window) * 100.0
            self.latest_blink_rate = (self.blink_count_window / elapsed) * 60.0
            self.latest_yawn_rate = (self.yawn_count_window / elapsed) * 60.0
            self.latest_nod_rate = (self.nod_count_window / elapsed) * 60.0

            self.latest_fatigue_score, self.latest_state = compute_fatigue_score(
                avg_ear=self.latest_avg_ear,
                avg_mar=self.latest_avg_mar,
                avg_tilt=self.latest_avg_tilt,
                avg_redness=self.latest_avg_redness,
                perclos=self.latest_perclos,
                blink_rate=self.latest_blink_rate,
                yawn_rate=self.latest_yawn_rate,
                nod_rate=self.latest_nod_rate,
            )

            if self.force_immediate_fatigue:
                self.latest_state = "Fatigued"
                self.latest_fatigue_score = max(self.latest_fatigue_score, 90.0)

            print("\n================ 8-SECOND SUMMARY ================")
            print(f"Avg EAR                 : {self.latest_avg_ear:.4f}")
            print(f"Avg MAR                 : {self.latest_avg_mar:.4f}")
            print(f"Avg Head Tilt           : {self.latest_avg_tilt:.2f}")
            print(f"Avg Eye Redness         : {self.latest_avg_redness:.4f}")
            print(f"PERCLOS                 : {self.latest_perclos:.2f}%")
            print(f"Blink Rate/min          : {self.latest_blink_rate:.2f}")
            print(f"Yawn Rate/min           : {self.latest_yawn_rate:.2f}")
            print(f"Nod Rate/min            : {self.latest_nod_rate:.2f}")
            print(f"Continuous Eye Closure  : {self.continuous_eye_closure_sec:.2f}s")
            print(f"Fatigue Score           : {self.latest_fatigue_score:.2f}")
            print(f"State                   : {self.latest_state}")
            print("==================================================")

            if SAVE_TO_CSV:
                self._append_csv()

        self.reset_window(now)
        return True

    def reset_window(self, now=None):
        self.summary_start_time = now if now is not None else time.time()

        self.ear_buffer.clear()
        self.mar_buffer.clear()
        self.tilt_buffer.clear()
        self.redness_buffer.clear()

        self.total_frames_window = 0
        self.closed_eye_frames_window = 0

        self.blink_count_window = 0
        self.blink_frame_counter = 0

        self.yawn_count_window = 0
        self.yawn_frame_counter = 0

        self.nod_count_window = 0
        self.nod_frame_counter = 0
        self.pitch_history.clear()

    def full_reset_after_alarm(self):
        self.summary_start_time = time.time()

        self.ear_buffer.clear()
        self.mar_buffer.clear()
        self.tilt_buffer.clear()
        self.redness_buffer.clear()

        self.total_frames_window = 0
        self.closed_eye_frames_window = 0

        self.blink_count_window = 0
        self.blink_frame_counter = 0

        self.yawn_count_window = 0
        self.yawn_frame_counter = 0

        self.nod_count_window = 0
        self.nod_frame_counter = 0
        self.pitch_history.clear()

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

        self.current_eye_closed_start = None
        self.continuous_eye_closure_sec = 0.0
        self.force_immediate_fatigue = False