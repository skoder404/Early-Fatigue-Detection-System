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

SUMMARY_INTERVAL = 8.0
MIN_OBSERVATION_BEFORE_DECISION = 4.0

MAX_FACES = 1
DRAW_LANDMARKS = True

EAR_CLOSED_THRESHOLD = 0.23
MAR_YAWN_THRESHOLD = 0.65

BLINK_CONSEC_FRAMES = 2
YAWN_CONSEC_FRAMES = 8

HEAD_NOD_DROP_THRESHOLD = 8.0
HEAD_NOD_CONSEC_FRAMES = 3

CONTINUOUS_EYE_CLOSURE_FATIGUE_SEC = 5.0

ALERT_MAX = 35
MILD_FATIGUE_MAX = 65

MODEL_MILD_THRESHOLD = 0.60
MODEL_FATIGUE_THRESHOLD = 0.72

CONSEC_WINDOWS_FOR_MILD = 1
CONSEC_WINDOWS_FOR_FATIGUE = 1
CONSEC_WINDOWS_FOR_ALERT = 1

# =========================================================
# ALARM SETTINGS
# =========================================================
MILD_ALARM_SOUND_PATH = "mild_alarm.wav"
FATIGUE_ALARM_SOUND_PATH = "alarm.wav"
STOP_ALARM_KEY = ord('s')

# =========================================================
# WHATSAPP / EMERGENCY SETTINGS
# =========================================================
EMERGENCY_COUNTDOWN_SEC = 30

EMERGENCY_CONTACT_NAME = "Family Member"

import os
from dotenv import load_dotenv
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")

EMERGENCY_WHATSAPP_NUMBERS = [
    os.getenv("EMERGENCY_WHATSAPP_1", ""),
    os.getenv("EMERGENCY_WHATSAPP_2", ""),
    os.getenv("EMERGENCY_WHATSAPP_3", ""),
    os.getenv("EMERGENCY_WHATSAPP_4", ""),
]

# Future emergency service placeholders
EMERGENCY_POLICE = "100"
EMERGENCY_AMBULANCE = "108"

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