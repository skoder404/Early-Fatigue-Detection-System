import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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