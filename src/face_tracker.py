import time

import cv2
import mediapipe as mp

class FaceTracker:

    def __init__(self, model_path="src/face_landmarker.task"):
        self.mp_face_landmarker = mp.tasks.vision.FaceLandmarker

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(
                model_asset_path = model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )

        self.landmarker = self.mp_face_landmarker.create_from_options(options)
        self.start_time = time.monotonic()

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    def close(self):
        self.landmarker.close()
