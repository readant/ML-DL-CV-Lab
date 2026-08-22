import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase


class FacialFeaturesTest(MediaPipeTestBase):
    def __init__(self):
        super().__init__("face_landmarker.task")

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def analyze_facial_features(self, face_landmarks):
        landmarks = face_landmarks[0]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]

        eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
        mouth_width = np.sqrt((mouth_left.x - mouth_right.x)**2 + (mouth_left.y - mouth_right.y)**2)
        mouth_height = np.sqrt((mouth_top.x - mouth_bottom.x)**2 + (mouth_top.y - mouth_bottom.y)**2)

        return {
            "eye_distance": eye_distance,
            "mouth_width": mouth_width,
            "mouth_height": mouth_height,
            "mouth_ratio": mouth_height / mouth_width if mouth_width > 0 else 0,
            "eye_mouth_distance": np.sqrt((nose.x - mouth_top.x)**2 + (nose.y - mouth_top.y)**2)
        }

    def draw_landmarks(self, frame, result):
        for landmarks in result.face_landmarks:
            h, w, _ = frame.shape
            for i, point in enumerate(landmarks):
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 2, (0, 255, 0), -1)

            features = self.analyze_facial_features(result.face_landmarks)
            y_offset = 100
            for key, value in features.items():
                cv2.putText(frame, f"{key}: {value:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 25

    def get_status_text(self, result):
        if result.face_landmarks:
            return True, "  Face Detected | Features: 5"
        return False, ""

    @property
    def title(self):
        return "Facial Features Analysis"


if __name__ == "__main__":
    test = FacialFeaturesTest()
    test.run_test()
