import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase


class ExpressionRecognitionTest(MediaPipeTestBase):
    def __init__(self):
        super().__init__("face_landmarker.task")

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def analyze_expression(self, landmarks):
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        mouth_width = abs(mouth_right.x - mouth_left.x)
        mouth_height = abs(mouth_bottom.y - mouth_top.y)
        eye_distance = abs(right_eye.x - left_eye.x)
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

        if mouth_ratio > 0.35:
            return "Surprise"
        elif mouth_ratio > 0.25:
            return "Smile"
        elif mouth_ratio < 0.1:
            return "Serious"
        return "Neutral"

    def draw_landmarks(self, frame, result):
        for landmarks in result.face_landmarks:
            h, w, _ = frame.shape
            for i, point in enumerate(landmarks):
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 2, (0, 255, 0), -1)
            expression = self.analyze_expression(landmarks)
            nose = landmarks[1]
            pos = (int(nose.x * w), int(nose.y * h) - 30)
            cv2.putText(frame, expression, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    def get_status_text(self, result):
        if result.face_landmarks:
            return True, "  Face Detected | Expression Recognition Active"
        return False, ""

    @property
    def title(self):
        return "Expression Recognition"


if __name__ == "__main__":
    test = ExpressionRecognitionTest()
    test.run_test()
