import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase


class FaceDetectionTest(MediaPipeTestBase):
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

    def draw_landmarks(self, frame, result):
        for landmarks in result.face_landmarks:
            h, w, _ = frame.shape
            for i, point in enumerate(landmarks):
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 2, (0, 255, 0), -1)

    def get_status_text(self, result):
        if result.face_landmarks:
            return True, "  Face Detected | Landmarks: 478"
        return False, ""

    @property
    def title(self):
        return "Face Detection"


if __name__ == "__main__":
    test = FaceDetectionTest()
    test.run_test()
