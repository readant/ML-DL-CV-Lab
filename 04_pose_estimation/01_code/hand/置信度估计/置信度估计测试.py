import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase, HAND_CONNECTIONS


class ConfidenceEstimationTest(MediaPipeTestBase):
    def __init__(self):
        super().__init__("hand_landmarker.task")

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def draw_landmarks(self, frame, result):
        for idx, landmarks in enumerate(result.hand_landmarks):
            self.draw_connections(frame, landmarks, HAND_CONNECTIONS)
            self.draw_points(frame, landmarks)
            h, w, _ = frame.shape
            wrist = landmarks[0]
            pos = (int(wrist.x * w), int(wrist.y * h))
            cv2.putText(frame, f"Hand {idx + 1}", (pos[0] + 10, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    def get_status_text(self, result):
        if result.hand_landmarks:
            return True, f"  Hands: {len(result.hand_landmarks)} | Confidence Estimation Active"
        return False, ""

    @property
    def title(self):
        return "Confidence Estimation"


if __name__ == "__main__":
    test = ConfidenceEstimationTest()
    test.run_test()
