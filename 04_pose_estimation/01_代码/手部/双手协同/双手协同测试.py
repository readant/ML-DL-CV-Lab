from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase, HAND_CONNECTIONS


class DualHandTest(MediaPipeTestBase):
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
            hand_label = "L" if result.handedness[idx][0].category_name == "Left" else "R"
            self.draw_connections(frame, landmarks, HAND_CONNECTIONS)
            self.draw_points(frame, landmarks)

    def get_status_text(self, result):
        if result.hand_landmarks:
            count = len(result.hand_landmarks)
            label = "Dual Hand" if count == 2 else "Single Hand"
            return True, f"  {label} Detected: {count} | Landmarks: 21 x {count}"
        return False, ""

    @property
    def title(self):
        return "Dual Hand Tracking"


if __name__ == "__main__":
    test = DualHandTest()
    test.run_test()
