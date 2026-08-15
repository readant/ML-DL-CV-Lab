import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase, HAND_CONNECTIONS


class HandDetectionTest(MediaPipeTestBase):
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
            h, w, _ = frame.shape
            hand_label = "L" if result.handedness[idx][0].category_name == "Left" else "R"
            self.draw_connections(frame, landmarks, HAND_CONNECTIONS)
            self.draw_points(frame, landmarks)
            index_tip = landmarks[8]
            pos = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.putText(frame, hand_label, (pos[0] + 15, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    def get_status_text(self, result):
        if result.hand_landmarks:
            return True, f"  Hands Detected: {len(result.hand_landmarks)} | Landmarks: 21 x 2"
        return False, ""

    @property
    def title(self):
        return "Hand Detection"


if __name__ == "__main__":
    test = HandDetectionTest()
    test.run_test()
