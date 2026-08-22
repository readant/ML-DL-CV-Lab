import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase, HAND_CONNECTIONS


class GestureRecognitionTest(MediaPipeTestBase):
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

    def recognize_gesture(self, landmarks):
        wrist = landmarks[0]
        thumb_tip, thumb_ip = landmarks[4], landmarks[3]
        index_tip, index_pip = landmarks[8], landmarks[7]
        middle_tip, middle_pip = landmarks[12], landmarks[11]
        ring_tip, ring_pip = landmarks[16], landmarks[15]
        pinky_tip, pinky_pip = landmarks[20], landmarks[19]

        def is_finger_extended(tip, pip):
            return tip.y < pip.y

        fingers_up = sum([
            thumb_tip.y < thumb_ip.y and abs(thumb_tip.x - wrist.x) > 0.1,
            is_finger_extended(index_tip, index_pip),
            is_finger_extended(middle_tip, middle_pip),
            is_finger_extended(ring_tip, ring_pip),
            is_finger_extended(pinky_tip, pinky_pip)
        ])

        gestures = {
            0: ("Fist", (0, 0, 255)),
            1: ("One", (0, 255, 255)),
            2: ("Two", (0, 255, 0)),
            3: ("Three", (255, 100, 0)),
            4: ("Four", (255, 255, 0)),
            5: ("Five", (0, 255, 255))
        }
        return gestures.get(fingers_up, ("Unknown", (128, 128, 128)))

    def draw_landmarks(self, frame, result):
        for idx, landmarks in enumerate(result.hand_landmarks):
            h, w, _ = frame.shape
            gesture, color = self.recognize_gesture(landmarks)
            hand_label = "L" if result.handedness[idx][0].category_name == "Left" else "R"
            self.draw_connections(frame, landmarks, HAND_CONNECTIONS)
            self.draw_points(frame, landmarks)
            index_tip = landmarks[8]
            pos = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.putText(frame, f"{hand_label}: {gesture}", (pos[0] + 15, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def get_status_text(self, result):
        if result.hand_landmarks:
            gestures = [self.recognize_gesture(lm)[0] for lm in result.hand_landmarks]
            return True, f"  {' | '.join(gestures)}"
        return False, ""

    @property
    def title(self):
        return "Gesture Recognition"


if __name__ == "__main__":
    test = GestureRecognitionTest()
    test.run_test()
