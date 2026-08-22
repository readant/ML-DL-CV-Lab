from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from base_test import MediaPipeTestBase, POSE_CONNECTIONS, POSE_COLOR_MAP


class FullBodyDetectionTest(MediaPipeTestBase):
    def __init__(self):
        super().__init__("pose_landmarker.task")

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def draw_landmarks(self, frame, result):
        for landmarks in result.pose_landmarks:
            self.draw_connections(frame, landmarks, POSE_CONNECTIONS, POSE_COLOR_MAP)
            self.draw_points(frame, landmarks, POSE_COLOR_MAP)

    def get_status_text(self, result):
        if result.pose_landmarks:
            return True, "  Full Body Detected | 33 Landmarks"
        return False, ""

    @property
    def title(self):
        return "Full Body Detection"


if __name__ == "__main__":
    test = FullBodyDetectionTest()
    test.run_test()
