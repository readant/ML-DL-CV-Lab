import cv2
import mediapipe as mp
import numpy as np
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (11, 12),
    (12, 24),
    (24, 23),
    (23, 11),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]

POSE_COLOR_MAP = {
    0: (0, 255, 255),
    1: (0, 255, 255),
    2: (0, 255, 255),
    3: (0, 255, 255),
    4: (0, 255, 255),
    5: (0, 255, 255),
    6: (0, 255, 255),
    7: (0, 255, 255),
    8: (0, 255, 255),
    9: (0, 255, 255),
    10: (0, 255, 255),
    11: (0, 255, 0),
    12: (0, 255, 0),
    13: (0, 200, 0),
    14: (0, 200, 0),
    15: (0, 150, 0),
    16: (0, 150, 0),
    23: (0, 255, 0),
    24: (0, 255, 0),
    25: (255, 165, 0),
    26: (255, 165, 0),
    27: (255, 165, 0),
    28: (255, 165, 0),
    29: (255, 165, 0),
    30: (255, 165, 0),
    31: (255, 165, 0),
    32: (255, 165, 0),
}


class MediaPipeTestBase:
    """MediaPipe 测试脚本基类，封装通用的初始化、主循环、UI 绘制和绘图工具。"""

    def __init__(self, model_file):
        # 模型统一放在仓库根级 models/，通过 __file__ 上溯定位
        self.model_path = Path(__file__).resolve().parents[2] / "models" / model_file
        self.detector = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self._initialize_detector()

    def _initialize_detector(self):
        raise NotImplementedError

    def detect(self, mp_image):
        raise NotImplementedError

    def draw_landmarks(self, frame, result):
        raise NotImplementedError

    def get_status_text(self, result):
        raise NotImplementedError

    @property
    def title(self):
        raise NotImplementedError

    def update_fps(self):
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.fps = 10 / elapsed if elapsed > 0 else 0
            self.start_time = time.time()

    def draw_neon_line(self, frame, start, end, color, thickness=2):
        overlay = frame.copy()
        cv2.line(overlay, start, end, color, thickness + 2)
        cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)
        cv2.line(frame, start, end, color, thickness)

    def draw_neon_point(self, frame, pos, color, radius=5):
        overlay = frame.copy()
        cv2.circle(overlay, pos, radius + 2, color, -1)
        cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)
        cv2.circle(frame, pos, radius, (255, 255, 255), -1)
        cv2.circle(frame, pos, radius - 2, color, -1)

    def draw_connections(self, frame, landmarks, connections, color_map=None):
        h, w, _ = frame.shape
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                sp = landmarks[start_idx]
                ep = landmarks[end_idx]
                start = (int(sp.x * w), int(sp.y * h))
                end = (int(ep.x * w), int(ep.y * h))
                color = (
                    color_map.get(start_idx, (0, 255, 255))
                    if color_map
                    else (0, 255, 255)
                )
                self.draw_neon_line(frame, start, end, color, 2)

    def draw_points(self, frame, landmarks, color_map=None):
        h, w, _ = frame.shape
        for i, point in enumerate(landmarks):
            pos = (int(point.x * w), int(point.y * h))
            color = color_map.get(i, (0, 200, 255)) if color_map else (0, 200, 255)
            self.draw_neon_point(frame, pos, color, 4)

    def draw_ui(self, frame, detected, status_text=""):
        h, w, _ = frame.shape

        header_height = 70
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        header.fill(25)
        cv2.addWeighted(
            frame[:header_height], 0.85, header, 0.15, 0, frame[:header_height]
        )
        cv2.putText(
            frame,
            self.title,
            (20, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            1.3,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (w - 120, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        status_bar = np.zeros((50, w, 3), dtype=np.uint8)
        status_bar.fill(20)
        frame[h - 50 : h, :] = status_bar

        if detected:
            cv2.putText(
                frame,
                status_text,
                (20, h - 18),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 255, 0), -1)
        else:
            cv2.putText(
                frame,
                f"  No {self.title.replace(' Detection', '').replace(' Recognition', '').replace(' Analysis', '')} Detected",
                (20, h - 18),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 0, 255), -1)

        cv2.putText(
            frame,
            "ESC to Exit",
            (w - 150, h - 18),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    def run_test(self):
        print(f"{self.title} Test")
        print("Press ESC to exit")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Cannot read camera frame")
                    break

                frame = cv2.flip(frame, 1)
                self.update_fps()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detect(mp_image)

                detected, status_text = self.get_status_text(result)
                if detected:
                    self.draw_landmarks(frame, result)

                self.draw_ui(frame, detected, status_text)
                cv2.imshow(self.title, frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")
