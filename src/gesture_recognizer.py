"""
Gesture recognition engine using MediaPipe Hands.
Extracts 21 3D hand landmarks and classifies static gestures via heuristics.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

from src.config import (
    FINGER_CLOSED_THRESHOLD,
    FINGER_OPEN_THRESHOLD,
    FINGER_TIP_INDICES,
    FINGER_MCP_INDICES,
    PALM_CENTER_INDICES,
    THUMB_IP_INDEX,
    THUMB_MCP_INDEX,
)

# Import from mediapipe.python.solutions for version 0.10.x compatibility
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing


def _make_landmark(x: float, y: float, z: float = 0.0):
    """Create a MediaPipe-compatible landmark-like object."""
    class Landmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    return Landmark(x, y, z)


class GestureRecognizer:
    """Recognizes hand gestures from MediaPipe landmarks using distance-based heuristics."""

    def __init__(self, static_image_mode: bool = False, max_hands: int = 1):
        self.mp_hands = mp_hands_module
        self.mp_drawing = mp_drawing
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.prev_frame = None
        self._hand_history: list[tuple[float, float]] = []
        self._swipe_threshold_x = 0.15
        self._swipe_threshold_y = 0.15
        self._swipe_window = 5

    @staticmethod
    def _euclidean_distance(p1, p2) -> float:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return (dx**2 + dy**2 + dz**2) ** 0.5

    def _get_palm_center(self, landmarks: list):
        indices = PALM_CENTER_INDICES
        avg_x = sum(landmarks[i].x for i in indices) / len(indices)
        avg_y = sum(landmarks[i].y for i in indices) / len(indices)
        avg_z = sum(landmarks[i].z for i in indices) / len(indices)
        return _make_landmark(avg_x, avg_y, avg_z)

    def _get_wrist(self, landmarks: list):
        return landmarks[0]

    def _is_finger_extended(self, landmarks: list, finger_name: str) -> bool:
        """
        Determine if a finger is extended by comparing
        tip-to-palm distance vs MCP-to-palm distance.
        Ratio > FINGER_OPEN_THRESHOLD means extended.
        """
        tip_idx = FINGER_TIP_INDICES[finger_name]
        mcp_idx = FINGER_MCP_INDICES[finger_name]
        palm = self._get_palm_center(landmarks)

        tip_dist = self._euclidean_distance(landmarks[tip_idx], palm)
        mcp_dist = self._euclidean_distance(landmarks[mcp_idx], palm)

        if mcp_dist < 1e-6:
            return False

        ratio = tip_dist / mcp_dist
        return ratio > FINGER_OPEN_THRESHOLD

    def _is_finger_folded(self, landmarks: list, finger_name: str) -> bool:
        return not self._is_finger_extended(landmarks, finger_name)

    def _is_thumb_extended(self, landmarks: list) -> bool:
        """
        Thumb detection: compare thumb tip to IP joint distance
        vs thumb MCP to IP joint distance.
        Also considers thumb direction (left/right of hand).
        """
        tip = landmarks[FINGER_TIP_INDICES["thumb"]]
        ip = landmarks[THUMB_IP_INDEX]
        mcp = landmarks[THUMB_MCP_INDEX]

        tip_to_ip = self._euclidean_distance(tip, ip)
        ip_to_mcp = self._euclidean_distance(ip, mcp)

        if ip_to_mcp < 1e-6:
            return False

        ratio = tip_to_ip / ip_to_mcp
        return ratio > 0.5

    def count_fingers(self, landmarks: list) -> int:
        """Count how many fingers are extended (excluding thumb)."""
        count = 0
        for finger_name in ["index", "middle", "ring", "pinky"]:
            if self._is_finger_extended(landmarks, finger_name):
                count += 1
        return count

    def is_thumb_up(self, landmarks: list) -> bool:
        """Detect thumbs-up: thumb extended upward, other 4 fingers folded."""
        if not self._is_thumb_extended(landmarks):
            return False

        for finger_name in ["index", "middle", "ring", "pinky"]:
            if self._is_finger_extended(landmarks, finger_name):
                return False

        # Verify thumb is pointing up (thumb tip Y < thumb IP Y in image coords)
        wrist = self._get_wrist(landmarks)
        thumb_tip = landmarks[FINGER_TIP_INDICES["thumb"]]
        thumb_ip = landmarks[THUMB_IP_INDEX]

        return thumb_tip.y < thumb_ip.y

    def is_fist(self, landmarks: list) -> bool:
        """Detect fist: all 4 fingers folded (thumb can be any position)."""
        for finger_name in ["index", "middle", "ring", "pinky"]:
            if self._is_finger_extended(landmarks, finger_name):
                return False
        return True

    def is_open_hand(self, landmarks: list) -> bool:
        """Detect open hand: all 4 fingers extended (thumb doesn't matter)."""
        for finger_name in ["index", "middle", "ring", "pinky"]:
            if not self._is_finger_extended(landmarks, finger_name):
                return False
        return True

    def classify(self, landmarks: list) -> str:
        """
        Classify the current hand state into a gesture name.
        Returns one of: 'fist', 'open_hand', 'thumbs_up', 'one_finger', 'two_fingers', 'three_fingers', 'four_fingers', 'five_fingers', 'none'
        """
        if self.is_thumb_up(landmarks):
            return "thumbs_up"

        if self.is_fist(landmarks):
            return "fist"

        count = self.count_fingers(landmarks)
        if count == 4:
            return "open_hand"
        elif count == 1:
            return "one_finger"
        elif count == 2:
            return "two_fingers"
        elif count == 3:
            return "three_fingers"
        elif count == 4:
            return "four_fingers"

        if self.is_open_hand(landmarks):
            return "open_hand"

        return "none"

    def process_frame(self, frame) -> Tuple:
        """
        Process a single frame: detect hands, extract landmarks, classify gesture.

        Returns: (frame_with_landmarks, landmarks_list_or_None, gesture_name_or_None)
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        output_frame = frame.copy()
        landmarks = None
        gesture = None

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            gesture = self.classify(landmarks)

            if self.mp_drawing is not None:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

            # Draw gesture label
            cv2.putText(
                output_frame,
                gesture,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

        return output_frame, landmarks, gesture

    def _detect_swipe(self, frame_width: int, frame_height: int) -> str:
        if len(self._hand_history) < self._swipe_window:
            return "none"

        recent = self._hand_history[-self._swipe_window:]
        start = recent[0]
        end = recent[-1]

        dx = (end[0] - start[0]) * frame_width
        dy = (end[1] - start[1]) * frame_height

        if abs(dx) < self._swipe_threshold_x * frame_width and abs(dy) < self._swipe_threshold_y * frame_height:
            return "none"

        if abs(dx) > abs(dy):
            if dx < 0:
                return "swipe_left"
            else:
                return "swipe_right"
        else:
            if dy < 0:
                return "swipe_up"
            else:
                return "swipe_down"

    def process_frame(self, frame) -> Tuple:
        """
        Process a single frame: detect hands, extract landmarks, classify gesture.

        Returns: (frame_with_landmarks, landmarks_list_or_None, gesture_name_or_None)
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        output_frame = frame.copy()
        landmarks = None
        gesture = None

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            gesture = self.classify(landmarks)

            wrist = self._get_wrist(landmarks)
            self._hand_history.append((wrist.x, wrist.y))

            if len(self._hand_history) > 15:
                self._hand_history.pop(0)

            if self.mp_drawing is not None:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

            # Draw gesture label
            cv2.putText(
                output_frame,
                gesture,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
        else:
            self._hand_history.clear()

        return output_frame, landmarks, gesture

    def release(self):
        self.hands.close()
