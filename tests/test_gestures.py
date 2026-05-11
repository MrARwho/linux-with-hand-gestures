"""
Unit tests for gesture detection heuristics.
Tests distance-based finger classification logic.

MediaPipe 21-landmark structure:
0: wrist
1: thumb CMC, 2: thumb MCP, 3: thumb IP, 4: thumb tip
5: index MCP, 6: index PIP, 7: index DIP, 8: index tip
9: middle MCP, 10: middle PIP, 11: middle DIP, 12: middle tip
13: ring MCP, 14: ring PIP, 15: ring DIP, 16: ring tip
17: pinky MCP, 18: pinky PIP, 19: pinky DIP, 20: pinky tip
"""

import pytest


def _make_landmark(x: float, y: float, z: float = 0.0):
    class Landmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    return Landmark(x, y, z)


def _make_hands_landmarks(
    finger_ratios: dict = None,
    thumb_tip_y_diff: float = -0.05,
) -> list:
    """
    Create a list of 21 MediaPipe landmarks.
    finger_ratios: dict mapping finger_name -> tip_to_palm_ratio
    thumb_tip_y_diff: thumb tip Y relative to thumb IP Y (negative = pointing up)
    """
    if finger_ratios is None:
        finger_ratios = {
            "index": 1.5,
            "middle": 1.5,
            "ring": 1.5,
            "pinky": 1.5,
        }

    palm_center_x = 0.5
    palm_center_y = 0.45

    # 0: wrist
    wrist = _make_landmark(0.5, 0.5)

    # 1: thumb CMC
    thumb_cmc = _make_landmark(0.42, 0.48)
    # 2: thumb MCP
    thumb_mcp = _make_landmark(0.40, 0.45)
    # 3: thumb IP
    thumb_ip = _make_landmark(0.36, 0.40)
    # 4: thumb tip
    thumb_tip = _make_landmark(0.32, thumb_ip.y + thumb_tip_y_diff)

    # 5-8: index (MCP, PIP, DIP, tip)
    index_mcp = _make_landmark(0.48, 0.36)
    index_pip = _make_landmark(0.47, 0.30)
    index_dip = _make_landmark(0.46, 0.25)
    index_tip_y = palm_center_y - (palm_center_y - index_mcp.y) * finger_ratios["index"]
    index_tip = _make_landmark(0.45, index_tip_y)

    # 9-12: middle (MCP, PIP, DIP, tip)
    middle_mcp = _make_landmark(0.47, 0.34)
    middle_pip = _make_landmark(0.46, 0.28)
    middle_dip = _make_landmark(0.46, 0.23)
    middle_tip_y = palm_center_y - (palm_center_y - middle_mcp.y) * finger_ratios["middle"]
    middle_tip = _make_landmark(0.45, middle_tip_y)

    # 13-16: ring (MCP, PIP, DIP, tip)
    ring_mcp = _make_landmark(0.46, 0.36)
    ring_pip = _make_landmark(0.46, 0.30)
    ring_dip = _make_landmark(0.45, 0.26)
    ring_tip_y = palm_center_y - (palm_center_y - ring_mcp.y) * finger_ratios["ring"]
    ring_tip = _make_landmark(0.44, ring_tip_y)

    # 17-20: pinky (MCP, PIP, DIP, tip)
    pinky_mcp = _make_landmark(0.45, 0.38)
    pinky_pip = _make_landmark(0.45, 0.32)
    pinky_dip = _make_landmark(0.44, 0.28)
    pinky_tip_y = palm_center_y - (palm_center_y - pinky_mcp.y) * finger_ratios["pinky"]
    pinky_tip = _make_landmark(0.43, pinky_tip_y)

    return [
        wrist,
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip,
        index_mcp, index_pip, index_dip, index_tip,
        middle_mcp, middle_pip, middle_dip, middle_tip,
        ring_mcp, ring_pip, ring_dip, ring_tip,
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip,
    ]


class TestFingerExtendedDetection:
    """Test the _is_finger_extended logic via the public count_fingers method."""

    def test_all_fingers_extended(self):
        """Open hand: all 4 fingers extended."""
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 1.5, "pinky": 1.5,
        })
        count = gr.count_fingers(landmarks)
        assert count == 4

    def test_all_fingers_folded(self):
        """Fist: all 4 fingers folded."""
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 0.4, "middle": 0.4, "ring": 0.4, "pinky": 0.4,
        })
        count = gr.count_fingers(landmarks)
        assert count == 0

    def test_one_finger_extended(self):
        """Only index finger extended."""
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 0.3, "ring": 0.3, "pinky": 0.3,
        })
        count = gr.count_fingers(landmarks)
        assert count == 1

    def test_two_fingers_extended(self):
        """Index and middle fingers extended."""
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 0.3, "pinky": 0.3,
        })
        count = gr.count_fingers(landmarks)
        assert count == 2

    def test_three_fingers_extended(self):
        """Index, middle, ring fingers extended."""
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 1.5, "pinky": 0.3,
        })
        count = gr.count_fingers(landmarks)
        assert count == 3


class TestFistDetection:
    def test_fist_detected(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 0.4, "middle": 0.4, "ring": 0.4, "pinky": 0.4,
        })
        assert gr.is_fist(landmarks) is True

    def test_open_hand_not_fist(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 1.5, "pinky": 1.5,
        })
        assert gr.is_fist(landmarks) is False


class TestOpenHandDetection:
    def test_open_hand_detected(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 1.5, "pinky": 1.5,
        })
        assert gr.is_open_hand(landmarks) is True

    def test_partial_hand_not_open(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 0.4, "pinky": 1.5,
        })
        assert gr.is_open_hand(landmarks) is False


class TestThumbsUpDetection:
    def test_thumbs_up(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(
            finger_ratios={"index": 0.3, "middle": 0.3, "ring": 0.3, "pinky": 0.3},
            thumb_tip_y_diff=-0.1,  # Thumb pointing up
        )
        assert gr.is_thumb_up(landmarks) is True

    def test_thumbs_down(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(
            finger_ratios={"index": 0.3, "middle": 0.3, "ring": 0.3, "pinky": 0.3},
            thumb_tip_y_diff=0.1,  # Thumb pointing down
        )
        assert gr.is_thumb_up(landmarks) is False


class TestGestureClassification:
    def test_classify_fist(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(
            finger_ratios={"index": 0.4, "middle": 0.4, "ring": 0.4, "pinky": 0.4},
            thumb_tip_y_diff=0.1,  # Thumb not pointing up
        )
        assert gr.classify(landmarks) == "fist"

    def test_classify_open_hand(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 1.5, "pinky": 1.5,
        })
        assert gr.classify(landmarks) == "open_hand"

    def test_classify_one_finger(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 0.3, "ring": 0.3, "pinky": 0.3,
        })
        assert gr.classify(landmarks) == "one_finger"

    def test_classify_two_fingers(self):
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(finger_ratios={
            "index": 1.5, "middle": 1.5, "ring": 0.3, "pinky": 0.3,
        })
        assert gr.classify(landmarks) == "two_fingers"

    def test_classify_ambiguous_ratios(self):
        """Ratios between 0.6-1.3 are treated as folded → classified as fist."""
        from src.gesture_recognizer import GestureRecognizer
        gr = GestureRecognizer()
        landmarks = _make_hands_landmarks(
            finger_ratios={"index": 0.8, "middle": 1.0, "ring": 0.7, "pinky": 0.9},
            thumb_tip_y_diff=0.1,  # Thumb not pointing up
        )
        # Ambiguous ratios (0.6-1.3) are treated as folded
        assert gr.classify(landmarks) == "fist"
