"""
Configuration for hand gesture control app.
Contains gesture detection thresholds, key mappings, and debounce timings.
"""

# --- Gesture Detection Thresholds ---

# Ratio of finger tip distance to palm center vs finger MCP distance to palm center
# If ratio < FINGER_CLOSED_THRESHOLD, finger is considered folded
FINGER_CLOSED_THRESHOLD = 0.6

# If ratio > FINGER_OPEN_THRESHOLD, finger is considered extended
FINGER_OPEN_THRESHOLD = 1.3

# Finger tip indices in MediaPipe landmarks (0=wrist, 4=thumb_tip, 8=index_tip, etc.)
FINGER_TIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

# Finger MCP (knuckle) indices
FINGER_MCP_INDICES = {
    "thumb": 2,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}

# Palm center landmark index (MediaPipe landmark 0 = wrist, we use average of 0,5,17)
PALM_CENTER_INDICES = [0, 5, 17]

# Thumb-specific: thumb tip vs thumb IP for direction detection
THUMB_IP_INDEX = 3
THUMB_MCP_INDEX = 2

# --- Debounce ---

# Minimum time (seconds) between triggering the same gesture
DEBOUNCE_INTERVAL = 0.5

# --- Swipe ---

# Number of frames to consider for swipe velocity calculation
SWIPE_WINDOW_FRAMES = 12

# Minimum velocity (pixels per frame) to count as a swipe
SWIPE_VELOCITY_THRESHOLD = 3.0

# --- Alt+Tab State ---

# Hand landmark X-coordinate to determine left vs right side of screen
# Used for swipe direction detection
SCREEN_WIDTH = 1920  # Will be overridden by actual screen detection

# --- Debug ---

# Draw landmarks and gestures on frame for debugging
DEBUG = False
DEBUG_WINDOW_NAME = "Hand Gesture Control"
