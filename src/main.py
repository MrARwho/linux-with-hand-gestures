"""
Entry point for hand gesture control app.
Handles webcam capture, CLI arguments, and the main processing loop.
"""

import argparse
import sys
import time
import threading
import cv2
import logging

from src.gesture_recognizer import GestureRecognizer
from src.action_dispatcher import ActionDispatcher
from src.numlock_monitor import NumLockMonitor
from src.config import DEBUG_WINDOW_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Linux Hand Gesture Control")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with visual overlay",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode (default in production)",
    )
    return parser.parse_args()


def capture_loop(args):
    """Main video capture and gesture detection loop."""
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        logger.error(f"Failed to open camera index {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recognizer = GestureRecognizer(static_mode=False, max_hands=1)
    dispatcher = ActionDispatcher()
    numlock_monitor = NumLockMonitor()

    logger.info("Camera opened successfully.")
    logger.info("Press 'q' to quit.")
    logger.info("Toggle NumLock to enable/disable gesture control.")

    numlock_monitor.add_callback(
        lambda on: logger.info(f"Gesture mode {'ENABLED' if on else 'DISABLED'}")
    )

    # Start NumLock monitor in a background thread
    monitor_thread = threading.Thread(target=numlock_monitor.start, daemon=True)
    monitor_thread.start()

    last_gesture = ""
    last_gesture_time = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            if not numlock_monitor.is_gesture_mode:
                output_frame = frame.copy()
            else:
                output_frame, landmarks, gesture = recognizer.process_frame(frame)

            if numlock_monitor.is_gesture_mode:
                debounce_ms = 500
                now = time.time()
                if gesture and gesture != "none" and gesture != last_gesture:
                    if now - last_gesture_time > debounce_ms / 1000:
                        dispatcher.dispatch(gesture)
                        last_gesture = gesture
                        last_gesture_time = now

            if args.debug:
                cv2.putText(
                    output_frame,
                    f"NumLock: {'ON' if numlock_monitor.is_gesture_mode else 'OFF'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                if numlock_monitor.is_gesture_mode and gesture and gesture != "none":
                    cv2.putText(
                        output_frame,
                        f"Gesture: {gesture}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(DEBUG_WINDOW_NAME, output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if numlock_monitor.is_gesture_mode and gesture and gesture != "none":
                    logger.info(f"Gesture: {gesture}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        numlock_monitor.stop()
        cap.release()
        recognizer.release()
        if args.debug:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    capture_loop(args)


if __name__ == "__main__":
    main()
