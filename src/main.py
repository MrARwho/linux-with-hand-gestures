"""
Entry point for hand gesture control app.
Handles webcam capture, CLI arguments, and the main processing loop.
"""

import argparse
import sys
import time
import cv2
import logging

from src.gesture_recognizer import GestureRecognizer
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

    # Set camera resolution to 640x480 for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Use static mode (one-shot) for better tracking during movement
    recognizer = GestureRecognizer(static_mode=False, max_hands=1)

    logger.info("Camera opened successfully.")
    logger.info("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            output_frame, landmarks, gesture = recognizer.process_frame(frame)

            fps = f"{1/0.033:.1f}"  # Approximate FPS

            if args.debug:
                cv2.putText(
                    output_frame,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    output_frame,
                    f"Landmarks: {'Yes' if landmarks else 'No'}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(DEBUG_WINDOW_NAME, output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # In production mode, just log the gesture
                if gesture and gesture != "none":
                    logger.info(f"Gesture: {gesture}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        recognizer.release()
        if args.debug:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    capture_loop(args)


if __name__ == "__main__":
    main()
