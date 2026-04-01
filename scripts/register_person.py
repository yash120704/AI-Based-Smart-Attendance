"""
Register a new person by capturing face images and timestamped behavior sequences.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BEHAVIOR_DATA_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FPS,
    KNOWN_FACES_DIR,
    REGISTRATION_SEQUENCE_STRIDE_SECONDS,
    WEBCAM_INDEX,
)
from core.pose_extractor import PoseExtractor
from utils.sequence_buffer import TimedSequenceBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def register_person(person_name):
    person_name = person_name.replace(" ", "_")
    person_dir = KNOWN_FACES_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    pose_extractor = PoseExtractor()
    logger.info(f"Registering person: {person_name}")

    logger.info("Step 1: Capturing 10 face images...")
    face_images_captured = 0
    frame_count = 0
    last_capture_frame = -60

    while face_images_captured < 10:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow(f"Registering {person_name} - Face Capture", frame)

        if frame_count - last_capture_frame >= (FPS * 2):
            image_path = person_dir / f"face_{face_images_captured + 1}.jpg"
            cv2.imwrite(str(image_path), frame)
            logger.info(f"Captured face image {face_images_captured + 1}/10: {image_path}")
            face_images_captured += 1
            last_capture_frame = frame_count

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Registration cancelled by user")
            cap.release()
            cv2.destroyAllWindows()
            return

    logger.info("Step 2: Capturing timestamped behavior sequences...")
    logger.info("Move naturally. The system will resample each saved sequence to exactly 30 frames over 1 second.")

    behavior_sequences = []
    timed_buffer = TimedSequenceBuffer()
    total_sequences = 0
    start_time = time.time()

    while total_sequences < 90:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        timestamp = time.perf_counter()
        frame = cv2.flip(frame, 1)
        pose_feature, _ = pose_extractor.extract(frame)
        timed_buffer.add(pose_feature, timestamp=timestamp)

        emitted_sequence = timed_buffer.emit_if_ready(REGISTRATION_SEQUENCE_STRIDE_SECONDS)
        if emitted_sequence is not None:
            behavior_sequences.append(emitted_sequence.reshape(-1))
            total_sequences += 1
            elapsed = time.time() - start_time
            eta_remaining = (elapsed / max(total_sequences, 1)) * (90 - total_sequences)
            logger.info(f"Captured behavior sequence {total_sequences}/90 (ETA: {eta_remaining:.1f}s)")

        progress = total_sequences / 90.0
        status_frame = frame.copy()
        cv2.putText(
            status_frame,
            f"Behavior Capture: {total_sequences}/90 ({progress * 100:.0f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(f"Registering {person_name} - Behavior", status_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Registration cancelled by user")
            break

    cap.release()
    cv2.destroyAllWindows()

    if behavior_sequences:
        behavior_sequences = np.array(behavior_sequences, dtype=np.float32)
        labels = np.array([person_name] * len(behavior_sequences), dtype=str)
        sequences_path = BEHAVIOR_DATA_DIR / f"{person_name}_sequences.npy"
        labels_path = BEHAVIOR_DATA_DIR / f"{person_name}_labels.npy"
        np.save(str(sequences_path), behavior_sequences)
        np.save(str(labels_path), labels)
        logger.info(f"Saved {len(behavior_sequences)} behavior sequences to {sequences_path}")
        logger.info(f"Saved labels to {labels_path}")
    else:
        logger.warning("No behavior sequences captured")

    logger.info(f"Registration complete for {person_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a new person for attendance")
    parser.add_argument("--name", type=str, required=True, help="Name of person to register")
    args = parser.parse_args()
    register_person(args.name)
