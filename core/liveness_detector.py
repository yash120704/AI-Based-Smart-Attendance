"""
Liveness Detection using blink detection (Eye Aspect Ratio)
"""
import logging
import numpy as np
import time
from config import ENABLE_RUNTIME_DEBUG

logger = logging.getLogger(__name__)


class LivenessDetector:
    """
    Detects blink-based liveness using Eye Aspect Ratio (EAR)
    """

    def __init__(self):
        """
        Initialize liveness detector
        """
        self.blink_detected = False
        self.start_time = None
        self.frame_counter = 0
        logger.info("LivenessDetector initialized")

    def start(self):
        """
        Start a new liveness detection session
        """
        self.start_time = time.time()
        self.blink_detected = False
        self.frame_counter = 0

    def compute_ear(self, eye):
        """
        Compute simplified Eye Aspect Ratio using 4 points: [inner, top, bottom, outer]

        Args:
            eye: 4 points representing the eye [inner, top, bottom, outer]

        Returns:
            ear: Eye Aspect Ratio (vertical / horizontal)
        """
        if len(eye) < 4:
            return 1.0
        
        # Vertical distance: distance from top to bottom
        vertical = np.linalg.norm(eye[1] - eye[2])
        
        # Horizontal distance: distance from inner to outer corner
        horizontal = np.linalg.norm(eye[0] - eye[3])
        
        # Simple ratio: vertical / horizontal
        # Closed eye: vertical approaches 0, ratio ~0
        # Open eye: vertical is larger, ratio ~0.5+
        ear = vertical / (horizontal + 1e-6)
        return ear

    def update(self, left_eye, right_eye, threshold, consec_frames):
        """
        Update blink detection with current eye position

        Args:
            left_eye: 6 points for left eye
            right_eye: 6 points for right eye
            threshold: EAR threshold for blink
            consec_frames: Consecutive frames needed for blink

        Returns:
            blink_detected: Bool indicating if blink was detected
        """
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        below_threshold = ear < threshold
        if ENABLE_RUNTIME_DEBUG:
            logger.debug(
                "EAR | L=%.3f R=%.3f Avg=%.3f | Thresh=%.3f | Below=%s | Counter=%s | Blink=%s",
                left_ear,
                right_ear,
                ear,
                threshold,
                below_threshold,
                self.frame_counter,
                self.blink_detected,
            )

        if ear < threshold:
            self.frame_counter += 1
            if self.frame_counter >= consec_frames:
                self.blink_detected = True
                if ENABLE_RUNTIME_DEBUG:
                    logger.debug("Blink detected with counter=%s", self.frame_counter)
        else:
            self.frame_counter = 0

        return self.blink_detected

    def is_timeout(self, limit):
        """
        Check if detection timeout exceeded

        Args:
            limit: Time limit in seconds

        Returns:
            bool: True if timeout exceeded
        """
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > limit

    def get_elapsed_time(self):
        """
        Get elapsed time since start

        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
