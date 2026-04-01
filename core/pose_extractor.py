"""
Pose extraction using MediaPipe Holistic with robust normalization.
"""
import logging

import cv2
import mediapipe as mp
import numpy as np

from config import POSE_PROCESS_SCALE

logger = logging.getLogger(__name__)

NOSE_IDX = 0
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24


def _landmarks_to_array(pose_landmarks):
    """
    Convert MediaPipe landmarks to a (33, 3) array.
    """
    coords = np.zeros((33, 3), dtype=np.float32)
    visibility = np.zeros(33, dtype=np.float32)

    for idx, landmark in enumerate(pose_landmarks.landmark):
        coords[idx] = [landmark.x, landmark.y, landmark.z]
        visibility[idx] = getattr(landmark, "visibility", 1.0)

    return coords, visibility


def normalize_landmarks(coords, visibility=None):
    """
    Make pose landmarks translation and scale invariant.

    Translation:
        Center at mid-hip when available, otherwise fall back to the nose.

    Scale:
        Use the larger of shoulder width and torso height, with safe fallbacks.
    """
    coords = np.asarray(coords, dtype=np.float32).reshape(33, 3)
    visibility = np.ones(33, dtype=np.float32) if visibility is None else np.asarray(visibility, dtype=np.float32)

    def visible(index):
        return visibility[index] > 0.3

    if visible(LEFT_HIP_IDX) and visible(RIGHT_HIP_IDX):
        center = (coords[LEFT_HIP_IDX] + coords[RIGHT_HIP_IDX]) / 2.0
    else:
        center = coords[NOSE_IDX]

    normalized = coords - center

    shoulder_width = 0.0
    if visible(LEFT_SHOULDER_IDX) and visible(RIGHT_SHOULDER_IDX):
        shoulder_width = float(np.linalg.norm(coords[LEFT_SHOULDER_IDX] - coords[RIGHT_SHOULDER_IDX]))

    torso_height = 0.0
    if visible(LEFT_SHOULDER_IDX) and visible(RIGHT_SHOULDER_IDX) and visible(LEFT_HIP_IDX) and visible(RIGHT_HIP_IDX):
        shoulder_mid = (coords[LEFT_SHOULDER_IDX] + coords[RIGHT_SHOULDER_IDX]) / 2.0
        hip_mid = (coords[LEFT_HIP_IDX] + coords[RIGHT_HIP_IDX]) / 2.0
        torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))

    scale = max(shoulder_width, torso_height, 1e-3)
    normalized /= scale

    return normalized.astype(np.float32).flatten()


class PoseExtractor:
    """
    Extracts normalized pose landmarks using MediaPipe Holistic.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, process_scale=POSE_PROCESS_SCALE):
        self.process_scale = float(process_scale)
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False,
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        logger.info("PoseExtractor initialized")

    def extract(self, frame):
        """
        Extract normalized pose landmarks from a frame.

        Returns:
            feature_vector: Flattened array of shape (99,)
            results: MediaPipe holistic results
        """
        process_frame = frame
        if self.process_scale < 0.999:
            process_frame = cv2.resize(
                frame,
                (0, 0),
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_AREA,
            )

        frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        feature_vector = np.zeros(99, dtype=np.float32)
        if results.pose_landmarks is not None:
            coords, visibility = _landmarks_to_array(results.pose_landmarks)
            feature_vector = normalize_landmarks(coords, visibility)
        else:
            logger.debug("No pose landmarks detected in frame")

        return feature_vector, results

    def draw_skeleton(self, frame, results):
        """
        Draw pose skeleton on frame.
        """
        if results is None:
            return frame

        if results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

        if results.left_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        if results.right_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        return frame

    def get_face_landmarks(self, results):
        """
        Get face landmarks for eye detection.
        """
        if results is not None and results.face_landmarks:
            return results.face_landmarks.landmark
        return None

    def __del__(self):
        if hasattr(self, "holistic"):
            self.holistic.close()
