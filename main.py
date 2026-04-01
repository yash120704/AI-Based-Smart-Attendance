"""
Main attendance system - webcam loop with state machine.
"""
import logging
import platform
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import (
    BEHAVIOR_CONFIDENCE_THRESHOLD,
    BEHAVIOR_OBSERVATION_TIME,
    BEHAVIOR_SEQUENCE_STRIDE_SECONDS,
    BLINK_TIME_LIMIT,
    CAMERA_BUFFER_SIZE,
    CAMERA_USE_MJPG,
    EYE_AR_CONSEC_FRAMES,
    EYE_AR_THRESHOLD,
    FACE_DETECTION_INTERVAL,
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_ATTEMPTS,
    MIN_BEHAVIOR_DECISION_SECONDS,
    MIN_BEHAVIOR_MOTION_SCORE,
    MIN_BEHAVIOR_TEMPORAL_STD,
    MODELS_DIR,
    STATS_REFRESH_SECONDS,
    WEBCAM_INDEX,
)
from core.attendance_logger import AttendanceLogger
from core.behavior_model import BehaviorModel
from core.detector import FaceDetector
from core.liveness_detector import LivenessDetector
from core.pose_extractor import PoseExtractor
from core.tracker import CentroidTracker
from utils.draw import (
    draw_blink_status,
    draw_detection,
    draw_progress_bar,
    draw_stats_overlay,
    draw_status,
)
from utils.feature_engineering import compute_motion_metrics
from utils.sequence_buffer import TimedSequenceBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EYE_AR_THRESHOLD = 0.34
EYE_AR_CONSEC_FRAMES = 1


class PersonState:
    """
    Manages state for a tracked person.
    """

    def __init__(self):
        self.state = "DETECTED"
        self.attempts = 0
        self.blocked = False
        self.liveness = LivenessDetector()
        self.behavior_buffer = TimedSequenceBuffer()
        self.state_start_time = time.time()
        self.face_name = ""
        self.face_confidence = 0.0
        self.behavior_name = ""
        self.behavior_confidence = 0.0
        self.blink_detected = False
        self.behavior_predictions = []
        self.behavior_confidences = []
        self.last_is_proxy = False
        self.last_diagnostic_log_time = 0.0
        self.last_motion_gate_log_time = 0.0
        self.last_behavior_skip_log_time = 0.0
        self.behavior_pose_missing_frames = 0
        self.behavior_not_centered_frames = 0
        self.behavior_accepted_frames = 0
        self.behavior_sequences_emitted = 0

    def reset_behavior_session(self):
        self.behavior_buffer.reset()
        self.behavior_predictions = []
        self.behavior_confidences = []
        self.behavior_name = ""
        self.behavior_confidence = 0.0
        self.last_is_proxy = False
        self.last_diagnostic_log_time = 0.0
        self.last_motion_gate_log_time = 0.0
        self.last_behavior_skip_log_time = 0.0
        self.behavior_pose_missing_frames = 0
        self.behavior_not_centered_frames = 0
        self.behavior_accepted_frames = 0
        self.behavior_sequences_emitted = 0

    def start_behavior(self):
        self.state = "BEHAVIOR"
        self.blink_detected = True
        self.reset_behavior_session()
        self.state_start_time = time.time()

    def start_blink_wait(self):
        self.state = "WAIT_BLINK"
        self.liveness.start()
        self.blink_detected = False
        self.reset_behavior_session()
        self.state_start_time = time.time()


def is_face_centered(bbox, frame_w, frame_h, center_margin=0.30):
    top, right, bottom, left = bbox
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0

    x_min = frame_w * (0.5 - center_margin / 2.0)
    x_max = frame_w * (0.5 + center_margin / 2.0)
    y_min = frame_h * (0.5 - center_margin / 2.0)
    y_max = frame_h * (0.5 + center_margin / 2.0)

    return x_min <= cx <= x_max and y_min <= cy <= y_max


def get_face_landmarks_eye_coords(face_landmarks):
    if face_landmarks is None:
        return None, None

    try:
        right_eye = np.array(
            [
                [face_landmarks[33].x, face_landmarks[33].y],
                [face_landmarks[159].x, face_landmarks[159].y],
                [face_landmarks[145].x, face_landmarks[145].y],
                [face_landmarks[133].x, face_landmarks[133].y],
            ]
        )
        left_eye = np.array(
            [
                [face_landmarks[263].x, face_landmarks[263].y],
                [face_landmarks[388].x, face_landmarks[388].y],
                [face_landmarks[374].x, face_landmarks[374].y],
                [face_landmarks[362].x, face_landmarks[362].y],
            ]
        )
        return left_eye, right_eye
    except (IndexError, AttributeError):
        return None, None


def open_webcam(camera_index):
    backends_to_try = []
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        backends_to_try.append(cv2.CAP_DSHOW)
    backends_to_try.append(None)

    for backend in backends_to_try:
        cap = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(camera_index)


def configure_capture(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if CAMERA_USE_MJPG and hasattr(cv2, "VideoWriter_fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    except cv2.error:
        pass


def map_detections_to_trackers(detections, tracked_objects):
    if not detections or not tracked_objects:
        return {}

    tracker_ids = list(tracked_objects.keys())
    tracker_centroids = np.array([tracked_objects[tracker_id] for tracker_id in tracker_ids], dtype=np.float32)
    detection_centroids = np.array(
        [
            [(detection["bbox"][3] + detection["bbox"][1]) / 2.0, (detection["bbox"][0] + detection["bbox"][2]) / 2.0]
            for detection in detections
        ],
        dtype=np.float32,
    )

    distance_matrix = np.linalg.norm(
        tracker_centroids[:, np.newaxis, :] - detection_centroids[np.newaxis, :, :],
        axis=2,
    )

    mapping = {}
    used_tracker_indexes = set()
    used_detection_indexes = set()

    while len(used_tracker_indexes) < len(tracker_ids) and len(used_detection_indexes) < len(detections):
        min_distance = None
        best_tracker_index = None
        best_detection_index = None

        for tracker_index in range(distance_matrix.shape[0]):
            if tracker_index in used_tracker_indexes:
                continue
            for detection_index in range(distance_matrix.shape[1]):
                if detection_index in used_detection_indexes:
                    continue
                current_distance = distance_matrix[tracker_index, detection_index]
                if min_distance is None or current_distance < min_distance:
                    min_distance = current_distance
                    best_tracker_index = tracker_index
                    best_detection_index = detection_index

        if best_tracker_index is None:
            break

        tracker_id = tracker_ids[best_tracker_index]
        mapping[tracker_id] = detections[best_detection_index]
        used_tracker_indexes.add(best_tracker_index)
        used_detection_indexes.add(best_detection_index)

    return mapping


def evaluate_behavior_votes(face_name, predictions, confidences):
    if not predictions:
        return "RETRY", "Unknown", 0.0

    counts = Counter(predictions)
    final_name, vote_count = counts.most_common(1)[0]
    selected_confidences = [
        confidence
        for prediction, confidence in zip(predictions, confidences)
        if prediction == final_name
    ]
    final_confidence = float(np.mean(selected_confidences)) if selected_confidences else 0.0

    if len(predictions) < 5:
        return "RETRY", final_name, final_confidence
    if final_name != face_name:
        return "RETRY", final_name, final_confidence
    if vote_count < 3:
        return "RETRY", final_name, final_confidence
    if final_confidence < BEHAVIOR_CONFIDENCE_THRESHOLD:
        return "RETRY", final_name, final_confidence

    return "SUCCESS", final_name, final_confidence


def log_live_behavior_diagnostics(global_model, person_state, raw_sequence, motion_metrics):
    diagnostics = global_model.get_sequence_diagnostics(raw_sequence)
    now = time.time()

    should_log = (now - person_state.last_diagnostic_log_time) >= 1.0
    if diagnostics.get("feature_mean_gap") is not None and diagnostics["feature_mean_gap"] > 0.75:
        should_log = True
    if (
        motion_metrics["motion_score"] < MIN_BEHAVIOR_MOTION_SCORE
        or motion_metrics["temporal_std"] < MIN_BEHAVIOR_TEMPORAL_STD
    ):
        should_log = True

    if should_log:
        logger.info(
            "Behavior diagnostic | live_feature_mean=%.4f train_feature_mean=%.4f "
            "| live_feature_std=%.4f train_feature_std=%.4f | feature_mean_gap=%.4f "
            "| motion_score=%.4f peak_motion=%.4f temporal_std=%.4f | label_map=%s",
            diagnostics["live_feature_mean"],
            diagnostics["training_feature_mean"] or 0.0,
            diagnostics["live_feature_std"],
            diagnostics["training_feature_std"] or 0.0,
            diagnostics["feature_mean_gap"] or 0.0,
            motion_metrics["motion_score"],
            motion_metrics["peak_motion"],
            motion_metrics["temporal_std"],
            global_model.label_map,
        )
        person_state.last_diagnostic_log_time = now


def log_behavior_collection_status(person_state, centered, pose_missing):
    now = time.time()
    if now - person_state.last_behavior_skip_log_time < 1.0:
        return

    logger.info(
        "Behavior collection for %s | centered=%s pose_missing=%s accepted_frames=%d "
        "pose_missing_frames=%d not_centered_frames=%d emitted_sequences=%d buffer_ready=%s",
        person_state.face_name,
        centered,
        pose_missing,
        person_state.behavior_accepted_frames,
        person_state.behavior_pose_missing_frames,
        person_state.behavior_not_centered_frames,
        person_state.behavior_sequences_emitted,
        person_state.behavior_buffer.is_ready(),
    )
    person_state.last_behavior_skip_log_time = now


def main():
    logger.info("Initializing Smart Attendance System...")

    pose_extractor = PoseExtractor()
    face_detector = FaceDetector()
    attendance_logger = AttendanceLogger()
    tracker = CentroidTracker()

    global_model = BehaviorModel()
    global_model_path = Path(MODELS_DIR) / "global_behavior.h5"
    if global_model_path.exists():
        global_model.load("global", MODELS_DIR)
        logger.info("Global behavior model loaded")
        logger.info(f"Behavior label map loaded at inference: {global_model.label_map}")
    else:
        logger.warning("Global behavior model not found - train first with: python scripts/train_behavior_models.py")

    cap = open_webcam(WEBCAM_INDEX)
    configure_capture(cap)

    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return

    logger.info("Webcam opened successfully")
    logger.info(face_detector.get_runtime_summary())
    logger.info(global_model.get_runtime_summary())

    person_states = {}
    frame_count = 0
    fps = 0.0
    detection_interval = max(1, FACE_DETECTION_INTERVAL)
    stats_refresh_seconds = max(0.1, STATS_REFRESH_SECONDS)
    fps_window_start = time.perf_counter()
    cached_detections = []
    today_count = attendance_logger.get_today_attendance_count()
    proxy_count = attendance_logger.get_proxy_alert_count()
    last_stats_refresh = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break

            capture_timestamp = time.perf_counter()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if frame_count % detection_interval == 0 or not cached_detections:
                cached_detections = face_detector.detect_and_identify(frame)
            detections = cached_detections

            tracked_objects = tracker.update([detection["bbox"] for detection in detections])
            detection_dict = map_detections_to_trackers(detections, tracked_objects)

            needs_pose_processing = any(
                tracker_id in detection_dict
                and tracker_id in person_states
                and person_states[tracker_id].state in {"WAIT_BLINK", "BEHAVIOR"}
                for tracker_id in tracked_objects
            )

            pose_feature = np.zeros(99, dtype=np.float32)
            pose_results = None
            annotated_frame = frame.copy()

            if detections and needs_pose_processing:
                pose_feature, pose_results = pose_extractor.extract(frame)
                annotated_frame = pose_extractor.draw_skeleton(annotated_frame, pose_results)

            now = time.perf_counter()
            if now - last_stats_refresh >= stats_refresh_seconds:
                today_count = attendance_logger.get_today_attendance_count()
                proxy_count = attendance_logger.get_proxy_alert_count()
                last_stats_refresh = now

            for tracker_id, centroid in tracked_objects.items():
                if tracker_id not in person_states:
                    person_states[tracker_id] = PersonState()

                person_state = person_states[tracker_id]
                detection = detection_dict.get(tracker_id)
                if detection is None:
                    continue

                person_state.face_name = detection["name"]
                person_state.face_confidence = detection["confidence"]
                bbox = detection["bbox"]

                if attendance_logger.is_person_blocked(person_state.face_name):
                    person_state.state = "BLOCKED"
                    person_state.blocked = True
                elif person_state.face_name == "Unknown":
                    person_state.state = "UNKNOWN"
                    person_state.last_is_proxy = False
                elif person_state.state in {"DETECTED", "UNKNOWN"}:
                    already_marked = attendance_logger.has_marked_today(person_state.face_name)
                    if already_marked:
                        person_state.state = "ALREADY_MARKED"
                    else:
                        person_state.start_blink_wait()

                if person_state.state == "WAIT_BLINK":
                    elapsed = person_state.liveness.get_elapsed_time()
                    face_landmarks = pose_extractor.get_face_landmarks(pose_results)
                    if face_landmarks:
                        left_eye, right_eye = get_face_landmarks_eye_coords(face_landmarks)
                        if left_eye is not None and right_eye is not None:
                            blink_detected = person_state.liveness.update(
                                left_eye,
                                right_eye,
                                EYE_AR_THRESHOLD,
                                EYE_AR_CONSEC_FRAMES,
                            )
                            if blink_detected:
                                person_state.start_behavior()
                                logger.info(f"Blink detected for {person_state.face_name}")

                    if person_state.liveness.is_timeout(BLINK_TIME_LIMIT):
                        person_state.attempts += 1
                        logger.info(
                            f"Blink timeout for {person_state.face_name}, attempts: "
                            f"{person_state.attempts}/{MAX_ATTEMPTS}"
                        )
                        if person_state.attempts >= MAX_ATTEMPTS:
                            person_state.state = "BLOCKED"
                            person_state.blocked = True
                            attendance_logger.block_person(person_state.face_name)
                            attendance_logger.log(
                                person_state.face_name,
                                person_state.face_confidence,
                                0.0,
                                is_proxy=True,
                                alert_message="Max attempts exceeded - no blink detected",
                                attempts=person_state.attempts,
                                status="BLOCKED",
                                blink_detected=0,
                            )
                            logger.warning(f"Person {person_state.face_name} BLOCKED")
                        else:
                            person_state.start_blink_wait()

                    annotated_frame = draw_blink_status(
                        annotated_frame,
                        person_state.blink_detected,
                        0.3,
                        EYE_AR_THRESHOLD,
                    )

                elif person_state.state == "BEHAVIOR":
                    pose_missing = np.allclose(pose_feature, 0.0)
                    centered = is_face_centered(bbox, w, h)
                    if (not pose_missing) and centered:
                        person_state.behavior_buffer.add(pose_feature, timestamp=capture_timestamp)
                        person_state.behavior_accepted_frames += 1
                    else:
                        if pose_missing:
                            person_state.behavior_pose_missing_frames += 1
                        if not centered:
                            person_state.behavior_not_centered_frames += 1
                        log_behavior_collection_status(person_state, centered, pose_missing)

                    elapsed = time.time() - person_state.state_start_time
                    status = None
                    final_name = "Unknown"
                    final_confidence = 0.0

                    raw_sequence = person_state.behavior_buffer.emit_if_ready(BEHAVIOR_SEQUENCE_STRIDE_SECONDS)
                    if raw_sequence is not None:
                        person_state.behavior_sequences_emitted += 1
                        motion_metrics = compute_motion_metrics(raw_sequence)
                        log_live_behavior_diagnostics(
                            global_model,
                            person_state,
                            raw_sequence,
                            motion_metrics,
                        )

                        motion_gate_passed = (
                            motion_metrics["motion_score"] >= MIN_BEHAVIOR_MOTION_SCORE
                            and motion_metrics["temporal_std"] >= MIN_BEHAVIOR_TEMPORAL_STD
                        )

                        if not motion_gate_passed:
                            gate_log_time = time.time()
                            if gate_log_time - person_state.last_motion_gate_log_time >= 1.0:
                                logger.info(
                                    "Behavior motion gate not met for %s | motion_score=%.4f "
                                    "peak_motion=%.4f temporal_std=%.4f",
                                    person_state.face_name,
                                    motion_metrics["motion_score"],
                                    motion_metrics["peak_motion"],
                                    motion_metrics["temporal_std"],
                                )
                                person_state.last_motion_gate_log_time = gate_log_time
                        else:
                            all_confidences = global_model.predict_all_confidences(raw_sequence)

                            if all_confidences:
                                predicted_person = max(all_confidences, key=all_confidences.get)
                                predicted_confidence = all_confidences[predicted_person]
                                if predicted_confidence >= BEHAVIOR_CONFIDENCE_THRESHOLD:
                                    person_state.behavior_predictions.append(predicted_person)
                                    person_state.behavior_confidences.append(predicted_confidence)

                                    face_votes = [
                                        confidence
                                        for prediction, confidence in zip(
                                            person_state.behavior_predictions,
                                            person_state.behavior_confidences,
                                        )
                                        if prediction == person_state.face_name
                                    ]
                                    if elapsed >= MIN_BEHAVIOR_DECISION_SECONDS and len(face_votes) >= 8:
                                        status = "SUCCESS"
                                        final_name = person_state.face_name
                                        final_confidence = float(np.mean(face_votes))

                    if status is None and elapsed >= BEHAVIOR_OBSERVATION_TIME:
                        if person_state.behavior_sequences_emitted == 0:
                            logger.info(
                                "Behavior timed out before any sequence was emitted for %s | accepted_frames=%d "
                                "pose_missing_frames=%d not_centered_frames=%d",
                                person_state.face_name,
                                person_state.behavior_accepted_frames,
                                person_state.behavior_pose_missing_frames,
                                person_state.behavior_not_centered_frames,
                            )
                        status, final_name, final_confidence = evaluate_behavior_votes(
                            person_state.face_name,
                            person_state.behavior_predictions,
                            person_state.behavior_confidences,
                        )

                    if status is not None:
                        person_state.behavior_name = final_name
                        person_state.behavior_confidence = final_confidence

                        if status == "SUCCESS":
                            person_state.state = "SUCCESS"
                            person_state.last_is_proxy = False
                            attendance_logger.log(
                                person_state.face_name,
                                person_state.face_confidence,
                                final_confidence,
                                is_proxy=False,
                                alert_message="Behavior verified - Attendance confirmed",
                                attempts=person_state.attempts,
                                status="SUCCESS",
                                blink_detected=1 if person_state.blink_detected else 0,
                            )
                            logger.info(f"SUCCESS: {person_state.face_name} attendance verified")
                            attendance_logger.register_person(person_state.face_name)
                            attendance_logger.update_person_attendance_count(person_state.face_name)
                        else:
                            person_state.attempts += 1
                            person_state.last_is_proxy = False
                            logger.info(
                                f"Attempt {person_state.attempts}/{MAX_ATTEMPTS}: "
                                f"{person_state.face_name} - RETRY"
                            )
                            if person_state.attempts >= MAX_ATTEMPTS:
                                person_state.state = "BLOCKED"
                                person_state.blocked = True
                                attendance_logger.block_person(person_state.face_name)
                                attendance_logger.log(
                                    person_state.face_name,
                                    person_state.face_confidence,
                                    final_confidence,
                                    is_proxy=False,
                                    alert_message=f"BLOCKED after {MAX_ATTEMPTS} failed behavior verification attempts",
                                    attempts=person_state.attempts,
                                    status="BLOCKED",
                                    blink_detected=1 if person_state.blink_detected else 0,
                                )
                                logger.warning(f"BLOCKED: {person_state.face_name} after {MAX_ATTEMPTS} attempts")
                            else:
                                person_state.start_blink_wait()
                                logger.info(f"Retrying: {person_state.face_name} (Attempt {person_state.attempts})")

                elapsed = time.time() - person_state.state_start_time
                if person_state.state in ["WAIT_BLINK", "BEHAVIOR"]:
                    progress_limit = BLINK_TIME_LIMIT if person_state.state == "WAIT_BLINK" else BEHAVIOR_OBSERVATION_TIME
                    annotated_frame = draw_progress_bar(
                        annotated_frame,
                        elapsed / progress_limit,
                        x=20,
                        y=80,
                        color=(0, 255, 0),
                    )

                annotated_frame = draw_detection(
                    annotated_frame,
                    person_state.face_name,
                    bbox,
                    person_state.last_is_proxy,
                    person_state.behavior_confidence,
                    person_state.face_confidence,
                )
                annotated_frame = draw_status(
                    annotated_frame,
                    person_state.state,
                    elapsed,
                    person_state.attempts,
                    person_state.blocked,
                    person_state.face_name,
                )

            annotated_frame = draw_stats_overlay(annotated_frame, fps, today_count, proxy_count)

            try:
                cv2.imshow("Smart Attendance System", annotated_frame)
            except cv2.error as exc:
                logger.debug(f"Display error (expected in headless environments): {exc}")

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.perf_counter() - fps_window_start
                fps = 30 / elapsed if elapsed > 0 else 0.0
                fps_window_start = time.perf_counter()

            try:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Exiting attendance system")
                    break
            except cv2.error:
                pass

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.error(f"Error in main loop: {exc}", exc_info=True)
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        attendance_logger.close()
        logger.info("Attendance system shut down")


if __name__ == "__main__":
    main()
