"""
FastAPI entry point for cloud deployments of Smart Attendance System.
"""
import asyncio
import base64
import logging
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    BEHAVIOR_CONFIDENCE_THRESHOLD,
    BEHAVIOR_OBSERVATION_TIME,
    BEHAVIOR_SEQUENCE_STRIDE_SECONDS,
    BLINK_TIME_LIMIT,
    MAX_ATTEMPTS,
    MIN_BEHAVIOR_DECISION_SECONDS,
    MIN_BEHAVIOR_MOTION_SCORE,
    MIN_BEHAVIOR_TEMPORAL_STD,
)
from core.attendance_logger import AttendanceLogger
from core.liveness_detector import LivenessDetector
from utils.feature_engineering import compute_motion_metrics
from utils.sequence_buffer import TimedSequenceBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Attendance API", version="1.0.0")

raw_allowed_origins = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in raw_allowed_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)


class VerifyFrameRequest(BaseModel):
    frame: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


EYE_AR_THRESHOLD = 0.34
EYE_AR_CONSEC_FRAMES = 1


class PersonState:
    """
    API-local copy of the main.py per-person state.
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


@dataclass
class VerifySession:
    session_id: str
    state: PersonState
    created_at: float
    last_seen: float


attendance_logger: Optional[AttendanceLogger] = None
face_detector: Optional[Any] = None
pose_extractor: Optional[Any] = None
behavior_model: Optional[Any] = None
behavior_model_loaded = False
vision_components_loaded = False

sessions: Dict[str, VerifySession] = {}
connected_websockets: Set[WebSocket] = set()
db_lock = threading.Lock()
vision_lock = threading.Lock()
processing_lock = asyncio.Lock()

SESSION_TIMEOUT_SECONDS = 60


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataframe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    clean_df = df.replace({np.nan: None})
    records = clean_df.to_dict(orient="records")

    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif isinstance(value, datetime):
                record[key] = value.isoformat()
            elif isinstance(value, date):
                record[key] = value.isoformat()
            elif isinstance(value, np.integer):
                record[key] = int(value)
            elif isinstance(value, np.floating):
                record[key] = float(value)
            elif isinstance(value, np.bool_):
                record[key] = bool(value)
    return records


def _require_db():
    if attendance_logger is None:
        raise HTTPException(status_code=503, detail="Attendance API is still initializing")


def _ensure_vision_components():
    global face_detector, pose_extractor, behavior_model, behavior_model_loaded, vision_components_loaded

    if face_detector is not None and pose_extractor is not None and behavior_model is not None:
        return

    with vision_lock:
        if face_detector is not None and pose_extractor is not None and behavior_model is not None:
            return

        logger.info("Lazy-loading vision and behavior components...")
        from core.behavior_model import BehaviorModel
        from core.detector import FaceDetector
        from core.pose_extractor import PoseExtractor

        face_detector = FaceDetector()
        pose_extractor = PoseExtractor()
        behavior_model = BehaviorModel()
        behavior_model_loaded = bool(behavior_model.is_trained)

        if not behavior_model_loaded:
            behavior_model_loaded = bool(behavior_model.load("global"))

        if attendance_logger is not None:
            for person_name in face_detector.known_names:
                attendance_logger.register_person(person_name)

        vision_components_loaded = True
        logger.info("Vision components loaded | behavior_model_loaded=%s", behavior_model_loaded)


def _run_script(script_name: str, args: Optional[List[str]] = None):
    args = args or []
    command = [sys.executable, str(_project_root() / "scripts" / script_name), *args]
    logger.info("Starting background script: %s", " ".join(command))
    subprocess.run(command, cwd=_project_root(), check=True)


def _initialize_database():
    global attendance_logger

    attendance_logger = AttendanceLogger()
    logger.info("FastAPI database initialized; vision stack will load on first verify-frame request")


async def _cleanup_sessions_loop():
    while True:
        await asyncio.sleep(15)
        now = time.time()
        expired_session_ids = [
            session_id
            for session_id, session in sessions.items()
            if now - session.last_seen > SESSION_TIMEOUT_SECONDS
        ]
        for session_id in expired_session_ids:
            sessions.pop(session_id, None)


@app.on_event("startup")
async def startup():
    await asyncio.to_thread(_initialize_database)
    asyncio.create_task(_cleanup_sessions_loop())


@app.on_event("shutdown")
async def shutdown():
    if attendance_logger is not None:
        attendance_logger.close()


def _decode_frame(frame_payload: str) -> np.ndarray:
    import cv2

    if "," in frame_payload:
        frame_payload = frame_payload.split(",", 1)[1]

    try:
        frame_bytes = base64.b64decode(frame_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 frame") from exc

    image_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Frame is not a valid JPEG image")
    return frame


def _session_for(session_id: str) -> VerifySession:
    now = time.time()
    session = sessions.get(session_id)
    if session is None:
        person_state = PersonState()
        session = VerifySession(
            session_id=session_id,
            state=person_state,
            created_at=now,
            last_seen=now,
        )
        sessions[session_id] = session
    else:
        session.last_seen = now
    return session


def _verification_response(
    stage: str,
    person_state: PersonState,
    message: str,
    attendance_marked: bool = False,
    decision: Optional[str] = None,
) -> Dict[str, Any]:
    elapsed = time.time() - person_state.state_start_time
    person_name = person_state.face_name or None
    return {
        "stage": stage,
        "person": person_name,
        "person_name": person_name,
        "face_confidence": float(person_state.face_confidence or 0.0),
        "behavior_confidence": float(person_state.behavior_confidence or 0.0),
        "elapsed_seconds": round(elapsed, 3),
        "attempt": int(person_state.attempts),
        "message": message,
        "attendance_marked": attendance_marked,
        "decision": decision or stage,
    }


def _attendance_event(person_state: PersonState, status: str, alert_message: str) -> Dict[str, Any]:
    return {
        "type": "attendance",
        "person_name": person_state.face_name,
        "timestamp": datetime.utcnow().isoformat(),
        "face_confidence": float(person_state.face_confidence or 0.0),
        "behavior_confidence": float(person_state.behavior_confidence or 0.0),
        "status": status,
        "alert_message": alert_message,
        "attempts": int(person_state.attempts),
        "blink_detected": bool(person_state.blink_detected),
    }


def _log_attendance_event(
    person_state: PersonState,
    behavior_confidence: float,
    is_proxy: bool,
    alert_message: str,
    status: str,
) -> Tuple[bool, Dict[str, Any]]:
    with db_lock:
        assert attendance_logger is not None
        result = attendance_logger.log(
            person_state.face_name,
            person_state.face_confidence,
            behavior_confidence,
            is_proxy=is_proxy,
            alert_message=alert_message,
            attempts=person_state.attempts,
            status=status,
            blink_detected=1 if person_state.blink_detected else 0,
        )
        attendance_logger.register_person(person_state.face_name)
        attendance_logger.update_person_attendance_count(person_state.face_name)

    marked = result.get("status") == "SUCCESS"
    return marked, _attendance_event(person_state, status, alert_message)


def _process_verify_frame(payload: VerifyFrameRequest) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    _require_db()
    _ensure_vision_components()
    assert attendance_logger is not None
    assert face_detector is not None
    assert pose_extractor is not None
    assert behavior_model is not None

    frame = _decode_frame(payload.frame)
    session = _session_for(payload.session_id)
    person_state = session.state
    capture_timestamp = time.perf_counter()
    h, w = frame.shape[:2]

    detections = face_detector.detect_and_identify(frame)
    if not detections:
        person_state.face_name = ""
        person_state.face_confidence = 0.0
        return _verification_response("FACE", person_state, "No face detected"), None

    detection = max(detections, key=lambda item: item["confidence"])
    person_state.face_name = detection["name"]
    person_state.face_confidence = detection["confidence"]
    bbox = detection["bbox"]

    if person_state.face_name == "Unknown":
        person_state.state = "UNKNOWN"
        person_state.last_is_proxy = False
        return _verification_response("RETRY", person_state, "Unknown face detected", decision="UNKNOWN_FACE"), None

    with db_lock:
        blocked = attendance_logger.is_person_blocked(person_state.face_name)
        already_marked = attendance_logger.has_marked_today(person_state.face_name)

    if blocked:
        person_state.state = "BLOCKED"
        person_state.blocked = True
        return _verification_response("BLOCKED", person_state, f"{person_state.face_name} is blocked", decision="BLOCKED"), None

    if already_marked and person_state.state not in {"SUCCESS", "BEHAVIOR", "WAIT_BLINK"}:
        person_state.state = "ALREADY_MARKED"
        return _verification_response(
            "SUCCESS",
            person_state,
            f"{person_state.face_name} already marked attendance today",
            attendance_marked=False,
            decision="ALREADY_MARKED",
        ), None

    if person_state.state in {"DETECTED", "UNKNOWN", "ALREADY_MARKED", "SUCCESS", "BLOCKED"}:
        person_state.start_blink_wait()

    pose_feature = np.zeros(99, dtype=np.float32)
    pose_results = None
    if person_state.state in {"WAIT_BLINK", "BEHAVIOR"}:
        pose_feature, pose_results = pose_extractor.extract(frame)

    if person_state.state == "WAIT_BLINK":
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
                    return _verification_response(
                        "BEHAVIOR",
                        person_state,
                        "Blink detected. Verifying behavior...",
                    ), None

        if person_state.liveness.is_timeout(BLINK_TIME_LIMIT):
            person_state.attempts += 1
            if person_state.attempts >= MAX_ATTEMPTS:
                person_state.state = "BLOCKED"
                person_state.blocked = True
                with db_lock:
                    attendance_logger.block_person(person_state.face_name)
                marked, event = _log_attendance_event(
                    person_state,
                    0.0,
                    is_proxy=True,
                    alert_message="Max attempts exceeded - no blink detected",
                    status="BLOCKED",
                )
                return _verification_response(
                    "BLOCKED",
                    person_state,
                    "Max attempts exceeded - no blink detected",
                    attendance_marked=marked,
                    decision="BLOCKED",
                ), event

            person_state.start_blink_wait()
            return _verification_response(
                "RETRY",
                person_state,
                f"Blink not detected. Retry attempt {person_state.attempts}/{MAX_ATTEMPTS}",
                decision="BLINK_RETRY",
            ), None

        return _verification_response("BLINK", person_state, "Waiting for blink..."), None

    if person_state.state == "BEHAVIOR":
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

        elapsed = time.time() - person_state.state_start_time
        status = None
        final_name = "Unknown"
        final_confidence = 0.0

        raw_sequence = person_state.behavior_buffer.emit_if_ready(BEHAVIOR_SEQUENCE_STRIDE_SECONDS)
        if raw_sequence is not None:
            person_state.behavior_sequences_emitted += 1
            motion_metrics = compute_motion_metrics(raw_sequence)
            log_live_behavior_diagnostics(behavior_model, person_state, raw_sequence, motion_metrics)

            motion_gate_passed = (
                motion_metrics["motion_score"] >= MIN_BEHAVIOR_MOTION_SCORE
                and motion_metrics["temporal_std"] >= MIN_BEHAVIOR_TEMPORAL_STD
            )

            if motion_gate_passed:
                all_confidences = behavior_model.predict_all_confidences(raw_sequence)
                if all_confidences:
                    predicted_person = max(all_confidences, key=all_confidences.get)
                    predicted_confidence = all_confidences[predicted_person]
                    person_state.behavior_name = predicted_person
                    person_state.behavior_confidence = predicted_confidence

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
            status, final_name, final_confidence = evaluate_behavior_votes(
                person_state.face_name,
                person_state.behavior_predictions,
                person_state.behavior_confidences,
            )

        if status is None:
            return _verification_response("BEHAVIOR", person_state, "Verifying behavior..."), None

        person_state.behavior_name = final_name
        person_state.behavior_confidence = final_confidence

        if status == "SUCCESS":
            person_state.state = "SUCCESS"
            person_state.last_is_proxy = False
            marked, event = _log_attendance_event(
                person_state,
                final_confidence,
                is_proxy=False,
                alert_message="Behavior verified - Attendance confirmed",
                status="SUCCESS",
            )
            return _verification_response(
                "SUCCESS",
                person_state,
                "Attendance confirmed",
                attendance_marked=marked,
                decision="SUCCESS",
            ), event

        person_state.attempts += 1
        if person_state.attempts >= MAX_ATTEMPTS:
            person_state.state = "BLOCKED"
            person_state.blocked = True
            with db_lock:
                attendance_logger.block_person(person_state.face_name)
            marked, event = _log_attendance_event(
                person_state,
                final_confidence,
                is_proxy=False,
                alert_message=f"BLOCKED after {MAX_ATTEMPTS} failed behavior verification attempts",
                status="BLOCKED",
            )
            return _verification_response(
                "BLOCKED",
                person_state,
                f"Blocked after {MAX_ATTEMPTS} failed behavior attempts",
                attendance_marked=marked,
                decision="BLOCKED",
            ), event

        person_state.start_blink_wait()
        return _verification_response(
            "RETRY",
            person_state,
            f"Behavior verification failed. Retry attempt {person_state.attempts}/{MAX_ATTEMPTS}",
            decision="BEHAVIOR_RETRY",
        ), None

    return _verification_response("FACE", person_state, "Detecting face..."), None


async def _broadcast_event(event: Dict[str, Any]):
    stale_clients = []
    for websocket in connected_websockets:
        try:
            await websocket.send_json(event)
        except Exception:
            stale_clients.append(websocket)
    for websocket in stale_clients:
        connected_websockets.discard(websocket)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": behavior_model_loaded,
        "vision_loaded": vision_components_loaded,
    }


@app.post("/api/register")
def register(payload: RegisterRequest, background_tasks: BackgroundTasks):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    background_tasks.add_task(_run_script, "register_person.py", ["--name", name])
    return {"status": "accepted", "message": f"Registration started for {name}"}


@app.post("/api/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_script, "train_behavior_models.py")
    return {"status": "accepted", "message": "Behavior model training started"}


@app.get("/api/attendance/today")
def attendance_today():
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        records = _dataframe_records(attendance_logger.get_today_attendance())
    return {"records": records}


@app.get("/api/attendance/history")
def attendance_history(
    start: date = Query(...),
    end: date = Query(...),
):
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        records = _dataframe_records(attendance_logger.get_attendance_range(start, end))
    return {"records": records}


@app.get("/api/persons")
def persons():
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        records = _dataframe_records(attendance_logger.get_persons_stats())
    return {"persons": records}


@app.post("/api/persons/{name}/block")
def block_person(name: str):
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        result = attendance_logger.block_person(name)
    return result


@app.post("/api/persons/{name}/unblock")
def unblock_person(name: str):
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        result = attendance_logger.unblock_person(name)
    return result


@app.post("/api/persons/{name}/reenable")
def reenable_person(name: str):
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        result = attendance_logger.mark_person_for_reattendance(name)
    return result


@app.get("/api/proxy-alerts")
def proxy_alerts():
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        records = _dataframe_records(attendance_logger.get_proxy_alerts())
    return {"alerts": records}


@app.get("/api/stats")
def stats():
    _require_db()
    with db_lock:
        assert attendance_logger is not None
        today_df = attendance_logger.get_today_attendance()
        persons_df = attendance_logger.get_persons_stats()
        proxy_alerts_count = attendance_logger.get_proxy_alert_count()

    total_attended_today = int((today_df["status"] == "SUCCESS").sum()) if not today_df.empty else 0
    if not persons_df.empty and "blocked" in persons_df.columns:
        active_persons = int((persons_df["blocked"].astype(bool) == False).sum())
    else:
        active_persons = 0

    return {
        "total_attended_today": total_attended_today,
        "fraud_alerts": proxy_alerts_count,
        "active_persons": active_persons,
        "fps": 0.0,
        "model_loaded": behavior_model_loaded,
    }


@app.post("/api/verify-frame")
async def verify_frame(payload: VerifyFrameRequest):
    async with processing_lock:
        response, event = await asyncio.to_thread(_process_verify_frame, payload)
    if event:
        await _broadcast_event(event)
    return response


@app.websocket("/ws/attendance-feed")
async def attendance_feed(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        await websocket.send_json({"type": "connected", "timestamp": datetime.utcnow().isoformat()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
    except Exception:
        connected_websockets.discard(websocket)
