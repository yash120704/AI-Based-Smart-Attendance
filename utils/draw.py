"""
Drawing utilities for visualization
"""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def draw_detection(frame, name, bbox, is_proxy, behavior_conf, face_conf=0.0):
    """
    Draw detection box and labels on frame

    Args:
        frame: Input frame (BGR)
        name: Person name
        bbox: Bounding box (top, right, bottom, left)
        is_proxy: Is proxy fraud detected
        behavior_conf: Behavior model confidence
        face_conf: Face recognition confidence

    Returns:
        frame: Frame with drawing
    """
    top, right, bottom, left = bbox

    if is_proxy:
        color = (0, 0, 255)  # Red for proxy
    elif behavior_conf < 0.6:
        color = (0, 255, 255)  # Yellow for suspicious
    else:
        color = (0, 255, 0)  # Green for legitimate

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    label = f"{name} | B:{behavior_conf*100:.0f}% | F:{face_conf*100:.0f}%"

    if is_proxy:
        label += " | ⚠ PROXY"

    (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    text_offset_x = left
    text_offset_y = top - 10 if top > 30 else bottom + 20
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 4, text_offset_y - text_height - 4))
    cv2.rectangle(frame, box_coords[0], box_coords[1], color, cv2.FILLED)
    cv2.putText(
        frame,
        label,
        (text_offset_x + 2, text_offset_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    return frame


def draw_status(frame, state, timer, attempts, blocked, person_name=""):
    """
    Draw status information on frame

    Args:
        frame: Input frame (BGR)
        state: Current state (WAIT_BLINK, BEHAVIOR, etc.)
        timer: Timer value in seconds
        attempts: Number of attempts
        blocked: Is person blocked
        person_name: Name of person

    Returns:
        frame: Frame with status drawn
    """
    h, w = frame.shape[:2]

    if blocked:
        color = (0, 0, 255)  # Red
        status_text = "ATTENDANCE BLOCKED"
    elif state == "WAIT_BLINK":
        color = (0, 165, 255)  # Orange
        status_text = f"State: BLINK DETECTION | Time: {int(timer)}s"
    elif state == "BEHAVIOR":
        color = (0, 255, 255)  # Cyan
        status_text = f"State: BEHAVIOR ANALYSIS | Time: {int(timer)}s"
    elif state == "ALREADY_MARKED":
        color = (255, 0, 255)  # Magenta
        status_text = "✅ ATTENDANCE ALREADY MARKED"
    else:
        color = (0, 255, 0)  # Green
        status_text = f"State: {state} | Time: {int(timer)}s"

    attempt_text = f"Attempts: {attempts}/{3}"

    y_pos = 40
    cv2.putText(frame, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_pos += 30
    cv2.putText(frame, attempt_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if person_name:
        y_pos += 30
        cv2.putText(
            frame,
            f"Person: {person_name}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return frame


def draw_stats_overlay(frame, fps, total_today, proxy_count):
    """
    Draw statistics overlay on frame

    Args:
        frame: Input frame (BGR)
        fps: Frames per second
        total_today: Total attendance today
        proxy_count: Proxy alert count

    Returns:
        frame: Frame with stats overlay
    """
    h, w = frame.shape[:2]

    stats = [
        f"FPS: {fps:.1f}",
        f"Attendance Today: {total_today}",
        f"Proxy Alerts: {proxy_count}",
    ]

    y_offset = h - 100
    for i, stat in enumerate(stats):
        cv2.putText(
            frame,
            stat,
            (20, y_offset + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

    return frame


def draw_blink_status(frame, blink_detected, ear, threshold):
    """
    Draw blink detection status

    Args:
        frame: Input frame (BGR)
        blink_detected: Was blink detected
        ear: Current Eye Aspect Ratio
        threshold: EAR threshold

    Returns:
        frame: Frame with blink status
    """

    if blink_detected:
        color = (0, 255, 0)
        status = "✅ BLINK DETECTED"
    else:
        color = (0, 0, 255) if ear < threshold else (0, 165, 255)
        status = "Blink: NOT YET"

    cv2.putText(
        frame,
        status,
        (frame.shape[1] - 300, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"EAR: {ear:.3f} (Threshold: {threshold:.3f})",
        (frame.shape[1] - 300, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return frame


def draw_face_label(frame, name, confidence, bbox):
    """
    Draw face label and confidence

    Args:
        frame: Input frame (BGR)
        name: Person name
        confidence: Face confidence
        bbox: Bounding box (top, right, bottom, left)

    Returns:
        frame: Frame with label
    """
    top, right, bottom, left = bbox

    label = f"{name} ({confidence*100:.1f}%)"
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(
        frame,
        label,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

    return frame


def put_text(frame, text, position, font_size=0.6, color=(255, 255, 255), thickness=1):
    """
    Put text on frame

    Args:
        frame: Input frame
        text: Text to draw
        position: (x, y) position
        font_size: Font size multiplier
        color: Color in BGR
        thickness: Text thickness

    Returns:
        annotated_frame: Frame with text
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    return frame


def draw_progress_bar(frame, progress, x=20, y=None, width=200, height=20, color=(0, 255, 0)):
    """
    Draw progress bar on frame

    Args:
        frame: Input frame
        progress: Progress value (0.0 to 1.0)
        x: X position
        y: Y position (default: frame height - 50)
        width: Bar width
        height: Bar height
        color: Bar color

    Returns:
        annotated_frame: Frame with progress bar
    """
    if y is None:
        y = frame.shape[0] - 50

    progress = np.clip(progress, 0, 1)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)
    filled_width = int(width * progress)
    if filled_width > 0:
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)

    percent_text = f"{progress*100:.0f}%"
    cv2.putText(frame, percent_text, (x + width + 10, y + height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
