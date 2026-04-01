"""
Proxy Attendance Fraud Detection
"""
import logging
from config import BEHAVIOR_CONFIDENCE_THRESHOLD, PROXY_ALERT_THRESHOLD

logger = logging.getLogger(__name__)


class ProxyDetector:
    """
    Detects proxy attendance fraud by comparing face identity with behavior identity
    """

    def __init__(self, global_behavior_model=None):
        """
        Initialize proxy detector

        Args:
            global_behavior_model: Global behavior model for verification
        """
        self.global_behavior_model = global_behavior_model
        logger.info("ProxyDetector initialized")

    def check(self, face_name, behavior_prediction, behavior_confidence):
        """
        Check if attendance is legitimate or fraudulent (enhanced logic)

        Args:
            face_name: Face recognition result
            behavior_prediction: Behavior model prediction
            behavior_confidence: Confidence score from behavior model

        Returns:
            result: Dict with fraud detection results
        """
        result = {
            "is_proxy": False,
            "face_identity": face_name,
            "behavior_identity": behavior_prediction,
            "behavior_confidence": behavior_confidence,
            "alert_message": "",
            "status": "SUCCESS",
        }

        # Detailed logging
        logger.info(f"Proxy Check - Face: {face_name}, Behavior: {behavior_prediction}, Conf: {behavior_confidence:.3f}")

        if face_name == "Unknown":
            result["is_proxy"] = True
            result["alert_message"] = "Unknown face detected"
            result["status"] = "PROXY"
            logger.warning("REJECT: Unknown face")
            return result

        # Check 1: Is confidence critically low?
        if behavior_confidence < PROXY_ALERT_THRESHOLD:
            result["is_proxy"] = True
            result["alert_message"] = f"Confidence too low: {behavior_confidence:.3f} < {PROXY_ALERT_THRESHOLD}"
            result["status"] = "PROXY"
            logger.warning(f"REJECT: Confidence {behavior_confidence:.3f} below alert threshold {PROXY_ALERT_THRESHOLD}")
            return result

        # Check 2: Do face and behavior identities match?
        if face_name != behavior_prediction:
            result["is_proxy"] = True
            result["alert_message"] = f"Identity mismatch: Face={face_name}, Behavior={behavior_prediction} (conf: {behavior_confidence:.3f})"
            result["status"] = "PROXY"
            logger.warning(f"REJECT: Mismatch - Face {face_name} vs Behavior {behavior_prediction}")
            return result

        # Perfect match: Same person detected by both face and behavior
        result["alert_message"] = f"VERIFIED: {face_name} (confidence: {behavior_confidence:.3f})"
        result["status"] = "SUCCESS"
        logger.info(f"ACCEPT: {face_name} verified - confidence {behavior_confidence:.3f}")
        return result

    def check_behavioral_mismatch(self, face_name, behavior_name):
        """
        Check if face and behavior identities differ

        Args:
            face_name: Face recognition result
            behavior_name: Behavior model result

        Returns:
            bool: True if mismatch detected
        """
        return face_name != behavior_name and face_name != "Unknown"
