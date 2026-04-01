"""
Face Detection and Recognition using face_recognition library
"""
import logging
import numpy as np
import face_recognition
import cv2
from pathlib import Path
from config import (
    KNOWN_FACES_DIR,
    FACE_RECOGNITION_TOLERANCE,
    FACE_DETECTION_MODEL,
    FACE_DETECTION_SCALE,
    FACE_DETECTION_UPSAMPLE,
    USE_GPU_IF_AVAILABLE,
)

try:
    import dlib
except ImportError:  # pragma: no cover - depends on local runtime
    dlib = None

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects and identifies faces using face_recognition library
    """

    def __init__(
        self,
        known_faces_dir=KNOWN_FACES_DIR,
        tolerance=FACE_RECOGNITION_TOLERANCE,
        detection_model=FACE_DETECTION_MODEL,
        process_scale=FACE_DETECTION_SCALE,
        use_gpu=USE_GPU_IF_AVAILABLE,
        upsample_times=FACE_DETECTION_UPSAMPLE,
    ):
        """
        Initialize face detector with known face encodings

        Args:
            known_faces_dir: Directory containing subdirectories for each person
            tolerance: Distance tolerance for face matching
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.known_encodings = {}
        self.known_encoding_matrix = np.empty((0, 128), dtype=np.float32)
        self.known_names = []
        self.process_scale = float(process_scale)
        self.upsample_times = int(upsample_times)
        self.dlib_cuda_available = self._is_dlib_cuda_available() if use_gpu else False
        self.detection_model = self._resolve_detection_model(detection_model)
        self._runtime_usage_logged = False
        self.load_known_faces()
        logger.info(
            "FaceDetector initialized with %s known persons | model=%s | dlib_cuda=%s | scale=%.2f",
            len(self.known_names),
            self.detection_model,
            self.dlib_cuda_available,
            self.process_scale,
        )

    def uses_gpu(self):
        """
        Return whether the active face-detection path is GPU-backed.

        Returns:
            bool: True when dlib CUDA and CNN detection are active
        """
        return self.dlib_cuda_available and self.detection_model == "cnn"

    def get_runtime_summary(self):
        """
        Human-readable runtime summary for logging.

        Returns:
            str: Runtime summary
        """
        if self.uses_gpu():
            return "Face detection: GPU using (dlib CUDA + CNN)"
        return f"Face detection: CPU using ({self.detection_model.upper()})"

    def _is_dlib_cuda_available(self):
        """
        Check whether the installed dlib build exposes CUDA support.

        Returns:
            bool: True when CUDA support is available
        """
        if dlib is None:
            return False

        try:
            return bool(getattr(dlib, "DLIB_USE_CUDA", False) and dlib.cuda.get_num_devices() > 0)
        except Exception:
            return False

    def _resolve_detection_model(self, requested_model):
        """
        Resolve the face detection model to use.

        Args:
            requested_model: Configured model name

        Returns:
            str: 'hog' or 'cnn'
        """
        if requested_model == "auto":
            return "cnn" if self.dlib_cuda_available else "hog"
        if requested_model == "cnn" and not self.dlib_cuda_available:
            logger.warning("CNN face detection requested, but dlib CUDA is unavailable. Falling back to HOG.")
            return "hog"
        return requested_model

    def _prepare_frame(self, frame):
        """
        Resize the frame for faster face detection if configured.

        Args:
            frame: Input frame (BGR)

        Returns:
            tuple[np.ndarray, float]: Process frame and scale factor back to source size
        """
        if self.process_scale >= 0.999:
            return frame, 1.0

        processed_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.process_scale,
            fy=self.process_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        return processed_frame, 1.0 / self.process_scale

    def load_known_faces(self):
        """
        Load and encode all known faces from directory structure
        """
        self.known_encodings = {}
        self.known_names = []

        if not self.known_faces_dir.exists():
            logger.warning(f"Known faces directory not found: {self.known_faces_dir}")
            return

        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            person_encodings = []

            for image_path in person_dir.glob("*.jpg"):
                try:
                    image = face_recognition.load_image_file(str(image_path))
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        person_encodings.extend(encodings)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")

            if person_encodings:
                avg_encoding = np.mean(person_encodings, axis=0)
                self.known_encodings[person_name] = avg_encoding
                self.known_names.append(person_name)
                logger.info(f"Loaded {len(person_encodings)} images for {person_name}")
            else:
                logger.warning(f"No valid face images found for {person_name}")

        if self.known_names:
            self.known_encoding_matrix = np.array(
                [self.known_encodings[name] for name in self.known_names],
                dtype=np.float32,
            )
        else:
            self.known_encoding_matrix = np.empty((0, 128), dtype=np.float32)

    def detect_and_identify(self, frame, model=None):
        """
        Detect and identify faces in frame

        Args:
            frame: Input frame (BGR)
            model: Detection model - 'hog' (fast, CPU) or 'cnn' (accurate, GPU)

        Returns:
            detections: List of dicts with keys: name, confidence, bbox
        """
        detections = []

        if not self.known_names:
            logger.warning("No known faces loaded for identification")
            return detections

        process_frame, scale_back = self._prepare_frame(frame)
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        detection_model = model or self.detection_model
        if not self._runtime_usage_logged:
            logger.info(self.get_runtime_summary())
            self._runtime_usage_logged = True
        face_locations = face_recognition.face_locations(
            rgb_frame,
            number_of_times_to_upsample=self.upsample_times,
            model=detection_model,
        )
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_encoding_matrix, face_encoding, tolerance=self.tolerance)
            name = "Unknown"
            confidence = 0.0

            face_distances = face_recognition.face_distance(self.known_encoding_matrix, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]

            top, right, bottom, left = face_location
            if scale_back != 1.0:
                top = int(top * scale_back)
                right = int(right * scale_back)
                bottom = int(bottom * scale_back)
                left = int(left * scale_back)
            detections.append(
                {
                    "name": name,
                    "confidence": float(confidence),
                    "bbox": (top, right, bottom, left),
                }
            )

        return detections

    def add_known_face(self, person_name, image_path):
        """
        Add a new known face to the system

        Args:
            person_name: Name of the person
            image_path: Path to the image file
        """
        try:
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)

            if encodings:
                if person_name not in self.known_encodings:
                    self.known_encodings[person_name] = encodings[0]
                    self.known_names.append(person_name)
                else:
                    self.known_encodings[person_name] = np.mean(
                        [self.known_encodings[person_name], encodings[0]], axis=0
                    )
                self.known_encoding_matrix = np.array(
                    [self.known_encodings[name] for name in self.known_names],
                    dtype=np.float32,
                )
                logger.info(f"Added face encoding for {person_name}")
            else:
                logger.warning(f"No faces detected in image {image_path}")
        except Exception as e:
            logger.error(f"Error adding known face: {e}")

    def __del__(self):
        pass
