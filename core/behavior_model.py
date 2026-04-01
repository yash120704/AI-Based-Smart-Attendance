"""
Behavior model with a shared sequence preprocessor for training and inference.
"""
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

from config import BEHAVIOR_CONFIDENCE_THRESHOLD, BEHAVIOR_FEATURE_DIM, MODELS_DIR, SEQUENCE_LENGTH
from utils.feature_engineering import add_velocity

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


def configure_tensorflow_runtime():
    """
    Enable safe GPU memory behavior when TensorFlow sees a GPU.
    """
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        for gpu_device in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu_device, True)
        return gpu_devices
    except Exception as exc:
        logger.warning(f"TensorFlow runtime configuration failed: {exc}")
        return []


class SequencePreprocessor:
    """
    Shared preprocessing for training and inference.
    """

    def __init__(self, sequence_length=SEQUENCE_LENGTH, feature_dim=BEHAVIOR_FEATURE_DIM):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.feature_mean = None
        self.feature_std = None
        self.raw_mean = None
        self.raw_std = None

    def _prepare_position_sequences(self, sequences):
        sequences = np.asarray(sequences, dtype=np.float32)

        if sequences.ndim == 3 and sequences.shape[1:] == (self.sequence_length, self.feature_dim):
            return sequences
        if sequences.ndim == 2 and sequences.shape == (self.sequence_length, self.feature_dim):
            return sequences[np.newaxis, ...]
        if sequences.ndim == 2 and sequences.shape[1] == self.sequence_length * self.feature_dim:
            return sequences.reshape((-1, self.sequence_length, self.feature_dim)).astype(np.float32)
        if sequences.ndim == 1 and sequences.shape[0] == self.sequence_length * self.feature_dim:
            return sequences.reshape((1, self.sequence_length, self.feature_dim)).astype(np.float32)

        raise ValueError(f"Unsupported raw sequence shape: {sequences.shape}")

    def _prepare_feature_sequences(self, sequences):
        sequences = np.asarray(sequences, dtype=np.float32)
        expected_feature_dim = self.feature_dim * 2

        if sequences.ndim == 3 and sequences.shape[1:] == (self.sequence_length, expected_feature_dim):
            return sequences
        if sequences.ndim == 2 and sequences.shape == (self.sequence_length, expected_feature_dim):
            return sequences[np.newaxis, ...]
        if sequences.ndim == 2 and sequences.shape[1] == self.sequence_length * expected_feature_dim:
            return sequences.reshape((-1, self.sequence_length, expected_feature_dim)).astype(np.float32)
        if sequences.ndim == 1 and sequences.shape[0] == self.sequence_length * expected_feature_dim:
            return sequences.reshape((1, self.sequence_length, expected_feature_dim)).astype(np.float32)

        raise ValueError(f"Unsupported processed sequence shape: {sequences.shape}")

    def fit(self, sequences):
        position_sequences = self._prepare_position_sequences(sequences)
        feature_sequences = np.array([add_velocity(sequence) for sequence in position_sequences], dtype=np.float32)

        self.raw_mean = position_sequences.mean(axis=(0, 1)).astype(np.float32)
        self.raw_std = position_sequences.std(axis=(0, 1)).astype(np.float32)
        self.feature_mean = feature_sequences.mean(axis=(0, 1)).astype(np.float32)
        self.feature_std = np.maximum(feature_sequences.std(axis=(0, 1)), 1e-6).astype(np.float32)

    def transform_positions(self, sequences):
        position_sequences = self._prepare_position_sequences(sequences)
        feature_sequences = np.array([add_velocity(sequence) for sequence in position_sequences], dtype=np.float32)

        if self.feature_mean is not None and self.feature_std is not None:
            feature_sequences = (feature_sequences - self.feature_mean) / self.feature_std

        return feature_sequences.astype(np.float32)

    def transform_features(self, sequences):
        feature_sequences = self._prepare_feature_sequences(sequences)
        if self.feature_mean is not None and self.feature_std is not None:
            feature_sequences = (feature_sequences - self.feature_mean) / self.feature_std
        return feature_sequences.astype(np.float32)

    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform_positions(sequences)

    def to_dict(self):
        return {
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "raw_mean": self.raw_mean,
            "raw_std": self.raw_std,
        }

    def load_dict(self, state):
        self.sequence_length = int(state.get("sequence_length", self.sequence_length))
        self.feature_dim = int(state.get("feature_dim", self.feature_dim))
        self.feature_mean = state.get("feature_mean")
        self.feature_std = state.get("feature_std")
        self.raw_mean = state.get("raw_mean")
        self.raw_std = state.get("raw_std")

    def build_diagnostics(self, raw_sequence):
        raw_sequence = self._prepare_position_sequences(raw_sequence)[0]
        processed_sequence = self.transform_positions(raw_sequence)[0]

        diagnostics = {
            "live_raw_mean": float(raw_sequence.mean()),
            "live_raw_std": float(raw_sequence.std()),
            "live_feature_mean": float(processed_sequence.mean()),
            "live_feature_std": float(processed_sequence.std()),
            "training_raw_mean": float(np.mean(self.raw_mean)) if self.raw_mean is not None else None,
            "training_raw_std": float(np.mean(self.raw_std)) if self.raw_std is not None else None,
            "training_feature_mean": float(np.mean(self.feature_mean)) if self.feature_mean is not None else None,
            "training_feature_std": float(np.mean(self.feature_std)) if self.feature_std is not None else None,
        }

        if self.feature_mean is not None:
            diagnostics["feature_mean_gap"] = float(abs(diagnostics["live_feature_mean"] - diagnostics["training_feature_mean"]))
        else:
            diagnostics["feature_mean_gap"] = None

        return diagnostics


class BehaviorModel:
    """
    LSTM-based behavior model with shared preprocessing.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(MODELS_DIR) / "global_behavior.h5"

        self.model_path = str(model_path)
        self.model = None
        self.label_map = {}
        self.reverse_map = {}
        self.preprocessor = SequencePreprocessor()
        self.is_trained = False
        self.gpu_devices = configure_tensorflow_runtime()
        self._runtime_usage_logged = False

        if os.path.exists(self.model_path):
            try:
                self._load_model()
                self.is_trained = True
            except Exception as exc:
                logger.warning(f"Could not load model from {self.model_path}: {exc}")

        if self.gpu_devices:
            logger.info(f"BehaviorModel (LSTM) initialized with TensorFlow GPU support: {len(self.gpu_devices)} device(s)")
        else:
            logger.info("BehaviorModel (LSTM) initialized with TensorFlow CPU execution")

    def uses_gpu(self):
        return bool(self.gpu_devices)

    def get_runtime_summary(self):
        if self.uses_gpu():
            device_names = ", ".join(device.name for device in self.gpu_devices)
            return f"Behavior model: GPU using (TensorFlow on {device_names})"
        return "Behavior model: CPU using"

    def _log_runtime_usage_once(self):
        if not self._runtime_usage_logged:
            logger.info(self.get_runtime_summary())
            self._runtime_usage_logged = True

    def build_model(self, num_classes):
        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, BEHAVIOR_FEATURE_DIM * 2)),
                Dropout(0.4),
                LSTM(64, return_sequences=False),
                Dropout(0.4),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(self, X, y, X_val=None, y_val=None, class_weights=None):
        if len(X) == 0 or len(y) == 0:
            logger.warning("Empty training data provided")
            return

        position_sequences = self.preprocessor._prepare_position_sequences(X)
        X_train_processed = self.preprocessor.fit_transform(position_sequences)
        logger.info(f"Training input shape after preprocessing: {X_train_processed.shape}")

        unique_labels = sorted(list(set(y)))
        if len(unique_labels) < 2:
            logger.error(f"Cannot train with only {len(unique_labels)} class(es). Need at least 2 classes.")
            return

        self.label_map = {name: idx for idx, name in enumerate(unique_labels)}
        self.reverse_map = {idx: name for name, idx in self.label_map.items()}
        logger.info(f"Behavior label map: {self.label_map}")

        y_encoded = np.array([self.label_map[label] for label in y], dtype=np.int32)
        y_categorical = to_categorical(y_encoded, num_classes=len(unique_labels))

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_processed = self.preprocessor.transform_positions(X_val)
            y_val_encoded = np.array([self.label_map[label] for label in y_val], dtype=np.int32)
            y_val_categorical = to_categorical(y_val_encoded, num_classes=len(unique_labels))
            validation_data = (X_val_processed, y_val_categorical)

        self.model = self.build_model(len(unique_labels))

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        logger.info("Starting behavior model training...")
        self.model.fit(
            X_train_processed,
            y_categorical,
            epochs=50,
            batch_size=16,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else None,
            class_weight=class_weights if class_weights else None,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        self.is_trained = True
        self.save("global", MODELS_DIR)
        logger.info(f"BehaviorModel trained on {len(X_train_processed)} samples, {len(unique_labels)} classes")

    def preprocess_live_sequence(self, sequence):
        """
        Shared preprocessing entry point for inference from raw 30x99 positions.
        """
        processed = self.preprocessor.transform_positions(sequence)
        return processed[0]

    def get_sequence_diagnostics(self, sequence):
        return self.preprocessor.build_diagnostics(sequence)

    def _prepare_inference_sequence(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)

        try:
            processed = self.preprocessor.transform_positions(sequence)
            return processed
        except ValueError:
            processed = self.preprocessor.transform_features(sequence)
            return processed

    def predict(self, sequence):
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return "Unknown", 0.0

        try:
            processed = self._prepare_inference_sequence(sequence)
            self._log_runtime_usage_once()
            predictions = self.model.predict(processed, verbose=0)[0]
            class_idx = int(np.argmax(predictions))
            confidence = float(predictions[class_idx])
            predicted_name = self.reverse_map.get(class_idx, "Unknown")

            if confidence < BEHAVIOR_CONFIDENCE_THRESHOLD:
                return "Unknown", confidence

            return str(predicted_name), confidence
        except Exception as exc:
            logger.error(f"Error during prediction: {exc}")
            return "Unknown", 0.0

    def predict_all_confidences(self, sequence):
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return {}

        try:
            processed = self._prepare_inference_sequence(sequence)
            self._log_runtime_usage_once()
            predictions = self.model.predict(processed, verbose=0)[0]
            return {
                str(person_name): float(predictions[class_idx])
                for class_idx, person_name in self.reverse_map.items()
            }
        except Exception as exc:
            logger.error(f"Error during prediction: {exc}")
            return {}

    def save(self, person_name, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{person_name}_behavior.h5"
        label_map_path = models_dir / f"{person_name}_label_map.pkl"
        preprocessor_path = models_dir / f"{person_name}_preprocessor.pkl"

        if self.model is not None:
            self.model.save(str(model_path))
            logger.info(f"LSTM model saved to {model_path}")

        joblib.dump(self.label_map, str(label_map_path))
        joblib.dump(self.preprocessor.to_dict(), str(preprocessor_path))
        logger.info(f"Label map saved to {label_map_path}")
        logger.info(f"Preprocessor stats saved to {preprocessor_path}")

    def load(self, person_name, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        model_path = models_dir / f"{person_name}_behavior.h5"
        label_map_path = models_dir / f"{person_name}_label_map.pkl"
        preprocessor_path = models_dir / f"{person_name}_preprocessor.pkl"

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False

        try:
            self._load_model_from_path(str(model_path), str(label_map_path), str(preprocessor_path))
            self.is_trained = True
            logger.info(f"BehaviorModel loaded from {model_path}")
            logger.info(f"Loaded label map: {self.label_map}")
            return True
        except Exception as exc:
            logger.error(f"Error loading model: {exc}")
            return False

    def _load_model(self):
        label_map_path = self.model_path.replace(".h5", "_label_map.pkl")
        preprocessor_path = self.model_path.replace(".h5", "_preprocessor.pkl")
        self._load_model_from_path(self.model_path, label_map_path, preprocessor_path)

    def _load_model_from_path(self, model_path, label_map_path, preprocessor_path):
        self.model = load_model(model_path)
        if os.path.exists(label_map_path):
            self.label_map = joblib.load(label_map_path)
            self.reverse_map = {idx: name for name, idx in self.label_map.items()}
        if os.path.exists(preprocessor_path):
            self.preprocessor.load_dict(joblib.load(preprocessor_path))
        else:
            logger.warning(f"Preprocessor stats not found: {preprocessor_path}")
