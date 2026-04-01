"""
Train behavior models from collected data using the shared preprocessing pipeline.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BEHAVIOR_DATA_DIR, MODELS_DIR
from core.behavior_model import BehaviorModel
from utils.data_validation import print_data_distribution, validate_and_filter_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def hard_reset_models(models_dir):
    """
    Delete old saved behavior artifacts so training is clean.
    """
    logger.info("=" * 70)
    logger.info("HARD RESET: Deleting old models...")
    logger.info("=" * 70)

    models_dir = Path(models_dir)
    deleted_count = 0
    for pattern in ("*_behavior.h5", "*_label_map.pkl", "*_preprocessor.pkl"):
        for model_file in models_dir.glob(pattern):
            try:
                model_file.unlink()
                logger.info(f"  Deleted {model_file.name}")
                deleted_count += 1
            except Exception as exc:
                logger.warning(f"  Failed to delete {model_file.name}: {exc}")

    if deleted_count > 0:
        logger.info(f"Hard reset complete: {deleted_count} files deleted\n")
    else:
        logger.info("No old models found (first training)\n")


def train_behavior_models():
    logger.info("=" * 70)
    logger.info("BEHAVIOR MODEL TRAINING - SHARED PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    behavior_data_dir = Path(BEHAVIOR_DATA_DIR)
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    hard_reset_models(models_dir)

    all_sequences = []
    all_labels = []
    per_person_sequences = {}
    per_person_labels = {}

    sequence_files = sorted(behavior_data_dir.glob("*_sequences.npy"))
    logger.info(f"Found {len(sequence_files)} sequence files")

    for seq_file in sequence_files:
        person_name = seq_file.stem.replace("_sequences", "")
        labels_file = behavior_data_dir / f"{person_name}_labels.npy"
        if not labels_file.exists():
            logger.warning(f"Labels file not found for {person_name}, skipping")
            continue

        sequences = np.load(str(seq_file))
        labels = np.load(str(labels_file), allow_pickle=True)
        valid_sequences, rejected = validate_and_filter_sequences(sequences, min_std=0.003)

        if len(valid_sequences) == 0:
            logger.warning(f"{person_name}: all sequences rejected during validation")
            continue

        valid_labels = labels[: len(valid_sequences)]
        per_person_sequences[person_name] = valid_sequences
        per_person_labels[person_name] = valid_labels
        all_sequences.append(valid_sequences)
        all_labels.extend(valid_labels.tolist())

        logger.info(
            f"{person_name:15} | original={len(sequences):3d} "
            f"| valid={len(valid_sequences):3d} | rejected={rejected:3d}"
        )

    if not all_sequences:
        logger.error("No valid training data found")
        return

    print_data_distribution(per_person_sequences, per_person_labels)

    X = np.vstack(all_sequences).astype(np.float32)
    y = np.array(all_labels)
    logger.info(f"Combined raw sequence shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    unique_train_classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_classes,
        y=y_train,
    )
    class_weights = {index: float(weight) for index, weight in enumerate(class_weights_array)}

    logger.info("Class weights:")
    for class_index, class_name in enumerate(unique_train_classes):
        logger.info(f"  {class_name:15} -> {class_weights[class_index]:.4f}")

    global_model = BehaviorModel()
    global_model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        class_weights=class_weights,
    )

    logger.info("\nGenerating test predictions with shared preprocessor...")
    y_pred = []
    y_conf = []
    for sample in X_test:
        pred_name, pred_conf = global_model.predict(sample)
        y_pred.append(pred_name)
        y_conf.append(pred_conf)

    y_pred = np.array(y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"\nGLOBAL MODEL TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    logger.info("\nPER-CLASS ACCURACY:")
    for person_name in sorted(np.unique(y)):
        mask = y_test == person_name
        if np.sum(mask) == 0:
            continue
        class_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        logger.info(f"  {person_name:15} | accuracy={class_accuracy:.4f} | samples={np.sum(mask)}")

    logger.info("\nCONFUSION MATRIX:")
    sorted_labels = sorted(np.unique(y))
    confusion = confusion_matrix(y_test, y_pred, labels=sorted_labels)
    logger.info("True \\ Pred " + "".join(f"{label:>14s}" for label in sorted_labels))
    for row_index, true_label in enumerate(sorted_labels):
        row_values = "".join(f"{value:14d}" for value in confusion[row_index])
        logger.info(f"{true_label:12s} {row_values}")

    logger.info("\nCLASSIFICATION REPORT:")
    logger.info(classification_report(y_test, y_pred, zero_division=0))

    diagnostics = global_model.get_sequence_diagnostics(X_train[0])
    logger.info(
        "Training feature stats | raw_mean=%.4f raw_std=%.4f feature_mean=%.4f feature_std=%.4f",
        diagnostics["training_raw_mean"] or 0.0,
        diagnostics["training_raw_std"] or 0.0,
        diagnostics["training_feature_mean"] or 0.0,
        diagnostics["training_feature_std"] or 0.0,
    )

    global_model.save("global", models_dir)
    logger.info(f"Saved global model to {models_dir}")
    logger.info(f"Label map used for training: {global_model.label_map}")


if __name__ == "__main__":
    train_behavior_models()
