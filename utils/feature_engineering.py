"""
Feature engineering helpers for behavior modeling.
"""
import numpy as np


def add_velocity(sequence):
    """
    Add first-order temporal velocity features aligned to frame n - frame n-1.

    Args:
        sequence: Array of shape (T, F)

    Returns:
        Array of shape (T, 2F) with [position, velocity] per frame
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {sequence.shape}")

    velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
    velocity[0] = 0.0
    return np.concatenate([sequence, velocity], axis=1).astype(np.float32)


def compute_motion_metrics(sequence):
    """
    Estimate whether a pose sequence contains meaningful movement.

    Args:
        sequence: Array of shape (T, 99) for positions or (T, 198) for
            [positions, velocity].

    Returns:
        Dict with motion_score, peak_motion, and temporal_std.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {sequence.shape}")
    if sequence.shape[0] == 0:
        return {
            "motion_score": 0.0,
            "peak_motion": 0.0,
            "temporal_std": 0.0,
        }

    if sequence.shape[1] == 198:
        position_sequence = sequence[:, :99]
    else:
        position_sequence = sequence

    if position_sequence.shape[1] % 3 != 0:
        raise ValueError(
            f"Expected feature dimension divisible by 3, got {position_sequence.shape[1]}"
        )

    joint_sequence = position_sequence.reshape(position_sequence.shape[0], -1, 3)
    velocity = np.diff(joint_sequence, axis=0, prepend=joint_sequence[:1])
    if velocity.shape[0] > 0:
        velocity[0] = 0.0

    joint_speed = np.linalg.norm(velocity, axis=2)
    speed_window = joint_speed[1:] if joint_speed.shape[0] > 1 else joint_speed
    motion_score = float(np.mean(speed_window)) if speed_window.size else 0.0
    peak_motion = float(np.max(speed_window)) if speed_window.size else 0.0
    temporal_std = float(np.mean(np.std(joint_sequence, axis=0)))

    return {
        "motion_score": motion_score,
        "peak_motion": peak_motion,
        "temporal_std": temporal_std,
    }
