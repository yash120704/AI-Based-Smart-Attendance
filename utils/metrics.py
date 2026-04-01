"""
Utility metrics for behavior analysis
"""
import numpy as np
from scipy.spatial.distance import cosine, euclidean


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors

    Args:
        a: First vector
        b: Second vector

    Returns:
        float: Cosine similarity (0 to 1, where 1 is identical)
    """
    a = np.array(a).flatten()
    b = np.array(b).flatten()

    if len(a) == 0 or len(b) == 0:
        return 0.0

    try:
        similarity = 1 - cosine(a, b)
        return float(np.clip(similarity, 0, 1))
    except Exception:
        return 0.0


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors

    Args:
        a: First vector
        b: Second vector

    Returns:
        float: Euclidean distance
    """
    a = np.array(a).flatten()
    b = np.array(b).flatten()

    if len(a) == 0 or len(b) == 0:
        return float("inf")

    return float(euclidean(a, b))


def normalize_sequence(seq):
    """
    Normalize sequence using min-max scaling

    Args:
        seq: Sequence of shape (n_frames, n_features)

    Returns:
        normalized_seq: Normalized sequence
    """
    seq = np.array(seq)

    if seq.ndim != 2:
        return seq

    min_vals = np.min(seq, axis=0, keepdims=True)
    max_vals = np.max(seq, axis=0, keepdims=True)

    denom = max_vals - min_vals
    denom[denom == 0] = 1

    normalized = (seq - min_vals) / denom

    return normalized


def compute_velocity(seq):
    """
    Compute velocity from pose sequence

    Args:
        seq: Sequence of shape (n_frames, n_features)

    Returns:
        velocity: Velocity magnitude for each frame
    """
    seq = np.array(seq)

    if len(seq) < 2:
        return np.zeros(1)

    diff = np.diff(seq, axis=0)
    velocity = np.linalg.norm(diff, axis=1)

    return velocity


def compute_acceleration(seq):
    """
    Compute acceleration from pose sequence

    Args:
        seq: Sequence of shape (n_frames, n_features)

    Returns:
        acceleration: Acceleration magnitude for each frame
    """
    velocity = compute_velocity(seq)

    if len(velocity) < 2:
        return np.zeros(1)

    acceleration = np.diff(velocity)

    return acceleration


def compute_jitter(seq):
    """
    Compute jitter (smoothness) from pose sequence

    Args:
        seq: Sequence of shape (n_frames, n_features)

    Returns:
        jitter: Average jitter value
    """
    accel = compute_acceleration(seq)

    if len(accel) == 0:
        return 0.0

    jitter = np.mean(np.abs(accel))

    return float(jitter)
