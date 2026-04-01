"""
Sequence buffer utilities for behavior modeling.
"""
import logging
import time
from collections import deque

import numpy as np

from config import BEHAVIOR_FEATURE_DIM, SEQUENCE_DURATION_SECONDS, SEQUENCE_LENGTH

logger = logging.getLogger(__name__)


class SequenceBuffer:
    """
    Sliding window buffer to collect frame sequences.
    """

    def __init__(self, maxlen=SEQUENCE_LENGTH, feature_dim=BEHAVIOR_FEATURE_DIM):
        self.deque = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.feature_dim = feature_dim

    def add(self, feature_vector):
        feature_vector = np.asarray(feature_vector, dtype=np.float32)
        if feature_vector.shape[0] != self.feature_dim:
            logger.warning(f"Feature vector shape mismatch: {feature_vector.shape}")
            return
        self.deque.append(feature_vector)

    def is_ready(self):
        return len(self.deque) == self.maxlen

    def get_sequence(self):
        return np.array(list(self.deque), dtype=np.float32)

    def get_flat(self):
        return self.get_sequence().flatten().astype(np.float32)

    def reset(self):
        self.deque.clear()

    def get_length(self):
        return len(self.deque)

    def __len__(self):
        return len(self.deque)


class TimedSequenceBuffer:
    """
    Timestamp-aware buffer that resamples to a fixed number of frames over a fixed duration.
    """

    def __init__(
        self,
        target_frames=SEQUENCE_LENGTH,
        feature_dim=BEHAVIOR_FEATURE_DIM,
        duration_seconds=SEQUENCE_DURATION_SECONDS,
        history_seconds=None,
    ):
        self.target_frames = target_frames
        self.feature_dim = feature_dim
        self.duration_seconds = float(duration_seconds)
        self.history_seconds = float(history_seconds or (duration_seconds * 2.5))
        self.samples = deque()
        self.last_emit_timestamp = None
        self.start_timestamp = None

    def add(self, feature_vector, timestamp=None):
        feature_vector = np.asarray(feature_vector, dtype=np.float32)
        if feature_vector.shape[0] != self.feature_dim:
            logger.warning(f"Feature vector shape mismatch: {feature_vector.shape}")
            return

        absolute_timestamp = float(time.perf_counter() if timestamp is None else timestamp)
        if self.start_timestamp is None:
            self.start_timestamp = absolute_timestamp

        relative_timestamp = absolute_timestamp - self.start_timestamp
        self.samples.append((relative_timestamp, feature_vector))
        self._trim(relative_timestamp)

    def _trim(self, current_timestamp):
        cutoff = current_timestamp - self.history_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def reset(self):
        self.samples.clear()
        self.last_emit_timestamp = None
        self.start_timestamp = None

    def is_ready(self):
        return len(self.samples) >= 2 and (self.samples[-1][0] - self.samples[0][0]) >= self.duration_seconds

    def _interpolate_at(self, target_timestamp):
        sample_list = list(self.samples)

        if target_timestamp <= sample_list[0][0]:
            return sample_list[0][1]
        if target_timestamp >= sample_list[-1][0]:
            return sample_list[-1][1]

        for index in range(1, len(sample_list)):
            left_timestamp, left_value = sample_list[index - 1]
            right_timestamp, right_value = sample_list[index]
            if left_timestamp <= target_timestamp <= right_timestamp:
                delta = max(right_timestamp - left_timestamp, 1e-6)
                alpha = (target_timestamp - left_timestamp) / delta
                return ((1.0 - alpha) * left_value + alpha * right_value).astype(np.float32)

        return sample_list[-1][1]

    def get_sequence(self, end_timestamp=None):
        if not self.is_ready():
            return None

        end_timestamp = self.samples[-1][0] if end_timestamp is None else float(end_timestamp)
        start_timestamp = end_timestamp - self.duration_seconds
        target_timestamps = np.linspace(start_timestamp, end_timestamp, self.target_frames, dtype=np.float64)
        sequence = np.stack([self._interpolate_at(timestamp) for timestamp in target_timestamps], axis=0)
        return sequence.astype(np.float32)

    def emit_if_ready(self, stride_seconds):
        if not self.is_ready():
            return None

        latest_timestamp = self.samples[-1][0]
        if self.last_emit_timestamp is not None and (latest_timestamp - self.last_emit_timestamp) < stride_seconds:
            return None

        self.last_emit_timestamp = latest_timestamp
        return self.get_sequence(end_timestamp=latest_timestamp)
