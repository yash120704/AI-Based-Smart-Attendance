"""
Centroid Tracker for tracking objects across frames
"""
import logging
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CentroidTracker:
    """
    Tracks objects across frames using centroid distance matching
    """

    def __init__(self, max_disappeared=30, max_distance=80):
        """
        Initialize centroid tracker

        Args:
            max_disappeared: Frames before object is deregistered
            max_distance: Maximum pixel distance to consider same object
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        logger.info("CentroidTracker initialized")

    def register(self, centroid):
        """
        Register a new object
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Deregister an object
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, bboxes):
        """
        Update tracker with new bounding boxes

        Args:
            bboxes: List of (top, right, bottom, left) bounding boxes

        Returns:
            objects: Dict of {object_id: centroid}
        """
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(bboxes), 2))
        for i, bbox in enumerate(bboxes):
            top, right, bottom, left = bbox
            cX = (left + right) // 2
            cY = (top + bottom) // 2
            input_centroids[i] = [cX, cY]

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            existing_centroids = np.array([self.objects[object_id] for object_id in object_ids])

            D = np.zeros((len(existing_centroids), len(input_centroids)))
            for i in range(len(existing_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = np.linalg.norm(existing_centroids[i] - input_centroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects
