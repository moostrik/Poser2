# Standard library imports
from threading import Lock

import numpy as np

from modules.pose.features.Similarity import Similarity
from modules.pose.batch.WindowSimilarity import SimilarityResult
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, replace
from modules.settings import BaseSettings, Field


class SimilarityApplicatorSettings(BaseSettings):
    """Configuration for SimilarityApplicator."""
    max_poses: Field[int] = Field(4, access=Field.INIT, min=1, max=16)


class SimilarityApplicator(FilterNode):
    """Filter that applies pre-computed similarity data to poses.

    Unlike SimilarityExtractor, this does not aggregate from pairwise batch.
    It simply applies dict[int, Similarity] where the Similarity objects
    are already computed per-pose by WindowSimilarity.

    Thread-safe: Uses lock to protect stored similarity dict.
    """

    def __init__(self, settings: SimilarityApplicatorSettings | None = None) -> None:
        self._settings = settings if settings is not None else SimilarityApplicatorSettings()
        max_poses = self._settings.max_poses
        self._similarity_dict: dict[int, Similarity] = {}
        self._lock: Lock = Lock()
        # Zero similarity with valid scores - used when a track is absent so that
        # downstream SimilarityStickyFiller receives real (zero) data instead of NaN,
        # which would cause it to hold the last stale value indefinitely.
        values = np.zeros(max_poses, dtype=np.float32)
        scores = np.ones(max_poses, dtype=np.float32)
        self._zero_similarity: Similarity = Similarity(values, scores)

    def set(self, similarity_dict: dict[int, Similarity]) -> None:
        """Store the per-pose similarity dict for processing.

        Args:
            similarity_dict: Maps track_id -> Similarity object
        """
        with self._lock:
            self._similarity_dict = similarity_dict

    def set_result(self, result: SimilarityResult) -> None:
        self.set(result.similarity)

    def process(self, pose: Frame) -> Frame:
        """Apply pre-computed similarity to this pose.

        Args:
            pose: Frame to enrich with similarity data

        Returns:
            Frame with updated similarity field
        """
        with self._lock:
            similarity: Similarity | None = self._similarity_dict.get(pose.track_id)

        # Always update - use zero Similarity if not found (resets stale data and
        # signals the downstream SimilarityStickyFiller with valid zeros instead of NaN)
        if similarity is None:
            similarity = self._zero_similarity
        return replace(pose, {Similarity: similarity})
