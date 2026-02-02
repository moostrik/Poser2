# Standard library imports
from dataclasses import replace
from threading import Lock

from modules.pose.features.Similarity import Similarity, configure_similarity
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.Frame import Frame


class SimilarityApplicator(FilterNode):
    """Filter that applies pre-computed similarity data to poses.

    Unlike SimilarityExtractor, this does not aggregate from pairwise batch.
    It simply applies dict[int, Similarity] where the Similarity objects
    are already computed per-pose by WindowSimilarity.

    Thread-safe: Uses lock to protect stored similarity dict.
    """

    def __init__(self, max_poses: int) -> None:
        configure_similarity(max_poses)
        self._similarity_dict: dict[int, Similarity] = {}
        self._lock: Lock = Lock()

    def submit(self, similarity_dict: dict[int, Similarity]) -> None:
        """Store the per-pose similarity dict for processing.

        Args:
            similarity_dict: Maps track_id -> Similarity object
        """
        with self._lock:
            self._similarity_dict = similarity_dict

    def process(self, pose: Frame) -> Frame:
        """Apply pre-computed similarity to this pose.

        Args:
            pose: Frame to enrich with similarity data

        Returns:
            Frame with updated similarity field
        """
        with self._lock:
            similarity: Similarity | None = self._similarity_dict.get(pose.track_id)

        if similarity is not None:
            pose = replace(pose, similarity=similarity)

        return pose
