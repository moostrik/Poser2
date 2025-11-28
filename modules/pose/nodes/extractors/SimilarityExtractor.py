# Standard library imports
from dataclasses import replace
from threading import Lock

import numpy as np

from modules.pose.features.Similarity import Similarity, configure_similarity, AggregationMethod
from modules.pose.similarity import SimilarityBatch
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame
from modules.pose.similarity.features.SimilarityFeature import SimilarityFeature

from modules.utils.HotReloadMethods import HotReloadMethods


class SimilarityExtractorConfig(NodeConfigBase):
    """Configuration for similarity processor with automatic change notification."""

    def __init__(self, max_poses: int, method: AggregationMethod = AggregationMethod.HARMONIC_MEAN, exponent: float = 2.0) -> None:
        super().__init__()
        self.max_poses: int = max_poses
        self.method: AggregationMethod = method
        self.exponent: float = exponent

class SimilarityExtractor(FilterNode):
    """Filter that enriches poses with similarity data using external batch context."""

    def __init__(self, config: SimilarityExtractorConfig) -> None:
        configure_similarity(config.max_poses)
        self._config: SimilarityExtractorConfig = config
        self._batch: SimilarityBatch | None = None
        self._lock: Lock = Lock()

        HotReload = HotReloadMethods(self.__class__, True, True)

    def submit(self, input_data: SimilarityBatch | None) -> None:
        """Store the similarity batch for processing (optional)."""
        with self._lock:
            self._batch = input_data

    def process(self, pose: Frame) -> Frame:
        """Enrich pose with similarity data."""
        with self._lock:
            batch: SimilarityBatch | None = self._batch

        if batch is not None:
            similarity: Similarity = SimilarityExtractor._extract_pose_similarities(pose.track_id, batch, self._config)
            pose = replace(pose, similarity=similarity)

        return pose

    @staticmethod
    def _extract_pose_similarities(pose_id: int, batch: SimilarityBatch, config: SimilarityExtractorConfig) -> Similarity:
        """Extract similarity array for a specific pose from the batch."""
        values: np.ndarray = np.full(config.max_poses, np.nan, dtype=np.float32)
        scores: np.ndarray = np.zeros(config.max_poses, dtype=np.float32)



        # For each possible other pose
        for other_idx in range(config.max_poses):

            # Skip self-comparison (leave as NaN with score 0)
            if pose_id == other_idx:
                continue

            # Find the pairwise comparison (if it exists)
            pair_id: tuple[int, int] = (min(pose_id, other_idx), max(pose_id, other_idx))

            similarity_feature: SimilarityFeature | None = batch.get_pair(pair_id)

            if similarity_feature is not None:
                # Aggregate the per-landmark similarities into overall similarity
                overall_sim: float = similarity_feature.aggregate(config.method, 0.0, config.exponent)

                if not np.isnan(overall_sim):
                    values[other_idx] = overall_sim
                    # Use mean confidence across valid landmarks as the score
                    valid_scores: np.ndarray = similarity_feature.scores[similarity_feature.valid_mask]
                    if len(valid_scores) > 0:
                        scores[other_idx] = np.mean(valid_scores)

        return Similarity(values, scores)
