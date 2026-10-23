# Standard library imports
from threading import Lock

from ...features import Similarity
from ...analytics import SimilarityResult
from ..Nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class SimilarityApplicatorSettings(BaseSettings):
    """Configuration for SimilarityApplicator."""
    max_poses: Field[int] = Field(4, access=Field.INIT, min=1, max=16)


class SimilarityApplicator(FilterNode):
    """Filter that stamps the analytics' per-pose Similarity rows onto the poses.

    It applies dict[int, Similarity] as computed per pose by WindowSimilarity (and held across gaps by
    SimilarityStickyFiller on the result, where presence is known). A pose without a row — no result yet, or
    the analytics saw fewer than two poses — gets the NaN dummy: no data. This node only stamps.

    Thread-safe: Uses lock to protect stored similarity dict.
    """

    def __init__(self, settings: SimilarityApplicatorSettings | None = None) -> None:
        self._settings = settings if settings is not None else SimilarityApplicatorSettings()
        self._similarity_dict: dict[int, Similarity] = {}
        self._lock: Lock = Lock()

    def set(self, result: SimilarityResult) -> None:
        """Store the per-pose similarity from a SimilarityResult."""
        with self._lock:
            self._similarity_dict = result.similarity

    def process(self, pose: Frame) -> Frame:
        """Apply pre-computed similarity to this pose.

        Args:
            pose: Frame to enrich with similarity data

        Returns:
            Frame with updated similarity field
        """
        with self._lock:
            similarity: Similarity | None = self._similarity_dict.get(pose.track_id)

        if similarity is None:
            similarity = Similarity.create_dummy()
        return replace(pose, {Similarity: similarity})
