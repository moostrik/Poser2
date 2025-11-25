from modules.pose.nodes.generators.PoseFromTracklet import PoseFromTracklet
from ..GeneratorTracker import GeneratorTracker

from modules.tracker.Tracklet import Tracklet

class PoseFromTrackletGenerator(GeneratorTracker):
    """Convenience tracker for generating poses from tracklets."""

    def __init__(self, num_tracks: int) -> None:
        super().__init__(
            num_tracks=num_tracks,
            generator_factory=lambda: PoseFromTracklet()
        )

    def submit_tracklets(self, tracklet_dict: dict[int, Tracklet]) -> None:
        """Set tracklets for pose generation."""
        self.submit(tracklet_dict)