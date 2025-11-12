from modules.pose.nodes.generators.PoseFromTracklet import PoseFromTracklet
from modules.pose.Pose import PoseDict
from ..GeneratorTracker import GeneratorTracker

class PoseFromTrackletGenerator(GeneratorTracker):
    """Convenience tracker for generating poses from tracklets."""

    def __init__(self, num_tracks: int) -> None:
        super().__init__(
            num_tracks=num_tracks,
            generator_factory=lambda: PoseFromTracklet()
        )