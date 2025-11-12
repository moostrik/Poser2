import time

from modules.pose.nodes.Nodes import GeneratorNode
from modules.pose.Pose import Pose
from modules.pose.features import BBoxFeature
from modules.tracker.Tracklet import Tracklet
from modules.utils.PointsAndRects import Rect


class PoseGenerator(GeneratorNode):
    """Generates basic Pose objects from tracklet and bounding box information.

    Creates minimal poses suitable for initialization or placeholder purposes.
    The generated pose contains:
    - Tracklet (for tracking/identification)
    - Bounding box (as BBoxFeature)
    - Timestamp
    - Empty feature data (to be populated by extractors/detectors)
    """

    def generate(self, tracklet: Tracklet, time_stamp: float | None = None) -> Pose:
        """Generate a basic Pose from tracklet.

        Args:
            tracklet: The tracklet to associate with this pose
            time_stamp: Optional timestamp. If None, uses current time

        Returns:
            A Pose with tracklet, bounding box, and timestamp populated.
            Feature data will be empty/default values.
        """
        bounding_box: BBoxFeature = BBoxFeature.from_rect(tracklet.roi)
        timestamp: float = time_stamp if time_stamp is not None else time.time()

        return Pose(
            tracklet=tracklet,
            bbox=bounding_box,
            time_stamp=timestamp,
            lost=tracklet.is_removed
        )

    def reset(self) -> None:
        """No state to reset - generator is stateless."""
        pass