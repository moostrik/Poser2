import time

from modules.pose.nodes.Nodes import GeneratorNode
from modules.pose.Pose import Pose
from modules.pose.features import BBoxFeature
from modules.tracker.Tracklet import Tracklet


class PoseFromTracklet(GeneratorNode[Tracklet]):
    """Generates basic Pose objects from tracklet and bounding box information.

    Creates minimal poses suitable for initialization or placeholder purposes.
    The generated pose contains:
    - Tracklet (for tracking/identification)
    - Bounding box (as BBoxFeature)
    - Timestamp
    - Empty feature data (to be populated by extractors/detectors)
    """

    def __init__(self) -> None:
        self._tracklet: Tracklet | None = None

    def set(self, input_data: Tracklet) -> None:
        """Set the tracklet for pose generation.

        Args:
            input_data: The tracklet to associate with this pose
        """
        self._tracklet = input_data

    def is_ready(self) -> bool:
        """Check if the generator is ready to produce a pose."""
        return self._tracklet is not None

    def generate(self, time_stamp: float | None = None) -> Pose:
        """Generate a basic Pose from stored tracklet.

        Args:
            time_stamp: Optional timestamp. If None, uses tracklet's timestamp

        Returns:
            A Pose with tracklet, bounding box, and timestamp populated.
            Feature data will be empty/default values.

        Raises:
            RuntimeError: If no tracklet has been set
        """
        if self._tracklet is None:
            raise RuntimeError("No tracklet set. Call set() before generate().")

        bounding_box: BBoxFeature = BBoxFeature.from_rect(self._tracklet.roi)
        if not time_stamp:
            time_stamp = self._tracklet.time_stamp

        return Pose(
            track_id=self._tracklet.id,
            cam_id=self._tracklet.cam_id,
            tracklet=self._tracklet,
            bbox=bounding_box,
            time_stamp=time_stamp,
            lost=self._tracklet.is_removed
        )

    def reset(self) -> None:
        """Reset internal state."""
        self._tracklet = None