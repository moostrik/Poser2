# Standard library imports
import time

# Pose imports
from modules.pose.nodes.Nodes import NodeBase
from modules.pose.features import BBoxFeature
from modules.pose.Pose import Pose
from modules.tracker.Tracklet import Tracklet


class TrackletExtractor(NodeBase):
    """Extracts basic pose data from a tracklet.

    Creates a Pose with:
    - track_id: From tracklet ID
    - tracklet: The tracklet object itself
    - time_stamp: Current time or tracklet time
    - lost: Tracklet lost status
    - bbox: Bounding box feature from tracklet rect
    """

    def extract(self, tracklet: Tracklet) -> Pose:
        """Extract pose data from tracklet.

        Args:
            tracklet: Input tracklet with tracking information

        Returns:
            Pose with basic tracking data populated
        """
        # Extract bounding box from tracklet
        bbox = BBoxFeature.from_rect(tracklet.roi)

        return Pose(
            track_id=tracklet.id,
            cam_id=tracklet.cam_id,
            tracklet=tracklet,
            time_stamp=time.time(),
            lost=tracklet.is_lost,
            bbox=bbox
        )