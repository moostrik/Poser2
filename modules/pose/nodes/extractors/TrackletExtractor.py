# Standard library imports
import time

# Pose imports
from ..Nodes import NodeBase
from ...features import BBox
from ...frame import Frame
from modules.tracker import Tracklet


class TrackletExtractor(NodeBase):
    """Extracts basic pose data from a tracklet.

    Creates a Pose with:
    - track_id: From tracklet ID
    - tracklet: The tracklet object itself
    - time_stamp: Current time or tracklet time
    - lost: Tracklet lost status
    - bbox: Bounding box feature from tracklet rect
    """

    def extract(self, tracklet: Tracklet) -> Frame:
        """Extract pose data from tracklet.

        Args:
            tracklet: Input tracklet with tracking information

        Returns:
            Pose with basic tracking data populated
        """
        # Extract bounding box from tracklet
        bbox = BBox.from_rect(tracklet.roi)

        return Frame(
            track_id=tracklet.id,
            cam_id=tracklet.cam_id,
            time_stamp=time.time(),
            features={BBox: bbox},
        )