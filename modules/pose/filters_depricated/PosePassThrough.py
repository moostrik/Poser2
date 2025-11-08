# Pose imports
from modules.pose.filters_depricated.PoseBatchFilterBase import PoseBatchFilterBase
from modules.pose.Pose import PoseDict

class PosePassThrough(PoseBatchFilterBase):
    def add_poses(self, poses: PoseDict) -> None:
        self._notify_callbacks(poses)