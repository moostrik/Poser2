# Pose imports
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.pose.Pose import PoseDict

class PosePassThrough(PoseFilterBase):
    def add_poses(self, poses: PoseDict) -> None:
        self._notify_callbacks(poses)