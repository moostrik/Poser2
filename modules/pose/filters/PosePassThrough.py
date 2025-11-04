from modules.pose.Pose import PoseDict
from modules.pose.filters.PoseFilterBase import PoseFilterBase

class PosePassThrough(PoseFilterBase):
    def add_poses(self, poses: PoseDict) -> None:
        self._notify_callbacks(poses)