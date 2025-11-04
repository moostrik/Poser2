# Pose imports
from .PoseFilterBase import PoseFilterBase
from ..Pose import PoseDict

class PosePassThrough(PoseFilterBase):
    def add_poses(self, poses: PoseDict) -> None:
        self._notify_callbacks(poses)