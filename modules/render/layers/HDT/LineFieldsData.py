
from dataclasses import dataclass
from enum import Enum

from modules.utils.OneEuroInterpolation import AngleEuroInterpolator, OneEuroSettings
from modules.pose.smooth.PoseSmoothDataManager import PoseJoint, PoseSmoothDataManager, SymmetricJointType


class param(Enum):
    elbow_L =       0
    elbow_L_Vel =   1
    shldr_L =       2
    shldr_L_Vel =   3
    elbow_R =       4
    elbow_R_Vel =   5
    shldr_R =       6
    shldr_R_Vel =   7
    head =          8
    head_vel =      9
    symmetry =      10

class LineFieldsData:

    def __init__(self, smooth_data: PoseSmoothDataManager, cam_id: int) -> None:
        self.smooth_data = smooth_data

        values: dict[param, float] = {}
        filters: dict[param, float] = {}

        for P in param:
            values[P] = 0.0
            filters[P] = AngleEuroInterpolator(OneEuroSettings(60, 1, 0))

    def update(self):
        elbow_L: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.left_elbow)
        elbow_L_Vel: float  = self.smooth_data.get_velocity(self.cam_id, PoseJoint.left_elbow)
        shldr_L: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.left_shoulder)
        shldr_L_Vel: float  = self.smooth_data.get_velocity(self.cam_id, PoseJoint.left_shoulder)
        elbow_R: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.right_elbow)
        elbow_R_Vel: float  = self.smooth_data.get_velocity(self.cam_id, PoseJoint.right_elbow)
        shldr_R: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.right_shoulder)
        shldr_R_Vel: float  = self.smooth_data.get_velocity(self.cam_id, PoseJoint.right_shoulder)
        head: float     = self.smooth_data.get_head(self.cam_id)
        motion: float   = self.smooth_data.get_cumulative_motion(self.cam_id)
        age: float      = self.smooth_data.get_age(self.cam_id)
        anchor: float   = 1.0 - self.smooth_data.rect_settings.centre_dest_y
        synchrony: float= self.smooth_data.get_mean_symmetry(self.cam_id)
