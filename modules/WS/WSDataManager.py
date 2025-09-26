from dataclasses import dataclass, field, fields, make_dataclass
# from modules.utils.SmoothValue import SmoothValue

import numpy as np
from time import time

from modules.utils.SmoothOneEuro import SmoothOneEuro, SmoothOneEuroCircular
from modules.tracker.Tracklet import Tracklet
from modules.pose.PoseDefinitions import PoseAngleData, Pose, PoseJoint, PosePointData, Rect, PoseAngleJoints, PoseAngleJointNames, PoseMeasurementData
from modules.pose.PoseStream import PoseStreamData

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class WSDataSettings:
    smoothness: float = 0.5
    responsiveness: float = 0.5
    _is_updated: bool = field(default=False, repr=False)

    def __setattr__(self, name, value) -> None:
        if name != "_is_updated" and hasattr(self, name) and getattr(self, name) != value:
            super().__setattr__("_is_updated", True)
        super().__setattr__(name, value)

    @property
    def is_updated(self) -> bool:
        val: bool = self._is_updated
        self._is_updated = False
        return val

    @is_updated.setter
    def is_updated(self, value: bool) -> None:
        self._is_updated = value

class WSData:
    def __init__(self, frequency: float) -> None:

        self.filters: dict[str, SmoothOneEuro | SmoothOneEuroCircular] = {}

        self.filters["world_angle"] = SmoothOneEuroCircular(frequency)
        self.filters["approximate_person_length"] = SmoothOneEuro(frequency)
        for key in PoseAngleJoints:
            angle_name: str = key.name
            self.filters[angle_name] = SmoothOneEuroCircular(frequency)

        self.present: bool = False
        self.start_age: float = 0.0
        self.age: float = 0.0
        self.approximate_person_length: float = 1.0

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def __getattr__(self, name) -> float:
        """Allow access to smoothed values as attributes."""
        if "filters" in self.__dict__ and name in self.filters:
            smoother: SmoothOneEuro | SmoothOneEuroCircular = self.filters[name]
            value: float | None = smoother.get_smoothed_value()
            if value is None:
                return 0.0
            return value
        raise AttributeError(f"'SmoothMetrics' object has no attribute '{name}'")

    def __setattr__(self, name, value) -> None:
        # Avoid recursion for internal attributes
        if "filters" in self.__dict__ and name in self.filters:
            filter: SmoothOneEuro | SmoothOneEuroCircular = self.filters[name]
            filter.add_value(value)
        else:
            super().__setattr__(name, value)

    def update(self) -> None:
        for filter in self.filters.values():
            filter.update()

        self.age = time() - self.start_age if self.start_age is not None else 0.0

    def add_pose(self, pose: Pose) -> None:
        tracklet: Tracklet | None = pose.tracklet

        if tracklet is not None and tracklet.is_active and tracklet.age_in_seconds > 2.0:
            if self.start_age == 0.0:
                self.start_age = time()
            self.present = True


            angle_data: PoseAngleData | None  = pose.angle_data
            if angle_data is not None:
                # print(angles)
                for i, name in enumerate(PoseAngleJointNames):
                    angle: float = angle_data.angles[i]
                    score: float = angle_data.scores[i]
                    if score > 0.0 and angle is not np.nan:
                        setattr(self, name, angle)

            pose_measurement_data: PoseMeasurementData | None = pose.measurement_data
            if pose_measurement_data is not None:
                self.approximate_person_length = pose_measurement_data.approximate_length #* min(self.age, 1.0)

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            nose_angle_offset: float | None = None
            pose_points: PosePointData | None = pose.point_data
            crop_rect: Rect | None = pose.crop_rect
            if pose_points is not None and crop_rect is not None:
                nose_x = pose_points.points[PoseJoint.nose.value][0]
                nose_conf = pose_points.scores[PoseJoint.nose.value]
                if nose_conf > 0.3:
                    # calculate nose offset from center of crop rect
                    crop_center_x: float = crop_rect.center.x
                    nose_offset_x: float = nose_x - 0.5
                    # convert to angle in degrees, assuming 110 degree horizontal FOV
                    fov_degrees: float = 110.0
                    nose_angle_offset = nose_offset_x * crop_rect.width * fov_degrees
                    # print(f"nose_offset_x: {nose_offset_x}, nose_angle_offset: {crop_rect.width}")


            world_angle += nose_angle_offset if nose_angle_offset is not None else 0.0

            # convert from [0...360] to [-pi, pi]
            self.world_angle: float = np.deg2rad(world_angle - 180)
            # print (self.world_angle)

        if pose.tracklet.is_removed:
            self.reset()

    def add_stream(self, stream: PoseStreamData) -> None:
        if not self.present:
            return

        # angles: list[float] = stream.get_last_angles()
        # for i, angle in enumerate(angles):
        #     name: str = PoseAngleNames[i]
        #     if angle is not np.nan:
        #         setattr(self, name, angle)

    def reset(self) -> None:
        """Reset all smoothers"""
        for smoother in self.filters.values():
            smoother.reset()
        self.present = False
        self.age = 0.0
        self.start_age = 0.0

    def set_smoothness(self, value: float) -> None:
        for smoother in self.filters.values():
            smoother.set_smoothness(value)

    def set_responsiveness(self, value: float) -> None:
        for smoother in self.filters.values():
            smoother.set_responsiveness(value)



class WSDataManager:
    def __init__(self, frequency: float, num_players: int, settings: WSDataSettings) -> None:
        self.settings: WSDataSettings = settings
        self.num_players: int = num_players

        self.smooth_metrics_dict: dict[int, WSData] = {}
        for i in range(self.num_players):
            self.smooth_metrics_dict[i] = WSData(frequency)

        self.num_active_players_smoother: SmoothOneEuro = SmoothOneEuro(frequency)
        self.presents: dict[int, bool] = {}

        self.world_positions: dict[int, float] = {}
        self.pose_lengths: dict[int, float] = {}
        self.ages: dict[int, float] = {}
        self.left_shoulders: dict[int, float] = {}
        self.right_shoulders: dict[int, float] = {}
        self.left_elbows: dict[int, float] = {}
        self.right_elbows: dict[int, float] = {}

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    @ property
    def num_active_players(self) -> int:
        count: int = 0
        for i in range(self.num_players):
            count += self.is_player_present(i)
        return count

    @property
    def smooth_num_active_players(self) -> float:
        smoothed_value: float | None = self.num_active_players_smoother.get_smoothed_value()
        if smoothed_value is None:
            return 0.0
        return smoothed_value

    def add_poses(self, poses: list[Pose]) -> None:
        for pose in poses:
            self.smooth_metrics_dict[pose.tracklet.id].add_pose(pose)

    def add_streams(self, streams: list) -> None:
        for stream in streams:
            self.smooth_metrics_dict[stream.id].add_stream(stream)

            # print(stream.id, stream.get_last_angles())

    def update(self) -> None:
        if self.settings.is_updated:
            for sm in self.smooth_metrics_dict.values():
                sm.set_smoothness(self.settings.smoothness)
                sm.set_responsiveness(self.settings.responsiveness)

        for key, sm in self.smooth_metrics_dict.items():
            sm.update()
            self.presents[key] = sm.present
            self.world_positions[key] = sm.world_angle
            self.pose_lengths[key] = sm.approximate_person_length
            self.ages[key] = sm.age
            self.left_shoulders[key] = sm.left_shoulder
            self.right_shoulders[key] = sm.right_shoulder
            self.left_elbows[key] = sm.left_elbow
            self.right_elbows[key] = sm.right_elbow

        self.num_active_players_smoother.add_value(self.num_active_players)
        self.num_active_players_smoother.update()

    def reset(self) -> None:
        for sm in self.smooth_metrics_dict.values():
            sm.reset()
        self.update()

    def is_player_present(self, index: int) -> bool:
        return self.presents.get(index, False)

