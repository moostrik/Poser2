from dataclasses import replace, dataclass

import numpy as np

from modules.pose.Pose import Pose, PoseDict, Rect
from modules.pose.features.PosePoints import PosePointData, POSE_NUM_JOINTS, PoseJoint
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS, AngleJoint
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.utils.Smoothing import OneEuroFilter, OneEuroFilterAngular
from modules.Settings import Settings


@dataclass
class SmootherSettings:
    """Configuration for point smoothing."""
    frequency: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.025
    d_cutoff: float = 1.0
    reset_on_reappear: bool = False


class PosePointSmoother(PoseFilterBase[SmootherSettings]):
    """Smooths pose keypoint positions using OneEuroFilter."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = SmootherSettings(
            frequency=settings.camera_fps,
            min_cutoff=2.0,
            beta=0.05,
            d_cutoff=1.0,
            reset_on_reappear=False
        )

        # Per-tracklet state: tracklet_id -> (filters, prev_valid)
        self._tracklets: dict[int, tuple[list[tuple[OneEuroFilter, OneEuroFilter]], np.ndarray]] = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Smooth keypoint positions for all poses."""
        smoothed_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Initialize tracklet filters if needed
            if tracklet_id not in self._tracklets:
                filters = [
                    (OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
                     OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff))
                    for _ in range(POSE_NUM_JOINTS)
                ]
                prev_valid = np.zeros(POSE_NUM_JOINTS, dtype=bool)
                self._tracklets[tracklet_id] = (filters, prev_valid)

            filters, prev_valid = self._tracklets[tracklet_id]
            timestamp: float = pose.time_stamp.timestamp()

            # Smooth points
            smoothed_values: np.ndarray = pose.point_data.values.copy()
            smoothed_values.flags.writeable = True

            for joint in PoseJoint:
                is_valid = pose.point_data.valid_mask[joint]
                was_valid = prev_valid[joint]

                if is_valid:
                    x, y = pose.point_data.values[joint]
                    x_filter, y_filter = filters[joint]

                    # Reset if joint reappeared
                    if not was_valid and self.settings.reset_on_reappear:
                        x_filter.reset()
                        y_filter.reset()

                    smoothed_values[joint] = [x_filter(x, timestamp), y_filter(y, timestamp)]

                prev_valid[joint] = is_valid

            smoothed_point_data = PosePointData(smoothed_values, pose.point_data.scores)
            smoothed_pose: Pose = replace(pose, point_data=smoothed_point_data)
            smoothed_poses[pose_id] = smoothed_pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._tracklets[tracklet_id]

        self._notify_callbacks(smoothed_poses)

    def _update_settings(self, settings: SmootherSettings) -> None:
        """Update filter parameters for all tracklets."""
        super()._update_settings(settings)
        for filters, _ in self._tracklets.values():
            for x_filter, y_filter in filters:
                x_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
                y_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)



class PoseAngleSmoother(PoseFilterBase[SmootherSettings]):
    """Smooths pose joint angles using OneEuroFilterAngular."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = SmootherSettings(
            frequency=settings.camera_fps,
            min_cutoff=2.0,
            beta=0.05,
            d_cutoff=1.0,
            reset_on_reappear=False
        )

        # Per-tracklet state: tracklet_id -> (filters, prev_valid)
        self._tracklets: dict[int, tuple[list[OneEuroFilterAngular], np.ndarray]] = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Smooth joint angles for all poses."""
        smoothed_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Initialize tracklet filters if needed
            if tracklet_id not in self._tracklets:
                filters = [
                    OneEuroFilterAngular(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
                    for _ in range(ANGLE_NUM_JOINTS)
                ]
                prev_valid = np.zeros(ANGLE_NUM_JOINTS, dtype=bool)
                self._tracklets[tracklet_id] = (filters, prev_valid)

            filters, prev_valid = self._tracklets[tracklet_id]
            timestamp: float = pose.time_stamp.timestamp()

            # Smooth angles
            smoothed_angles: np.ndarray = pose.angle_data.values.copy()
            smoothed_angles.flags.writeable = True

            for angle_joint in AngleJoint:
                is_valid = pose.angle_data.valid_mask[angle_joint]
                was_valid = prev_valid[angle_joint]

                if is_valid:
                    angle = float(pose.angle_data.values[angle_joint])
                    angle_filter = filters[angle_joint]

                    # Reset if angle reappeared
                    if not was_valid and self.settings.reset_on_reappear:
                        angle_filter.reset()

                    smoothed_angles[angle_joint] = angle_filter(angle, timestamp)

                prev_valid[angle_joint] = is_valid

            smoothed_angle_data = PoseAngleData(smoothed_angles, pose.angle_data.scores)
            smoothed_pose: Pose = replace(pose, angle_data=smoothed_angle_data)
            smoothed_poses[pose_id] = smoothed_pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._tracklets[tracklet_id]

        self._notify_callbacks(smoothed_poses)

    def _update_settings(self, settings: SmootherSettings) -> None:
        """Update filter parameters for all tracklets."""
        super()._update_settings(settings)
        for filters, _ in self._tracklets.values():
            for angle_filter in filters:
                angle_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)


class PoseBBoxSmoother(PoseFilterBase[SmootherSettings]):
    """Smooths pose bounding boxes using OneEuroFilter."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = SmootherSettings(
            frequency=settings.camera_fps,
            min_cutoff=2.0,
            beta=0.05,
            d_cutoff=1.0
        )

        # Per-tracklet state: tracklet_id -> (x, y, w, h filters)
        self._tracklets: dict[int, tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter]] = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Smooth bounding boxes for all poses."""
        smoothed_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Initialize tracklet filters if needed
            if tracklet_id not in self._tracklets:
                filters = (
                    OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
                    OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
                    OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
                    OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
                )
                self._tracklets[tracklet_id] = filters

            x_filter, y_filter, w_filter, h_filter = self._tracklets[tracklet_id]
            timestamp: float = pose.time_stamp.timestamp()

            # Smooth bounding box
            smoothed_x = x_filter(pose.bounding_box.x, timestamp)
            smoothed_y = y_filter(pose.bounding_box.y, timestamp)
            smoothed_w = w_filter(pose.bounding_box.width, timestamp)
            smoothed_h = h_filter(pose.bounding_box.height, timestamp)

            smoothed_bbox = Rect(smoothed_x, smoothed_y, smoothed_w, smoothed_h)
            smoothed_pose: Pose = replace(pose, bounding_box=smoothed_bbox)
            smoothed_poses[pose_id] = smoothed_pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._tracklets[tracklet_id]

        self._notify_callbacks(smoothed_poses)

    def _update_settings(self, settings: SmootherSettings) -> None:
        """Update filter parameters for all tracklets."""
        super()._update_settings(settings)
        for x_filter, y_filter, w_filter, h_filter in self._tracklets.values():
            x_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
            y_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
            w_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
            h_filter.setParameters(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)