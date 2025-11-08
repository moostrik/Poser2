# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.Pose import Pose


class PoseDeltaExtractor(PoseFilterBase):
    """Computes frame-to-frame changes (deltas) for a single pose.

    Calculates:
    - Angle displacement: Angular change (with proper wrapping) since last frame

    Handles occlusion: Sets deltas to NaN when joints reappear after being invalid.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prev_pose: Pose | None = None

    def process(self, pose: Pose) -> Pose:
        # Compute deltas (or empty if no previous pose)
        if self._prev_pose is None:
            delta_data: PoseAngleData = PoseAngleData.create_empty()
        else:
            delta_data = pose.angle_data.subtract(self._prev_pose.angle_data)

        # Update state for next frame
        self._prev_pose = pose

        # Create enriched pose
        enriched_pose: Pose = replace(
            pose,
            delta_data=delta_data
        )

        # Cleanup if pose is lost
        if pose.lost:
            self._prev_pose = None

        return enriched_pose

    def reset(self) -> None:
        self._prev_pose = None




