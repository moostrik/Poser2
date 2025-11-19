# Standard library imports
from dataclasses import replace

from numpy import pi

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.features import Angles
from modules.pose.Pose import Pose


class DeltaExtractor(FilterNode):
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
            deltas: Angles = Angles.create_dummy()
        else:
            # Compute time delta
            dt = pose.time_stamp - self._prev_pose.time_stamp

            # print(dt)

            # Compute angular displacement
            angle_displacement = pose.angles.subtract(self._prev_pose.angles)

            # Convert to angular velocity (rad/s) by dividing by dt
            if dt > 0:
                angular_velocity_values = angle_displacement.values / dt
                # clamp at pi radians per second (already ridiculously high)
                angular_velocity_values = angular_velocity_values.clip(-pi, pi)
                deltas = Angles(values=angular_velocity_values, scores=angle_displacement.scores)
                # clamp at pi radians per second
            else:
                # Handle dt=0 case (same timestamp or clock issue)
                deltas = Angles.create_dummy()

        # Update state for next frame
        self._prev_pose = pose

        # Create enriched pose
        enriched_pose: Pose = replace(
            pose,
            deltas=deltas
        )

        return enriched_pose

    def reset(self) -> None:
        self._prev_pose = None




