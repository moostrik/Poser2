# Standard library imports
from dataclasses import replace

from numpy import pi

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.features import Angles, AngleVelocity
from modules.pose.Frame import Frame
from modules.settings import Settings, Field


class AngleVelExtractorSettings(Settings):
    """Configuration for AngleVelExtractor."""
    fps: Field[float] = Field(30.0, access=Field.INIT)


class AngleVelExtractor(FilterNode):
    """Computes frame-to-frame changes (deltas) for a single pose.

    Calculates:
    - Angle displacement: Angular change (with proper wrapping) since last frame

    Handles occlusion: Sets deltas to NaN when joints reappear after being invalid.
    """

    def __init__(self, settings: AngleVelExtractorSettings | None = None) -> None:
        super().__init__()
        self._settings = settings if settings is not None else AngleVelExtractorSettings()
        self._prev_pose: Frame | None = None
        self._dt = 1.0 / self._settings.fps

    def process(self, pose: Frame) -> Frame:
        # Compute deltas (or empty if no previous pose)
        if self._prev_pose is None:
            angle_vel: AngleVelocity = AngleVelocity.create_dummy()
        else:
            # Compute time delta
            dt: float = self._dt

            # print(dt)

            # Compute angular displacement
            angle_displacement: Angles = pose.angles.subtract(self._prev_pose.angles)

            # Convert to angular velocity (rad/s) by dividing by dt
            if dt > 0:
                angular_velocity_values = angle_displacement.values / dt
                angle_vel = AngleVelocity(values=angular_velocity_values, scores=angle_displacement.scores)
            else:
                angle_vel = AngleVelocity.create_dummy()

        # Update state for next frame
        self._prev_pose = pose

        # Create enriched pose
        enriched_pose: Frame = replace(
            pose,
            angle_vel=angle_vel
        )

        return enriched_pose

    def reset(self) -> None:
        self._prev_pose = None




