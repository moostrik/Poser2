# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.Nodes import FilterNode
from modules.pose.features.AngleFeature import AngleFeature
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
            deltas: AngleFeature = AngleFeature.create_empty()
        else:
            deltas = pose.angles.subtract(self._prev_pose.angles)

        # Update state for next frame
        self._prev_pose = pose

        # Create enriched pose
        enriched_pose: Pose = replace(
            pose,
            deltas=deltas
        )

        # Cleanup if pose is lost
        if pose.lost:
            self._prev_pose = None

        return enriched_pose

    def reset(self) -> None:
        self._prev_pose = None




