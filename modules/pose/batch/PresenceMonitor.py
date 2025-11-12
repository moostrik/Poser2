"""Monitor for pose presence and lifecycle validation."""

from dataclasses import replace
from time import time
import warnings

from modules.pose.Pose import Pose, PoseDict
from modules.pose.callback import PoseDictCallbackMixin


class PresenceMonitor(PoseDictCallbackMixin):
    """Monitors pose presence and validates lifecycle transitions.

    Tracks which poses are present/absent across frames and validates
    pose lifecycle (warns if poses disappear without being marked lost).

    Unlike node-based trackers (FilterTracker, GeneratorTracker), this is
    a stateful monitor that observes pose streams rather than transforming them.
    """

    def __init__(self, num_poses: int, warn: bool = True, fix: bool = False) -> None:
        """Initialize presence monitor.

        Args:
            num_poses: Number of poses to track.
            warn: If True, issue warnings when poses disappear unexpectedly.
            fix: If True, broadcast missing poses with lost=True and current timestamp.
        """
        super().__init__()  # Initialize mixin
        self.num_poses: int = num_poses
        self._warn: bool = warn
        self._fix: bool = fix
        self._previous_poses: dict[int, Pose] = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Process poses and check for unexpected losses.

        Args:
            poses: Current pose dictionary.
        """
        current_pose_ids: set[int] = set(poses.keys())
        all_pose_ids: set[int] = set(range(self.num_poses))

        # Find poses that should be present but aren't
        missing_pose_ids: set[int] = all_pose_ids - current_pose_ids

        for pose_id in missing_pose_ids:
            # Only process poses we've seen before
            if pose_id in self._previous_poses:
                previous_pose: Pose = self._previous_poses[pose_id]

                # Warn if pose disappeared without being marked lost
                if self._warn and not previous_pose.lost:
                    warnings.warn(
                        f"Pose {pose_id} disappeared without being marked as lost",
                        RuntimeWarning,
                        stacklevel=2
                    )

                # Fix: Add the lost pose with current timestamp
                if self._fix:
                    # Use current timestamp since this is a synthetic pose being emitted now
                    # This prevents duplicate timestamps and maintains temporal ordering
                    lost_pose: Pose = replace(previous_pose, lost=True, time_stamp=time())
                    poses[pose_id] = lost_pose

        # Update previous poses (store current state for next frame)
        self._previous_poses = poses.copy()

        # Emit to callbacks
        self._notify_pose_dict_callbacks(poses)

    def reset(self) -> None:
        """Reset all tracked poses."""
        self._previous_poses.clear()