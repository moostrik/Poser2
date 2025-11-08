"""Tracks pose presence and detects unexpected disappearances."""
from dataclasses import replace
from typing import Callable
import warnings

from modules.pose.Pose import Pose, PoseDict, PoseDictCallback


class PosePresenceTracker:
    """Tracks which poses are present and warns when they disappear unexpectedly.

    Monitors PoseDict across frames and detects when pose IDs disappear
    without being properly marked as lost first. Can optionally broadcast
    fixed poses with lost=True.
    """

    def __init__(self, warn: bool = True, fix: bool = False):
        """Initialize presence tracker.

        Args:
            warn: Whether to issue warnings when poses disappear unexpectedly.
            fix: Whether to broadcast last known pose with lost=True when pose disappears unexpectedly.
        """
        self._warn = warn
        self._fix = fix
        self._active_pose_ids: set[int] = set()
        self._lost_pose_ids: set[int] = set()
        self._last_poses: dict[int, Pose] = {}  # Store last known pose for each ID
        self._fix_callbacks: list[PoseDictCallback] = []

    def check(self, poses: PoseDict) -> set[int]:
        """Check for unexpectedly disappeared poses and return their IDs.

        Args:
            poses: Current pose dictionary.

        Returns:
            Set of pose IDs that disappeared unexpectedly (without being marked lost first).
        """
        current_ids = set(poses.keys())

        # Store current poses for future reference
        for pose_id, pose in poses.items():
            self._last_poses[pose_id] = pose

        # Update which poses are currently marked as lost
        currently_lost = {pose_id for pose_id, pose in poses.items() if pose.lost}

        # Find poses that disappeared
        disappeared_ids = self._active_pose_ids - current_ids

        # Filter out expected disappearances (poses that were marked lost)
        unexpected_disappeared = disappeared_ids - self._lost_pose_ids

        # Warn about unexpected disappearances
        if self._warn and unexpected_disappeared:
            for pose_id in unexpected_disappeared:
                warnings.warn(
                    f"Pose {pose_id} disappeared from dict without being marked lost",
                    RuntimeWarning,
                    stacklevel=2
                )

        # Fix unexpected disappearances by broadcasting last pose with lost=True
        if self._fix and unexpected_disappeared:
            fixed_poses: dict[int, Pose] = {}
            for pose_id in unexpected_disappeared:
                if pose_id in self._last_poses:
                    # Create copy of last pose with lost=True
                    last_pose = self._last_poses[pose_id]
                    fixed_pose = replace(last_pose, lost=True)
                    fixed_poses[pose_id] = fixed_pose

            # Broadcast fixed poses
            if fixed_poses:
                self._emit_fix_callbacks(fixed_poses)

        # Update tracking
        self._active_pose_ids = current_ids
        self._lost_pose_ids = currently_lost

        return unexpected_disappeared

    def reset(self) -> None:
        """Reset tracking state."""
        self._active_pose_ids.clear()
        self._lost_pose_ids.clear()
        self._last_poses.clear()

    # CALLBACKS
    def add_fix_callback(self, callback: PoseDictCallback) -> None:
        """Register callback for fixed poses (unexpectedly disappeared poses with lost=True).

        Args:
            callback: Function that receives dict of fixed poses.
        """
        self._fix_callbacks.append(callback)

    def remove_fix_callback(self, callback: PoseDictCallback) -> None:
        """Unregister a fix callback.

        Args:
            callback: Function to remove.
        """
        if callback in self._fix_callbacks:
            self._fix_callbacks.remove(callback)

    def _emit_fix_callbacks(self, fixed_poses: dict[int, Pose]) -> None:
        """Emit callbacks with fixed poses."""
        for callback in self._fix_callbacks:
            try:
                callback(fixed_poses)
            except Exception as e:
                warnings.warn(
                    f"PosePresenceTracker: Error in fix callback: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )

    @property
    def active_pose_ids(self) -> set[int]:
        """Get set of currently active pose IDs."""
        return self._active_pose_ids.copy()

    @property
    def lost_pose_ids(self) -> set[int]:
        """Get set of pose IDs currently marked as lost."""
        return self._lost_pose_ids.copy()