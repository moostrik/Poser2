"""Tracker that logs pose lifecycle events (new, reset, updated)."""

from .TrackerBase import TrackerBase
from modules.pose.Pose import PoseDict


class DebugTracker(TrackerBase):
    """Tracks pose lifecycle and prints debug messages.

    Logs when poses are:
    - First seen (new)
    - Lost and reset
    - Updated (optional)
    """

    def __init__(self, num_players: int, log_updates: bool = False) -> None:
        """Initialize debug tracker.

        Args:
            num_players: Expected number of players/poses
            log_updates: If True, logs every update. If False, only logs new/reset events.
        """
        super().__init__()
        self._num_players = num_players
        self._tracked_ids: set[int] = set()
        self._log_updates = log_updates

    def process(self, poses: PoseDict) -> PoseDict:
        """Track pose lifecycle and log events."""
        current_ids = set(poses.keys())

        # Check if pose count exceeds expected number
        if len(current_ids) > self._num_players:
            print(f"DebugTracker: WARNING - More poses than expected! Found {len(current_ids)}, expected {self._num_players}")

        # Detect new poses
        new_ids = current_ids - self._tracked_ids
        for pose_id in new_ids:
            print(f"DebugTracker: New pose detected [id={pose_id}]")

        # Detect lost poses (reset)
        lost_ids = self._tracked_ids - current_ids
        for pose_id in lost_ids:
            print(f"DebugTracker: Pose lost/reset [id={pose_id}]")

        # Log updates if enabled
        if self._log_updates:
            for pose_id in current_ids & self._tracked_ids:
                print(f"DebugTracker: Pose updated [id={pose_id}]")

        # Update tracked set
        self._tracked_ids = current_ids

        # Pass through poses unchanged
        self._notify_poses_callbacks(poses)
        return poses

    def reset(self) -> None:
        """Reset tracker state."""
        if self._tracked_ids:
            print(f"DebugTracker: Reset all poses {self._tracked_ids}")
        self._tracked_ids.clear()

    def reset_at(self, id: int) -> None:
        """Reset tracking for specific pose."""
        if id in self._tracked_ids:
            print(f"DebugTracker: Reset pose [id={id}]")
            self._tracked_ids.discard(id)