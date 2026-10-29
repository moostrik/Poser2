"""Tracker that logs pose lifecycle events (new, reset, updated)."""

from .tracker_base import TrackerBase
from ..frame import FrameDict

import logging
logger = logging.getLogger(__name__)


class DebugTracker(TrackerBase):
    """Tracks pose lifecycle and prints debug messages.

    Logs when poses are:
    - First seen (new)
    - Lost and reset
    - Updated (optional)
    """

    def __init__(self, max_players: int, log_updates: bool = False) -> None:
        """Initialize debug tracker.

        Args:
            max_players: Expected number of players/poses
            log_updates: If True, logs every update. If False, only logs new/reset events.
        """
        super().__init__()
        self._max_players = max_players
        self._tracked_ids: set[int] = set()
        self._log_updates = log_updates

    def process(self, poses: FrameDict) -> FrameDict:
        """Track pose lifecycle and log events."""
        current_ids = set(poses.keys())

        # Check if pose count exceeds expected number
        if len(current_ids) > self._max_players:
            logger.warning(f"WARNING - More poses than expected! Found {len(current_ids)}, expected {self._max_players}")

        # Detect new poses
        new_ids = current_ids - self._tracked_ids
        for track_id in new_ids:
            logger.debug(f"New pose detected [id={track_id}]")

        # Detect lost poses (reset)
        lost_ids = self._tracked_ids - current_ids
        for track_id in lost_ids:
            logger.debug(f"Pose lost/reset [id={track_id}]")

        # Log updates if enabled
        if self._log_updates:
            for track_id in current_ids & self._tracked_ids:
                logger.debug(f"Pose updated [id={track_id}]")

        # Update tracked set
        self._tracked_ids = current_ids

        # Pass through poses unchanged
        self._notify_frames_callbacks(poses)
        return poses

    def reset(self) -> None:
        """Reset tracker state."""
        if self._tracked_ids:
            logger.debug(f"Reset all poses {self._tracked_ids}")
        self._tracked_ids.clear()

    def reset_at(self, id: int) -> None:
        """Reset tracking for specific pose."""
        if id in self._tracked_ids:
            logger.debug(f"Reset pose [id={id}]")
            self._tracked_ids.discard(id)