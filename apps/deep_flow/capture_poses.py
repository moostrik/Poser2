"""The poses that place deep_flow's center view: those of the newest capture, so the view is placed by
the pose of the very image and depth it shows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.board import HasCaptures

if TYPE_CHECKING:
    from modules.inference import Capture
    from modules.pose import Frame


class CapturePoses:
    """A stand-in for the board's frames where `CenterGeometry` reads its pose (`get_frame`, the one
    method it calls): the pose of the camera's newest capture, whatever stage is asked; the capture
    carries the SMOOTH stage's. One player per camera, so the id asked for is the camera's."""

    def __init__(self, board: HasCaptures) -> None:
        self._board: HasCaptures = board

    def get_frame(self, stage: int, track_id: int) -> "Frame | None":
        capture: "Capture | None" = self._board.get_capture(track_id)
        if capture is None:
            return None
        return next((pose for pose in capture.poses.values() if pose.cam_id == track_id), None)
