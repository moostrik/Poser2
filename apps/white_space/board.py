"""White Space app-wide blackboard."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    SegmentationImageStoreMixin, DepthTrackletStoreMixin, TrackletStoreMixin,
    CompositionOutputStoreMixin, SequenceStoreMixin,
)

if TYPE_CHECKING:
    from .composition.transport import Transport


class Board(
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    SegmentationImageStoreMixin, DepthTrackletStoreMixin, TrackletStoreMixin,
    CompositionOutputStoreMixin, SequenceStoreMixin,
):
    """Thread-safe blackboard for the White Space app.

    All pipeline stages (compositor, cameras, pose, tracking) write their own
    slices here. Render layers and output modules poll whatever slices they need.
    """

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        SegmentationImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
        TrackletStoreMixin.__init__(self)
        CompositionOutputStoreMixin.__init__(self)
        SequenceStoreMixin.__init__(self)

        self._transport_lock = Lock()
        self._transport: Transport | None = None

    def get_transport(self) -> Transport | None:
        with self._transport_lock:
            return self._transport

    def set_transport(self, transport: Transport) -> None:
        with self._transport_lock:
            self._transport = transport
