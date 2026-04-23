"""Deep Flow render board — thread-safe shared data store for the Deep Flow app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    SegmentationImageStoreMixin, DepthTrackletStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
                 SegmentationImageStoreMixin, DepthTrackletStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        SegmentationImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
