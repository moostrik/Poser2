"""Deep Flow render board — thread-safe shared data store for the Deep Flow app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CameraDepthStoreMixin, CropImageStoreMixin,
    SegmentationImageStoreMixin, CameraTrackletStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CameraDepthStoreMixin, CropImageStoreMixin,
                 SegmentationImageStoreMixin, CameraTrackletStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CameraDepthStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        SegmentationImageStoreMixin.__init__(self)
        CameraTrackletStoreMixin.__init__(self)
