"""Deep Flow render board — thread-safe shared data store for the Deep Flow app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CaptureStoreMixin, CropImageStoreMixin,
    CameraTrackletStoreMixin, TimeSeriesStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CaptureStoreMixin, CropImageStoreMixin,
                 CameraTrackletStoreMixin, TimeSeriesStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CaptureStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        CameraTrackletStoreMixin.__init__(self)
        TimeSeriesStoreMixin.__init__(self)
