"""Low Flow render board — thread-safe shared data store for the Low Flow app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin, CameraTrackletStoreMixin,
    TimeSeriesStoreMixin, InferenceFrameStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, InferenceFrameStoreMixin, CropImageStoreMixin,
                 CameraTrackletStoreMixin, TimeSeriesStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        InferenceFrameStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        CameraTrackletStoreMixin.__init__(self)
        TimeSeriesStoreMixin.__init__(self)
