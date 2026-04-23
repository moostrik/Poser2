"""HD Trio render board — thread-safe shared data store for the HD Trio app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    MaskImageStoreMixin, DepthTrackletStoreMixin, SequenceStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
                 MaskImageStoreMixin, DepthTrackletStoreMixin, SequenceStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        MaskImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
        SequenceStoreMixin.__init__(self)
