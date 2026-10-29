"""HD Trio render board — thread-safe shared data store for the HD Trio app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    SegmentationImageStoreMixin, CameraTrackletStoreMixin, SequencerStateStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
                 SegmentationImageStoreMixin, CameraTrackletStoreMixin, SequencerStateStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        SegmentationImageStoreMixin.__init__(self)
        CameraTrackletStoreMixin.__init__(self)
        SequencerStateStoreMixin.__init__(self)
