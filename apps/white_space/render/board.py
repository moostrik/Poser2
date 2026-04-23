"""White Space render board — thread-safe shared data store for the White Space app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    SegmentationImageStoreMixin, DepthTrackletStoreMixin, SequenceStoreMixin,
    TrackletStoreMixin, CompositionOutputStoreMixin, CompositionDebugStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
                 SegmentationImageStoreMixin, DepthTrackletStoreMixin, TrackletStoreMixin,
                 CompositionOutputStoreMixin, CompositionDebugStoreMixin, SequenceStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        SegmentationImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
        TrackletStoreMixin.__init__(self)
        CompositionOutputStoreMixin.__init__(self)
        CompositionDebugStoreMixin.__init__(self)
        SequenceStoreMixin.__init__(self)
