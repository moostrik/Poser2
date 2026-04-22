"""White Space render board — thread-safe shared data store for the White Space app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, ImageStoreMixin,
    DepthTrackletStoreMixin, SequenceStoreMixin,
    TrackletStoreMixin, WSOutputStoreMixin,
)


class RenderBoard(FrameStoreMixin, WindowStoreMixin, ImageStoreMixin,
                 DepthTrackletStoreMixin, TrackletStoreMixin,
                 WSOutputStoreMixin, SequenceStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        ImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
        TrackletStoreMixin.__init__(self)
        WSOutputStoreMixin.__init__(self)
        SequenceStoreMixin.__init__(self)
