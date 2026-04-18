"""Deep Flow blackboard — thread-safe shared data store for the Deep Flow app."""

from modules.whiteboard import (
    FrameStoreMixin, WindowStoreMixin, ImageStoreMixin,
    DepthTrackletStoreMixin,
)


class Blackboard(FrameStoreMixin, WindowStoreMixin, ImageStoreMixin,
                 DepthTrackletStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        ImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
