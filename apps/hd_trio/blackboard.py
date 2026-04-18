"""HD Trio blackboard — thread-safe shared data store for the HD Trio app."""

from modules.blackboard import (
    FrameStoreMixin, WindowStoreMixin, ImageStoreMixin,
    DepthTrackletStoreMixin, SequenceStoreMixin,
)


class Blackboard(FrameStoreMixin, WindowStoreMixin, ImageStoreMixin,
                 DepthTrackletStoreMixin, SequenceStoreMixin):

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        ImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
        SequenceStoreMixin.__init__(self)
