"""White Space app-wide blackboard."""

from modules.board import (
    FrameStoreMixin, GhostStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    DepthTrackletStoreMixin, TrackletStoreMixin, ObservationStoreMixin,
    VideoImageStoreMixin, CompositionOutputStoreMixin, SequenceStoreMixin,
    PlayheadStoreMixin, SoundLevelStoreMixin, FlashStoreMixin, HitStreakStoreMixin,
)


class Board(
    FrameStoreMixin, GhostStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    DepthTrackletStoreMixin, TrackletStoreMixin, ObservationStoreMixin,
    VideoImageStoreMixin, CompositionOutputStoreMixin, SequenceStoreMixin,
    PlayheadStoreMixin, SoundLevelStoreMixin, FlashStoreMixin, HitStreakStoreMixin,
):
    """Thread-safe blackboard for the White Space app.

    All pipeline stages (compositor, cameras, pose, tracking) write their own
    slices here. Render layers and output modules poll whatever slices they need.
    """

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        GhostStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        DepthTrackletStoreMixin.__init__(self)
        TrackletStoreMixin.__init__(self)
        ObservationStoreMixin.__init__(self)
        VideoImageStoreMixin.__init__(self)
        CompositionOutputStoreMixin.__init__(self)
        SequenceStoreMixin.__init__(self)
        PlayheadStoreMixin.__init__(self)
        SoundLevelStoreMixin.__init__(self)
        FlashStoreMixin.__init__(self)
        HitStreakStoreMixin.__init__(self)
