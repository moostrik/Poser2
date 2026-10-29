"""White Space render board — thread-safe shared data store for the White Space app."""

from modules.board import (
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    CameraTrackletStoreMixin, TrackletStoreMixin, ObservationStoreMixin,
    VideoImageStoreMixin, SequencerStateStoreMixin,
    PlayheadStoreMixin, SoundLevelStoreMixin, FlashStoreMixin, HitStreakStoreMixin,
)

from .light_frame_store import LightFrameStoreMixin


class RenderBoard(
    FrameStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    CameraTrackletStoreMixin, TrackletStoreMixin, ObservationStoreMixin,
    VideoImageStoreMixin, LightFrameStoreMixin, SequencerStateStoreMixin,
    PlayheadStoreMixin, SoundLevelStoreMixin, FlashStoreMixin, HitStreakStoreMixin,
):
    """Thread-safe blackboard for the White Space app.

    All pipeline stages (compositor, cameras, pose, tracking) write their own
    slices here. Render layers and output modules poll whatever slices they need.
    """

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        CameraTrackletStoreMixin.__init__(self)
        TrackletStoreMixin.__init__(self)
        ObservationStoreMixin.__init__(self)
        VideoImageStoreMixin.__init__(self)
        LightFrameStoreMixin.__init__(self)
        SequencerStateStoreMixin.__init__(self)
        PlayheadStoreMixin.__init__(self)
        SoundLevelStoreMixin.__init__(self)
        FlashStoreMixin.__init__(self)
        HitStreakStoreMixin.__init__(self)
