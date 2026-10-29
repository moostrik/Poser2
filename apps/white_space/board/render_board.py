"""White Space render board — thread-safe shared data store for the White Space app."""

from modules.board import (
    FrameStoreMixin, GroupStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    CameraTrackletStoreMixin, TrackletStoreMixin, ObservationStoreMixin, SequencerStateStoreMixin,
)

from .flashes import FlashStoreMixin
from .light_frame import LightFrameStoreMixin
from .playhead import PlayheadStoreMixin
from .sound_levels import SoundLevelStoreMixin
from .video_images import VideoImageStoreMixin


class RenderBoard(
    FrameStoreMixin, GroupStoreMixin, WindowStoreMixin, CameraImageStoreMixin, CropImageStoreMixin,
    CameraTrackletStoreMixin, TrackletStoreMixin, ObservationStoreMixin, SequencerStateStoreMixin,
    VideoImageStoreMixin, LightFrameStoreMixin, PlayheadStoreMixin, SoundLevelStoreMixin, FlashStoreMixin,
):
    """Thread-safe blackboard for the White Space app: the module slices (poses, windows, images, tracklets,
    the sequencer state) and the app's own (the playhead, the light frame, the flashes, the sound levels,
    the video images). Producers write their own slices; render layers and output modules poll what they need."""

    def __init__(self) -> None:
        FrameStoreMixin.__init__(self)
        GroupStoreMixin.__init__(self)
        WindowStoreMixin.__init__(self)
        CameraImageStoreMixin.__init__(self)
        CropImageStoreMixin.__init__(self)
        CameraTrackletStoreMixin.__init__(self)
        TrackletStoreMixin.__init__(self)
        ObservationStoreMixin.__init__(self)
        SequencerStateStoreMixin.__init__(self)
        VideoImageStoreMixin.__init__(self)
        LightFrameStoreMixin.__init__(self)
        PlayheadStoreMixin.__init__(self)
        SoundLevelStoreMixin.__init__(self)
        FlashStoreMixin.__init__(self)
