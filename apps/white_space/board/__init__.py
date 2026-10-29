"""White Space's board: the slices only this app reads, as protocol + mixin pairs like ``modules/board``'s,
and ``RenderBoard``, the app's composition of both."""

from .playhead import       HasPlayhead, PlayheadStoreMixin, PlayheadSignals
from .flashes import        HasFlashes, FlashStoreMixin, Flash
from .sound_levels import   HasSoundLevels, SoundLevelStoreMixin, SoundLevels
from .video_images import   HasVideoImages, VideoImageStoreMixin
from .light_frame import    HasLightFrame, LightFrameStoreMixin, HasCompositionDebug, CompositionDebugStoreMixin
from .render_board import   RenderBoard
