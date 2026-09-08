"""Low-regime light layers (`LowLayer`) — the < ~200 rpm "low pixel system".

At slow rotation the pixel bar no longer blurs into a persistence-of-vision ring; instead each
output pixel drives a discrete physical lamp on the bar. The hardware mapping (per channel, with
R = light resolution) is:

    WHITE channel
        white[0]      → white lamps at the FRONT of the pixel bar
        white[R // 2] → white lamps at the BACK   (the middle pixel of all pixels)
    BLUE channel
        blue[0]       → blue lamps on the LEFT side
        blue[R // 2]  → blue lamps on the RIGHT side (the halfway blue pixel)

Layers in this package light those specific pixels rather than drawing a ring (the `LowLayer`
base encodes the mapping as named lamps). Above ~200 rpm the bar blurs into a ring and the
high-regime layers take over — the states choose per regime. Show layers and ``test_``-prefixed
debug layers live side by side: the name carries the role, the folder carries the regime.
"""

from .playhead_low        import PlayheadLow,   PlayheadLowSettings
from .playhead_flash      import PlayheadFlash, PlayheadFlashSettings, offset_to_level
from .sound_light         import SoundLight,    SoundLightSettings
from .wind_down           import WindDown,      WindDownSettings
from .test_haunted_flash  import HauntedFlash,  HauntedFlashSettings
from .test_slow           import TestSlow,      TestSlowSettings
