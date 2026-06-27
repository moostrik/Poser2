"""Slow-speed light layers — the < ~200 rpm "low pixel system".

At slow rotation the pixel bar no longer blurs into a persistence-of-vision ring; instead each
output pixel drives a discrete physical lamp on the bar. The hardware mapping (per channel, with
R = light resolution) is:

    WHITE channel
        white[0]      → white lamps at the FRONT of the pixel bar
        white[R // 2] → white lamps at the BACK   (the middle pixel of all pixels)
    BLUE channel
        blue[0]       → blue lamps on the LEFT side
        blue[R // 2]  → blue lamps on the RIGHT side (the halfway blue pixel)

Layers in this package light those specific pixels rather than drawing a ring. Above ~200 rpm the
bar blurs into a ring and the high-speed layers take over (the compositor crossfades by motor rpm).
"""

from .playhead_flash import PlayheadFlash, PlayheadFlashSettings, phase_to_level
from .playhead       import Playhead,      PlayheadSettings
