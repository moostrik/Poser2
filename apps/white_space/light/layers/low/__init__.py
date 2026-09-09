"""Low-regime light layers (`LowLayer`) — the lamp regime.

Commanded below ``FIXTURE_SLOW_RPM`` the fixture is in slot mode: it drives its four discrete
lamps (front/back white, left/right blue) directly and reads no ring pixels. The fixture's
readout mode follows the *commanded* rpm, switching on receipt of the rpm regardless of the
bar's actual speed (see ``inout/osc_light_sender.py`` for the wire contract) — so a spinning-
down bar in slot mode shows its lamps smeared into a wall, not the ring.

Layers in this package therefore write the four bar lights by name (``Frame.bar_lights``,
indexed by ``BarLightId``) and nothing else; the light sender maps them to the firmware's
pixel slots on the way out, and the render simulates them as beams on the walls. Commanded
at or above the threshold the fixture steps the ring and the high-regime layers take over —
the states choose per regime. Show layers and debug layers live side by side; each layer's
docstring states its role (``playhead_haunted`` and ``playhead_test`` are debug tools, never
in a state's mix).
"""

from .playhead_low        import PlayheadLow,     PlayheadLowSettings
from .playhead_flash      import PlayheadFlash,   PlayheadFlashSettings, offset_to_level
from .sound_light         import SoundLight,      SoundLightSettings
from .wind_down           import WindDown,        WindDownSettings
from .playhead_haunted    import PlayheadHaunted, PlayheadHauntedSettings
from .playhead_test       import PlayheadTest,    PlayheadTestSettings
