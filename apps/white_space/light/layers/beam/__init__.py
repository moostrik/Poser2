"""Beam-mode light layers (`BeamLayer`).

Commanded below ``FIXTURE_PROJECTION_RPM`` the fixture is in beam mode: it drives its four
discrete lamps (front/back white, left/right blue) directly and reads no ring pixels. The
fixture's readout mode follows the *commanded* rpm, switching on receipt of the rpm regardless
of the bar's actual speed (see ``inout/osc_light_sender.py`` for the wire contract) — so a
spinning-down bar in beam mode shows its lamps smeared into a wall, not the ring.

Layers in this package therefore write the four beam lights by name (``Frame.beam_lights``,
indexed by ``BeamLightId``) and nothing else; the light sender maps them to the firmware's
pixel slots on the way out, and the render simulates them as beams on the walls. Commanded at
or above the threshold the fixture steps the ring and the projection layers take over — the
states choose per mode. Show layers and debug layers live side by side; each layer's docstring
states its role (``beam_haunted`` and ``beam_test`` are debug tools, never in a state's mix).
"""

from .blue_sound    import BlueSound,    BlueSoundSettings
from .playhead      import BeamPlayhead, BeamPlayheadSettings
from .flash         import Flash,        FlashSettings, offset_to_level
from .wind_down     import WindDown,     WindDownSettings
from .haunted       import Haunted,      HauntedSettings
from .test          import BeamTest,     BeamTestSettings
