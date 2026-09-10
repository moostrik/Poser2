"""WindDown — the dying wall of light: the S9/S10 ending fade.

Writes the two white beam lights at a fading level; everything else is physics. The fixture
is in beam mode from S9's first packet (its readout mode follows the commanded rpm, see
``inout/osc_light_sender.py``), so the two lamps spin at whatever speed the bar still has:
a wall of white while it is fast, thinning into two beams as it slows — and the fade rides
through both. One mechanism, and the layer never needs to know when the bar is slow. The
S8 → S9 hand-off is seamless at the DACs: the flood at 1.0 in projection mode drives the same two
white outputs as this layer at 1.0 in beam mode.

Fade level: ``f = 1 − ease(elapsed / spin_down_seconds)``, hand-tuned to ride the physical
spin-down — its visible slider is ``statemachine.spin_down_seconds`` (next to
``spin_up_seconds``, its mirror), shared into this layer's hidden field via the root. The
states put the landing look (the playhead line, the sound visuals) underneath at constant
weight; this layer fades the wall to nothing and thereby reveals it. ``reset()`` (a show
state's entry) restarts at the full wall. ``progress`` is the read-only fade readout the
states' exit and stage_progress (and S10's sound-visual reveal) ride.
"""

import numpy as np
import pytweening

from modules.settings import Field, Widget

from .._base_layer import BeamLayer, LayerSettings
from ...frame import Frame


class WindDownSettings(LayerSettings):
    level:             Field[float] = Field(1.0,  min=0.0, max=1.0,  step=0.01, description="Wall white level (the fade starts here and lands at 0)")
    spin_down_seconds: Field[float] = Field(10.0, min=1.0, max=60.0, step=0.5, visible=False, description="Timed wall fade (seconds) — shared from statemachine.spin_down_seconds, edit it there")
    progress:          Field[float] = Field(0.0,  min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Fade progress (0 = full wall, 1 = gone)")


class WindDown(BeamLayer):
    """The two white beam lights at the current fade level; see the module docstring."""

    def __init__(self, resolution: int, config: WindDownSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._elapsed: float = 0.0

    def reset(self) -> None:
        self._elapsed = 0.0
        self._config.progress = 0.0

    def _draw(self, frame: Frame, beam_lights: np.ndarray) -> None:
        self._elapsed += frame.tick.dt
        t = min(self._elapsed / max(self._config.spin_down_seconds, 1e-6), 1.0)
        f = 1.0 - pytweening.easeInOutSine(t)
        self._config.progress = 1.0 - f
        wall = f * self._config.level
        self._add_beam_lights(beam_lights, front_white=wall, back_white=wall)
