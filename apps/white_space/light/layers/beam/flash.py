"""BeamFlash — the INTRO flash: the beam lights strike on at each player's hit, the tick the rotating
playhead is closest to them.

The hit is the pose's ``Hit`` (``pose/hit.py``: the one crossing detector, a LERP step, so the flash, the
instrument and the state machine agree by construction), and the flash lasts ``frames`` ticks (1–3) from
it: the hit tick alone, or the hit tick and the ticks after. The hit needs no history, so the tick that
starts INTRO flashes even though INTRO resets this layer on entry. Every hit flashes the same — the
brightness is plain ``white`` / ``blue``, one level for everyone.

Every tick it lights, the flash is also posted to the board (``add_flash``: the playhead heading
and the white and blue flash levels), so the render can mark it longer than the ticks it lasts on
the fixture.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BeamLayer, LayerSettings
from ...frame import LightFrame
from ....pose import Hit


class BeamFlashSettings(LayerSettings):
    base_white: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Base brightness of the front white beam light")
    base_blue:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Base brightness of both blue beam lights")
    white:      Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="White flash brightness", newline=True)
    blue:       Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Blue flash brightness")
    frames:     Field[int]   = Field(1,   min=1,   max=3,   step=1,    description="Flash length: the hit tick and the ticks after it, 1–3", newline=True)


class BeamFlash(BeamLayer):
    """Constant base level plus a flash of ``frames`` ticks from each player's hit, read from the poses'
    ``Hit``; see the module docstring."""

    def __init__(self, resolution: int, settings: BeamFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, settings, board)
        self._settings: BeamFlashSettings = settings
        self._pose_stage = pose_stage
        self._remaining = 0                 # ticks of the current flash still to light, after this one

    def reset(self) -> None:
        self._remaining = 0

    def _draw(self, frame: LightFrame, beam_lights: np.ndarray) -> None:
        P = self._settings
        if any(pose[Hit].on for pose in self._board.get_frames(self._pose_stage).values()):
            self._remaining = int(P.frames)
        lit = self._remaining > 0
        self._remaining = max(self._remaining - 1, 0)

        flash_white = P.white if lit else 0.0
        flash_blue = P.blue if lit else 0.0
        if lit and not math.isnan(frame.playhead):
            self._board.add_flash(frame.playhead, flash_white, flash_blue, frame.tick.now)

        # The flash is the front white lamp plus both blue lamps, on a constant base.
        self._add_beam_lights(beam_lights,
                             front_white=P.base_white + flash_white,
                             left_blue=P.base_blue + flash_blue,
                             right_blue=P.base_blue + flash_blue)
