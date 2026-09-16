"""BeamFlash — the INTRO flash: the beam lights strike on for the ticks the rotating playhead is
closest to each player, driven by the continuous ``PlayheadOffset``.

Each pass flashes exactly ``frames`` ticks (1–3) per person: the ticks closest to the crossing,
measured in playhead steps at ``beam_rpm`` (``ticks_to_crossing``: 1 = the closest, 2 = the pair
straddling it, 3 = the closest and both neighbours). Which ticks light is ``PlayheadCrossing``'s
rule (``pose/playhead_offset.py``): no history is needed for the closest tick, so the tick that starts
INTRO flashes even though INTRO resets this layer on entry, and a jittered pass lights neither an
extra tick nor none. Every hit flashes the same — the brightness is plain ``white`` / ``blue``, one
level for everyone. The ghost-driven variant is the separate ``beam_haunted`` debug layer.

The state machine's hit is the same closest tick (``statemachine/machine.py``), so INTRO starts on
the flash.

Every tick it lights, the flash is also posted to the board (``add_flash``: the playhead heading
and the white and blue flash levels), so the render can mark it longer than the ticks it lasts on
the fixture.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BeamLayer, LayerSettings
from ...frame import Frame
from ....pose import PlayheadCrossing, PlayheadOffset, playhead_step


class BeamFlashSettings(LayerSettings):
    base_white: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Base brightness of the front white lamp")
    base_blue:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Base brightness of both blue lamps")
    white:      Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="White flash brightness", newline=True)
    blue:       Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Blue flash brightness")
    frames:     Field[int]   = Field(1,   min=1,   max=3,   step=1,    description="Flash length: the ticks closest to the crossing, 1–3", newline=True)


class BeamFlash(BeamLayer):
    """Constant base level plus a flash of the ``frames`` ticks closest to each player's crossing,
    read from ``PlayheadOffset``; see the module docstring."""

    def __init__(self, resolution: int, config: BeamFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage
        self._crossing = PlayheadCrossing()

    def reset(self) -> None:
        self._crossing.reset()

    def _draw(self, frame: Frame, beam_lights: np.ndarray) -> None:
        P = self._config
        step: float = playhead_step(frame.motor_command.beam_rpm, frame.tick.interval)
        offsets = {pose.track_id: pose[PlayheadOffset].value
                   for pose in self._board.get_frames(self._pose_stage).values()}
        lit = bool(self._crossing.update(offsets, step, int(P.frames)))

        flash_white = P.white if lit else 0.0
        flash_blue = P.blue if lit else 0.0
        if lit and not math.isnan(frame.playhead):
            self._board.add_flash(frame.playhead, flash_white, flash_blue)

        # The flash is the front white lamp plus both blue lamps, on a constant base.
        self._add_beam_lights(beam_lights,
                             front_white=P.base_white + flash_white,
                             left_blue=P.base_blue + flash_blue,
                             right_blue=P.base_blue + flash_blue)
