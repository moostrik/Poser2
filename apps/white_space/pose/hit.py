"""Hit — the tick the playhead is closest to a player, stamped on the pose.

``Hit`` is a White Space pose feature: 1 on the one tick per pass the playhead is closest to the pose, 0 on
every other tick; ``on`` reads it as the flag, and a pose without it reads off. ``HitExtractor`` is the LERP
group step that writes it, the show's one ``PlayheadCrossing``: the flash, the instrument and the hit sync
read the pose's ``Hit`` instead of running a detector each, so they agree by construction. The crossing is
measured in playhead steps at ``beam_rpm`` and the light rate (``playhead_step``), from this tick's
``PlayheadOffset``; a pose absent from the frame forgets its pass. Runs on the light tick, inside LERP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.pose import FrameDict, NormalizedSingleValue, nodes, replace

from .playhead_offset import PlayheadCrossing, PlayheadOffset, playhead_step

if TYPE_CHECKING:
    from ..light import LightSettings


class Hit(NormalizedSingleValue):
    """The playhead crosses the pose this tick: 1 on the hit tick, 0 otherwise; see the module docstring."""

    @property
    def on(self) -> bool:
        """This tick is the pose's hit."""
        return self.value >= 0.5

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)


class HitExtractor(nodes.GroupNode):
    """Stamps ``Hit`` on every pose of the frame from the one crossing detector; see the module docstring."""

    def __init__(self, light: LightSettings) -> None:
        self._light = light
        self._crossing = PlayheadCrossing()

    def process(self, frames: FrameDict) -> FrameDict:
        step = playhead_step(self._light.motor.beam_rpm, 1.0 / self._light.light_rate)
        offsets = {id: frame[PlayheadOffset].value for id, frame in frames.items()}
        lit = self._crossing.update(offsets, step, 1)
        return {id: replace(frame, {Hit: Hit.from_value(1.0 if id in lit else 0.0)}) for id, frame in frames.items()}

    def reset(self) -> None:
        """Forget every pass under way."""
        self._crossing.reset()
