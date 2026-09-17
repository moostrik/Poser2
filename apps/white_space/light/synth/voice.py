"""Voice — one person's instance of the light synth (``docs/LIGHT_SYNTH.md``, *The voice*).

Two oscillators, one per output, on one time; the amp stage (each side's window, and presence)
after the slots; the push on the speed. The patch, the settings objects given to the constructor,
is shared by every voice; what flows through it, the sources, is each voice's own. A voice knows
nothing of colour or pose: what its outputs are projected in, what feeds its sources, what its
reaches are and when it is hit are the caller's.
"""

from enum import IntEnum, auto

import numpy as np

from .envelope import Envelope, WindowSettings, PresenceSettings, PushSettings
from .oscillator import Oscillator, OscillatorSettings, Value
from .slot import Slot


class Input(IntEnum):
    """An oscillator's inputs: the keys of its sources."""
    INTERVAL    = 0
    PULSE_WIDTH = auto()
    PHASE       = auto()
    SPEED       = auto()
    HARDNESS    = auto()


Sources = dict[Input, Value]            # a missing input has no source: it is its base


class Voice:
    """One person's pattern; see the module docstring."""

    def __init__(self, oscillator_1: OscillatorSettings, oscillator_2: OscillatorSettings,
                 window: WindowSettings, presence: PresenceSettings, push: PushSettings) -> None:
        self._patches = (oscillator_1, oscillator_2)
        self._window = window
        self._presence_settings = presence
        self._push_settings = push
        self._oscillators = (Oscillator(), Oscillator())
        self._presence = Envelope()
        self._push = Envelope()
        self._intervals = [float(oscillator_1.interval), float(oscillator_2.interval)]     # this tick's, after the slot

    def reset(self) -> None:
        for oscillator in self._oscillators:
            oscillator.reset()
        self._presence.reset()
        self._push.reset()

    @property
    def alive(self) -> bool:
        """False once presence has closed: the voice draws nothing and can be dropped."""
        return self._presence.level > 0.0

    @property
    def presence(self) -> float:
        return self._presence.value

    def update(self, dt: float, present: bool, hit: bool, sources: tuple[Sources, Sources], min_interval: float) -> None:
        """Advance the voice one tick: presence follows its gate, a hit opens the push, and both
        oscillators travel at their speed plus what the push adds. ``min_interval`` is the visual
        limit's floor on the interval, in the positions' units."""
        P = self._presence_settings
        self._presence.update(present, dt, P.attack_seconds, P.release_seconds)
        push = self._push.update(hit, dt, 0.0, self._push_settings.settle_seconds)
        for i, (oscillator, patch, source) in enumerate(zip(self._oscillators, self._patches, sources)):
            interval = Slot.modulate_octaves(patch.interval, patch.interval_amount, float(source.get(Input.INTERVAL, 0.0)))
            self._intervals[i] = max(float(interval), min_interval)
            speed = Slot.modulate(patch.speed, patch.speed_amount, float(source.get(Input.SPEED, 0.0)))
            oscillator.update(dt, self._intervals[i], float(speed) + patch.push * push)

    def render(self, distance: np.ndarray, left: np.ndarray, reach_left: float, reach_right: float,
               sources: tuple[Sources, Sources]) -> tuple[np.ndarray, np.ndarray]:
        """The two outputs over a strip of pixels, 0..1. ``distance`` is each pixel's unsigned
        distance from the person and ``left`` which pixels are on their left; the reaches are the
        caller's, in the same units, and presence multiplies them. The window thins the pulse
        width after its slot, so a dark output stays dark."""
        reach = np.where(left, reach_left, reach_right) * self._presence.value
        taper = reach * self._window.taper
        outputs = []
        for oscillator, patch, source, interval in zip(self._oscillators, self._patches, sources, self._intervals):
            pulse_width = Slot.unit(Slot.modulate(patch.pulse_width, patch.pulse_width_amount, source.get(Input.PULSE_WIDTH, 0.0)))
            phase = Slot.modulate(patch.phase, patch.phase_amount, source.get(Input.PHASE, 0.0))
            hardness = Slot.unit(Slot.modulate(patch.hardness, patch.hardness_amount, source.get(Input.HARDNESS, 0.0)))
            cycle = oscillator.cycle(distance, interval, phase)
            # The window is read at the centre of the line a pixel belongs to, not at the pixel, so
            # a line in the taper has one width: thinned, whole and still centred where it belongs.
            line_centre = np.abs(distance - ((cycle + 0.5) % 1.0 - 0.5) * interval)
            window = Envelope.over_positions(line_centre, 0.0, taper, reach)
            outputs.append(Oscillator.pulse(cycle, pulse_width * window, hardness))
        return outputs[0], outputs[1]
