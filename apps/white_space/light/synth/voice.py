"""Voice — one person's instance of the light synth (``docs/LIGHT_SYNTH.md``, *The voice*).

Two oscillators, one per output, on one time; the amp stage (each side's window, and presence)
after the slots; each oscillator's push on its speed; one LFO in time, whose output is a source
for the caller to wire. The patch, the settings objects given to the constructor,
is shared by every voice; what flows through it, the sources, is each voice's own. A voice knows
nothing of colour or pose: what its outputs are projected in, what feeds its sources, what its
reaches are and when it is hit are the caller's.
"""

from enum import IntEnum, auto

import numpy as np

from .envelope import Envelope, WindowSettings
from .oscillator import Oscillator, OscillatorSettings, LfoSettings, Value
from .slot import Slot

_LFO_POSITION = np.zeros(1)              # an LFO in time has one position


class Parameter(IntEnum):
    """An oscillator's parameters: the keys of its sources."""
    PITCH       = 0
    PULSE_WIDTH = auto()
    PHASE       = auto()
    SPEED       = auto()
    HARDNESS    = auto()


Sources = dict[Parameter, Value]        # a parameter with no source is its base


class Voice:
    """One person's pattern; see the module docstring."""

    def __init__(self, oscillator_1: OscillatorSettings, oscillator_2: OscillatorSettings,
                 window: WindowSettings, lfo: LfoSettings, turn: float = 360.0) -> None:
        """``turn`` is one revolution in the positions' units: a pitch of *n* lines per revolution
        is an interval of ``turn / n``."""
        self._patches = (oscillator_1, oscillator_2)
        self._window = window
        self._lfo_settings = lfo
        self._turn = turn
        self._oscillators = (Oscillator(), Oscillator())
        self._lfo_oscillator = Oscillator()
        self._lfo = 0.0                                                     # this tick's LFO output, −1..1
        self._presence = Envelope()
        self._pushes = (Envelope(), Envelope())                             # each oscillator's push
        self._intervals = [self._interval(oscillator_1.pitch, 0.0), self._interval(oscillator_2.pitch, 0.0)]   # this tick's, after the slot

    def _interval(self, pitch: float, min_interval: float) -> float:
        """The interval a pitch gives, in the positions' units: floored by the visual limit, and
        never coarser than one line per half turn."""
        return max(self._turn / max(float(pitch), 2.0), min_interval)

    def reset(self) -> None:
        for oscillator in (*self._oscillators, self._lfo_oscillator):
            oscillator.reset()
        self._lfo = 0.0
        self._presence.reset()
        for push in self._pushes:
            push.reset()

    @property
    def lfo(self) -> float:
        """This tick's LFO output, −level..level: a source like any other, wired by the caller."""
        return self._lfo

    def update_lfo(self, dt: float, level_source: float) -> float:
        """Advance the LFO one tick and set its output. Its level comes from its slot, so at level
        0 it is silent and whatever it feeds is at its base."""
        L = self._lfo_settings
        self._lfo_oscillator.update(dt, 1.0, L.rate)                        # one position: the speed is the rate
        level = Slot.unit(Slot.modulate(L.level, L.level_amount, self._played(L.level_bypass, L.level_curve, level_source)))
        self._lfo = float(Oscillator.sine(self._lfo_oscillator.cycle(_LFO_POSITION, 1.0, L.phase), level)[0])
        return self._lfo

    @property
    def alive(self) -> bool:
        """False once presence has closed: the voice draws nothing and can be dropped."""
        return self._presence.level > 0.0

    @property
    def presence(self) -> float:
        return self._presence.value

    def update(self, dt: float, present: bool, hit: bool, sources: tuple[Sources, Sources], min_interval: float) -> None:
        """Advance the voice one tick: presence follows its gate, a hit opens each oscillator's
        push, and each travels at its speed plus what its push adds. ``min_interval`` is the
        visual limit's floor on the interval, in the positions' units."""
        W = self._window
        self._presence.update(present, dt, W.attack_seconds, W.release_seconds)
        for i, (oscillator, patch, source, push) in enumerate(zip(self._oscillators, self._patches, sources, self._pushes)):
            pitch_source = self._played(patch.pitch_bypass, patch.pitch_curve, float(source.get(Parameter.PITCH, 0.0)))
            pitch = Slot.modulate(patch.pitch, patch.pitch_amount, pitch_source)
            self._intervals[i] = self._interval(pitch, min_interval)
            speed_source = self._played(patch.speed_bypass, patch.speed_curve, float(source.get(Parameter.SPEED, 0.0)))
            speed = Slot.modulate(patch.speed, patch.speed_amount, speed_source)
            pushed = push.update(hit, dt, 0.0, patch.push_release_seconds)
            oscillator.update(dt, self._intervals[i], float(speed) + patch.push * pushed)

    @staticmethod
    def _played(bypass: bool, curve: int, source: Value) -> Value:
        """A source as its slot sees it: nothing while bypassed, else eased by its curve."""
        return Slot.curve(Slot.bypassed(bypass, source), int(curve))

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
            if not patch.enabled:                                           # switched off: dark, whatever its slots say
                outputs.append(np.zeros(distance.shape, dtype=np.float32))
                continue
            pulse_width_source = self._played(patch.pulse_width_bypass, patch.pulse_width_curve, source.get(Parameter.PULSE_WIDTH, 0.0))
            phase_source = self._played(patch.phase_bypass, patch.phase_curve, source.get(Parameter.PHASE, 0.0))
            hardness_source = self._played(patch.hardness_bypass, patch.hardness_curve, source.get(Parameter.HARDNESS, 0.0))
            pulse_width = Slot.unit(Slot.modulate(patch.pulse_width, patch.pulse_width_amount, pulse_width_source))
            phase = Slot.modulate(patch.phase, patch.phase_amount, phase_source)
            hardness = Slot.unit(Slot.modulate(patch.hardness, patch.hardness_amount, hardness_source))
            cycle = oscillator.cycle(distance, interval, phase)
            # The window is read at the centre of the line a pixel belongs to, not at the pixel, so
            # a line in the taper has one width: thinned, whole and still centred where it belongs.
            line_centre = np.abs(distance - ((cycle + 0.5) % 1.0 - 0.5) * interval)
            window = Envelope.over_positions(line_centre, 0.0, taper, reach)
            outputs.append(Oscillator.pulse(cycle, pulse_width * window, hardness))
        return outputs[0], outputs[1]
