"""Voice — one person's instance of the light synth (``docs/LIGHT_SYNTH.md``, *The voice*).

Two oscillators, one per output, on one time; the amp stage (each side's window, and presence)
after the slots; each oscillator's push on its speed, and its strobe on its lines (``strobe.py``,
on the clock's tick index, so every voice's strobe sits on one grid); one LFO in time, whose
output is a source for the caller to wire. The patch, the settings objects given to the constructor,
is shared by every voice; what flows through it, the sources, is each voice's own. A voice knows
nothing of colour or pose: what its outputs are projected in, what feeds its sources, what its
reaches are and when it is hit are the caller's. The caller gives each pixel's signed position
from the person; each oscillator mirrors it (both sides the same) or runs through it (one pattern
passing behind the person), by its ``mirror`` switch.
"""

from enum import IntEnum, auto

import numpy as np

from .envelope import Envelope, WindowSettings
from .oscillator import Oscillator, OscillatorSettings, LfoSettings, Value
from .slot import Slot
from .strobe import Strobe, StrobeSettings

_LFO_POSITION = np.zeros(1)              # an LFO in time has one position


class Parameter(IntEnum):
    """An oscillator's parameters and its strobe's: the keys of its sources."""
    PITCH         = 0
    PULSE_WIDTH   = auto()
    PHASE         = auto()
    SPEED         = auto()
    HARDNESS      = auto()
    STROBE_RATE   = auto()
    STROBE_WIDTH  = auto()
    STROBE_PHASE  = auto()
    STROBE_SHIFT  = auto()


Sources = dict[Parameter, Value]        # a parameter with no source is its base


class Voice:
    """One person's pattern; see the module docstring."""

    def __init__(self, oscillator_1: OscillatorSettings, oscillator_2: OscillatorSettings,
                 window: WindowSettings, lfo: LfoSettings,
                 strobe_1: StrobeSettings, strobe_2: StrobeSettings, turn: float = 360.0) -> None:
        """``turn`` is one revolution in the positions' units: a pitch of *n* lines per revolution
        is an interval of ``turn / n``."""
        self._patches = (oscillator_1, oscillator_2)
        self._strobe_patches = (strobe_1, strobe_2)
        self._window = window
        self._lfo_settings = lfo
        self._turn = turn
        self._oscillators = (Oscillator(), Oscillator())
        self._lfo_oscillator = Oscillator()
        self._lfo = 0.0                                                     # this tick's LFO output, −1..1
        self._presence = Envelope()
        self._pushes = (Envelope(), Envelope())                             # each oscillator's push
        self._intervals = [self._interval(oscillator_1.pitch, 0.0), self._interval(oscillator_2.pitch, 0.0)]   # this tick's, after the slot
        self._strobes: list[tuple[int, float, float, float]] = [(0, 1.0, 0.0, 0.0)] * 2   # this tick's rate, width, phase, shift, after the slots
        self._tick = 0                                                      # the clock's tick index this tick
        self._ticks_per_second = 1

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

    def update(self, dt: float, present: bool, hit: bool, sources: tuple[Sources, Sources], min_interval: float,
               tick: int, ticks_per_second: int) -> None:
        """Advance the voice one tick: presence follows its gate, a hit opens each oscillator's
        push, each travels at its speed plus what its push adds, and each strobe reads its slots.
        ``min_interval`` is the visual limit's floor on the interval, in the positions' units;
        ``tick`` is the clock's tick index, the strobes' shared time."""
        W = self._window
        self._presence.update(present, dt, W.attack_seconds, W.release_seconds)
        self._tick, self._ticks_per_second = int(tick), max(1, int(ticks_per_second))
        for i, (oscillator, patch, strobe, source, push) in enumerate(zip(self._oscillators, self._patches, self._strobe_patches, sources, self._pushes)):
            pitch_source = self._played(patch.pitch_bypass, patch.pitch_curve, float(source.get(Parameter.PITCH, 0.0)))
            pitch = Slot.modulate(patch.pitch, patch.pitch_amount, pitch_source)
            self._intervals[i] = self._interval(pitch, min_interval)
            speed_source = self._played(patch.speed_bypass, patch.speed_curve, float(source.get(Parameter.SPEED, 0.0)))
            speed = Slot.modulate(patch.speed, patch.speed_amount, speed_source)
            pushed = push.update(hit, dt, 0.0, patch.push_release_seconds)
            oscillator.update(dt, self._intervals[i], float(speed) + patch.push * pushed)
            self._strobes[i] = self._strobe(strobe, source)

    def _strobe(self, S: StrobeSettings, source: Sources) -> tuple[int, float, float, float]:
        """A strobe's four parameters this tick, each through its slot: the rate quantized to the
        powers of two, the width a fraction, the phase and the shift as they come."""
        rate   = Strobe.rate(float(Slot.modulate(S.rate, S.rate_amount, self._played(S.rate_bypass, S.rate_curve, float(source.get(Parameter.STROBE_RATE, 0.0))))), self._ticks_per_second)
        width  = float(Slot.unit(Slot.modulate(S.width, S.width_amount, self._played(S.width_bypass, S.width_curve, float(source.get(Parameter.STROBE_WIDTH, 0.0))))))
        phase  = float(Slot.modulate(S.phase, S.phase_amount, self._played(S.phase_bypass, S.phase_curve, float(source.get(Parameter.STROBE_PHASE, 0.0)))))
        shift  = float(Slot.modulate(S.shift, S.shift_amount, self._played(S.shift_bypass, S.shift_curve, float(source.get(Parameter.STROBE_SHIFT, 0.0)))))
        return rate, width, phase, shift

    @staticmethod
    def _played(bypass: bool, curve: int, source: Value) -> Value:
        """A source as its slot sees it: nothing while bypassed, else eased by its curve."""
        return Slot.curve(Slot.bypassed(bypass, source), int(curve))

    def render(self, position: np.ndarray, reach_left: float, reach_right: float,
               sources: tuple[Sources, Sources]) -> tuple[np.ndarray, np.ndarray]:
        """The two outputs over a strip of pixels, 0..1. ``position`` is each pixel's signed
        angle from the person, negative on their left; the reaches are the caller's, in the same
        units, and presence multiplies them. A mirrored oscillator reads the position without its
        sign, an unmirrored one as it is. The window thins the pulse width after its slot, so a
        dark output stays dark; the strobe gates whole lines after the window."""
        reach = np.where(position < 0.0, reach_left, reach_right) * self._presence.value
        taper = reach * self._window.taper
        outputs = []
        for oscillator, patch, source, interval, strobe in zip(self._oscillators, self._patches, sources, self._intervals, self._strobes):
            if not patch.enabled:                                           # switched off: dark, whatever its slots say
                outputs.append(np.zeros(position.shape, dtype=np.float32))
                continue
            pulse_width_source = self._played(patch.pulse_width_bypass, patch.pulse_width_curve, source.get(Parameter.PULSE_WIDTH, 0.0))
            phase_source = self._played(patch.phase_bypass, patch.phase_curve, source.get(Parameter.PHASE, 0.0))
            hardness_source = self._played(patch.hardness_bypass, patch.hardness_curve, source.get(Parameter.HARDNESS, 0.0))
            pulse_width = Slot.unit(Slot.modulate(patch.pulse_width, patch.pulse_width_amount, pulse_width_source))
            phase = Slot.modulate(patch.phase, patch.phase_amount, phase_source)
            hardness = Slot.unit(Slot.modulate(patch.hardness, patch.hardness_amount, hardness_source))
            positions = np.abs(position) if patch.mirror else position
            cycle = oscillator.cycle(positions, interval, phase)
            # The window is read at the centre of the line a pixel belongs to, not at the pixel, so
            # a line in the taper has one width: thinned, whole and still centred where it belongs.
            # The strobe is read there too, as the line's count from the person, so a line is
            # never half strobed.
            centre = positions - ((cycle + 0.5) % 1.0 - 0.5) * interval
            line_centre = np.abs(centre)
            window = Envelope.over_positions(line_centre, 0.0, taper, reach)
            output = Oscillator.pulse(cycle, pulse_width * window, hardness)
            if strobe[0] > 0:
                output *= Strobe.gate(self._tick, centre / interval, *strobe, self._ticks_per_second)
            outputs.append(output)
        return outputs[0], outputs[1]
