"""The Sequence tab: a visitor's cycle step by step, with the grin's playback and the saver as its cards."""

from enum import Enum
from typing import Callable

import pytweening as easing

from modules.settings import BaseSettings, Field, Group, Widget

from ..save import ClipSaverSettings
from .grin_player import GrinPlayerSettings


class Easing(Enum):
    LINEAR       = 'Linear'
    SINE_IN      = 'Sine In'
    SINE_OUT     = 'Sine Out'
    SINE_IN_OUT  = 'Sine In Out'
    QUAD_IN      = 'Quad In'
    QUAD_OUT     = 'Quad Out'
    QUAD_IN_OUT  = 'Quad In Out'
    CUBIC_IN     = 'Cubic In'
    CUBIC_OUT    = 'Cubic Out'
    CUBIC_IN_OUT = 'Cubic In Out'
    QUART_IN     = 'Quart In'
    QUART_OUT    = 'Quart Out'
    QUART_IN_OUT = 'Quart In Out'
    QUINT_IN     = 'Quint In'
    QUINT_OUT    = 'Quint Out'
    QUINT_IN_OUT = 'Quint In Out'

_EASING_FUNCTIONS: dict[Easing, Callable[[float], float]] = {
    Easing.LINEAR:       easing.linear,
    Easing.SINE_IN:      easing.easeInSine,
    Easing.SINE_OUT:     easing.easeOutSine,
    Easing.SINE_IN_OUT:  easing.easeInOutSine,
    Easing.QUAD_IN:      easing.easeInQuad,
    Easing.QUAD_OUT:     easing.easeOutQuad,
    Easing.QUAD_IN_OUT:  easing.easeInOutQuad,
    Easing.CUBIC_IN:     easing.easeInCubic,
    Easing.CUBIC_OUT:    easing.easeOutCubic,
    Easing.CUBIC_IN_OUT: easing.easeInOutCubic,
    Easing.QUART_IN:     easing.easeInQuart,
    Easing.QUART_OUT:    easing.easeOutQuart,
    Easing.QUART_IN_OUT: easing.easeInOutQuart,
    Easing.QUINT_IN:     easing.easeInQuint,
    Easing.QUINT_OUT:    easing.easeOutQuint,
    Easing.QUINT_IN_OUT: easing.easeInOutQuint,
}

def easing_function(value: Easing) -> Callable[[float], float]:
    return _EASING_FUNCTIONS[value]


class SequencerSettings(BaseSettings):
    start:              Field[bool]   = Field(False, widget=Widget.button, description="Start a sequence (key Enter)")
    reset:              Field[bool]   = Field(False, widget=Widget.button, description="Reset the sequence")
    pause:              Field[bool]   = Field(False, pinned=True, description="No new visitors: a found face does not start a cycle (Start still works)")

    state:              Field[str]    = Field("", access=Field.READ, newline=True, description="Current sequence state")
    state_progress:     Field[float]  = Field(0.0, access=Field.READ, description="Progress of the current state")
    animation:          Field[str]    = Field("", access=Field.READ, description="Current animation")
    animation_progress: Field[float]  = Field(0.0, access=Field.READ, description="Progress of the current animation")

    # the steps of a visitor's cycle in the order they happen, one row each; the row's title names the step
    charge:             Field[float]  = Field(2.0, min=0.1, max=10.0, step=0.1, newline=True, row_label="Face found", label="Charge (s)", description="Time a face must stay qualified until the flash (s)")
    drain:              Field[float]  = Field(4.0, min=0.1, max=10.0, step=0.1, label="Drain (s)", description="Time the charge takes to fall back when the face is lost (s)")
    charge_ease:        Field[Easing] = Field(Easing.LINEAR, label="Ease", description="Easing of the face's turn to color")
    flash_in:           Field[float]  = Field(0.5, min=0.1, max=6.0, step=0.1, newline=True, row_label="Flash in", label="Duration (s)", description="The camera flashes; the capture is taken halfway (s)")
    flash_in_ease:      Field[Easing] = Field(Easing.QUAD_IN, label="Ease", description="Easing of the flash in")
    flash_hold:         Field[float]  = Field(0.5, min=0.0, max=3.0, step=0.1, label="Hold (s)", description="Hold the full flash at least this long before the capture appears (s)")
    flash_out_auto:     Field[bool]   = Field(True, newline=True, row_label="Flash out", label="Auto", description="Match the flash out to the measured wait for the first portrait step")
    flash_out:          Field[float]  = Field(2.2, min=0.1, max=6.0, step=0.1, label="Duration (s)", description="The capture appears out of the flash (s); set by Auto", access=Field.READWRITE)
    flash_out_ease:     Field[Easing] = Field(Easing.SINE_OUT, label="Ease", description="Easing of the flash out")
    timeout:            Field[float]  = Field(5.0, min=0.1, max=6.0, step=0.1, newline=True, label="Timeout (s)", description="The result stays after the grin, before the bloom (s)")
    bloom:              Field[float]  = Field(0.4, min=0.1, max=6.0, step=0.1, newline=True, row_label="Bloom", label="Duration (s)", description="The result blooms (s)")
    bloom_ease:         Field[Easing] = Field(Easing.QUAD_IN, label="Ease", description="Easing of the bloom")
    fade:               Field[float]  = Field(1.1, min=0.1, max=6.0, step=0.1, newline=True, row_label="Fade", label="Duration (s)", description="The result dissolves into the camera (s)")
    fade_ease:          Field[Easing] = Field(Easing.CUBIC_OUT, label="Ease", description="Easing of the fade")

    grin:               Group[GrinPlayerSettings] = Group(GrinPlayerSettings)     # the grin's playback, between the flash out and the timeout
    saver:              Group[ClipSaverSettings]  = Group(ClipSaverSettings)      # each finished visitor saved to output/
