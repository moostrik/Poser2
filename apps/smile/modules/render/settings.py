from enum import Enum
from typing import Callable

import pytweening as easing

from modules.settings import BaseSettings, Field, Widget


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


class RenderSettings(BaseSettings):
    fullscreen:    Field[bool]  = Field(False, widget=Widget.switch, description="Render window fullscreen (key f)")
    show_camera:   Field[bool]  = Field(False, description="Draw the raw camera image on top")
    freeze:        Field[bool]  = Field(False, description="Hold the current frame")
    show_prompt:   Field[bool]  = Field(True, description="Draw the text field during diffusion")
    trail_active:  Field[bool]  = Field(True, description="Blend new diffusion images over the previous ones")
    trail:         Field[float] = Field(0.995, min=0.98, max=1.0, step=0.001, description="Trail amount")
    destroy_blur:  Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Blur while fading the last image out")
    destroy_bloom: Field[float] = Field(0.6, min=0.0, max=1.0, step=0.1, description="Bloom while fading the last image out")


class SequencerSettings(BaseSettings):
    start:              Field[bool]   = Field(False, widget=Widget.button, description="Start a sequence (key Enter)")
    reset:              Field[bool]   = Field(False, widget=Widget.button, description="Reset the sequence")

    state:              Field[str]    = Field("", access=Field.READ, description="Current sequence state")
    state_progress:     Field[float]  = Field(0.0, access=Field.READ, description="Progress of the current state")
    animation:          Field[str]    = Field("", access=Field.READ, description="Current animation")
    animation_progress: Field[float]  = Field(0.0, access=Field.READ, description="Progress of the current animation")

    destroy_in:         Field[float]  = Field(0.4, min=0.1, max=6.0, step=0.1, description="Bloom of the last image (s)")
    destroy_in_ease:    Field[Easing] = Field(Easing.QUAD_IN, description="Bloom easing")
    destroy_out:        Field[float]  = Field(1.1, min=0.1, max=6.0, step=0.1, description="Fade from the last image to the camera (s)")
    destroy_out_ease:   Field[Easing] = Field(Easing.CUBIC_OUT, description="Fade easing")
    camera_in:          Field[float]  = Field(1.0, min=0.1, max=6.0, step=0.1, description="Camera, face found (s)")
    camera_in_ease:     Field[Easing] = Field(Easing.LINEAR, description="Camera in easing")
    camera_out:         Field[float]  = Field(1.0, min=0.1, max=6.0, step=0.1, description="Camera, recording (s)")
    camera_out_ease:    Field[Easing] = Field(Easing.LINEAR, description="Camera out easing")
    flash_in:           Field[float]  = Field(0.5, min=0.1, max=6.0, step=0.1, description="Flash and capture (s)")
    flash_in_ease:      Field[Easing] = Field(Easing.QUAD_IN, description="Flash in easing")
    flash_out_auto:     Field[bool]   = Field(True, description="Match the flash out to the measured generation time")
    flash_out:          Field[float]  = Field(2.2, min=0.1, max=6.0, step=0.1, description="Flash out (s), set automatically when auto is on")
    flash_out_ease:     Field[Easing] = Field(Easing.SINE_OUT, description="Flash out easing")
    timeout:            Field[float]  = Field(5.0, min=0.1, max=6.0, step=0.1, description="Show the result before resetting (s)")
