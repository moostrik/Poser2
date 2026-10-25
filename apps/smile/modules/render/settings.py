from enum import Enum
from typing import Callable

import pytweening as easing

from modules.credit import LINE as CREDIT_LINE, LINES as CREDIT_LINES, NAME as CREDIT_NAME, URL as CREDIT_URL
from modules.credit_scroll import DEFAULT_GAP_PX as CREDIT_SCROLL_GAP, DEFAULT_ROWS as CREDIT_SCROLL_ROWS
from modules.gl import WindowSettings
from modules.settings import BaseSettings, Field, Group, Widget
from modules.utils import Color
from modules.video.ffmpeg import Encoder


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


class FaceFill(Enum):
    FADE  = 'Fade'      # colour over the whole box
    RISE  = 'Rise'      # colour climbs from the bottom
    IRIS  = 'Iris'      # colour grows as a circle from the centre
    TRACE = 'Trace'     # no colour: the outline draws itself around the box
    SCAN  = 'Scan'      # as Rise, with a bright line at the colour's edge


class FaceEffect(Enum):
    DARK   = 'Dark'
    PIXELS = 'Pixels'
    BLUR   = 'Blur'
    FOG    = 'Fog'


class RenderSettings(BaseSettings):
    window:        Group[WindowSettings] = Group(WindowSettings)
    show_camera:   Field[bool]  = Field(False, description="Draw the raw camera image on top; no sequence starts", pinned=True)
    face_marks:    Field[bool]  = Field(False, description="Draw the faces' outlines, dots and reject reasons on the camera")
    face_color:    Field[Color] = Field(Color(0.2, 1.0, 0.2), description="Colour of the chosen face's box and dots")
    face_fill:     Field[FaceFill] = Field(FaceFill.FADE, description="How the chosen face's box turns to colour as its charge fills")
    face_effect:   Field[FaceEffect] = Field(FaceEffect.DARK, description="How a face's box shows its distance from good")
    face_amount:   Field[float] = Field(0.5, min=0.0, max=1.0, step=0.05, description="Strength of the effect on a face that is far from good")
    freeze:        Field[bool]  = Field(False, description="Hold the current frame")
    trail_active:  Field[bool]  = Field(True, description="Blend new diffusion images over the previous ones")
    trail:         Field[float] = Field(0.995, min=0.98, max=1.0, step=0.001, description="Trail amount")
    loop_trail:    Field[float] = Field(0.5, min=0.0, max=0.99, step=0.01, description="Trail while the grin animation plays (0 = none)")
    destroy_blur:  Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Blur while fading the last image out")
    destroy_bloom: Field[float] = Field(0.6, min=0.0, max=1.0, step=0.1, description="Bloom while fading the last image out")
    credit:        Field[bool]  = Field(True, newline=True, row_label="Credit", label="Show", description="Show the QR code in the corner, and the artist's line with the grin and the result")
    credit_name:   Field[str]   = Field(CREDIT_NAME, label="Name", description="The artist's name, the caption's first line")
    credit_lines:  Field[str]   = Field(CREDIT_LINES, access=Field.INIT, label="Lines file", description="Text file with the lines under the name, one per line")
    credit_line:   Field[str]   = Field(CREDIT_LINE, access=Field.READ, label="Line", description="The line under the name for the current visitor")
    credit_url:    Field[str]   = Field(CREDIT_URL, label="URL", description="Page the QR code opens")
    credit_camera: Field[bool]  = Field(False, newline=True, row_label="Caption", label="On camera", description="Also show the caption over the camera view, while waiting for a visitor")
    credit_players: Field[bool] = Field(False, label="On players", description="Show each clip's caption on the player walls (name top, line bottom)")
    credit_qr_players: Field[bool] = Field(True, label="QR on players", description="One cell of each player wall shows the QR code and the name instead of its clip, one play at a time")
    credit_size:   Field[int]   = Field(200, min=120, max=480, step=10, label="QR size (px)", description="Size of the QR code (px)")
    credit_scroll: Field[bool]  = Field(False, newline=True, row_label="Scroll", label="On players", description="Each word of the name goes round its own ring of the player walls without end, like a ticker")
    credit_scroll_rows: Field[str] = Field(CREDIT_SCROLL_ROWS, label="Rows", description="Rings the words go round, 0 (top) to 4, the first word on the first; 1 to 3 have 12 screens")
    credit_scroll_offset: Field[int] = Field(90, min=0, max=359, step=5, label="Offset (deg)", description="The second word starts this far round the sphere ahead of the first")
    credit_scroll_speed: Field[float] = Field(1.0, min=0.1, max=5.0, step=0.1, label="Speed (screens/s)", description="How fast the words move, in screen widths per second")
    credit_scroll_size: Field[int] = Field(512, min=64, max=640, step=16, label="Size (px)", description="Height of the letters with their edge, of the screen's 640 px, centred")
    credit_scroll_gap: Field[int] = Field(CREDIT_SCROLL_GAP, min=0, max=2048, step=16, label="Gap (px)", description="The wall between two neighbouring screens, as pixels the letters cross unseen")
    credit_scroll_reverse: Field[bool] = Field(False, label="Reverse", description="The words go the other way round the sphere")
    credit_scroll_last: Field[str] = Field('', access=Field.READ, label="Going round", description="Which word goes round which row now")


class SequencerSettings(BaseSettings):
    start:              Field[bool]   = Field(False, widget=Widget.button, description="Start a sequence (key Enter)")
    reset:              Field[bool]   = Field(False, widget=Widget.button, description="Reset the sequence")
    pause:              Field[bool]   = Field(False, pinned=True, description="No new visitors: a found face does not start a cycle (Start still works)")
    save:               Field[bool]   = Field(True, pinned=True, newline=True, label="Save visitors", description="Save each finished visitor (grin clip, capture, portrait) to output/ for the players")
    encoder:            Field[Encoder] = Field(Encoder.AUTO, access=Field.INIT, label="Encoder", description="Video encoder for the grin clip (Auto tries Intel, Nvidia, CPU)")

    state:              Field[str]    = Field("", access=Field.READ, newline=True, description="Current sequence state")
    state_progress:     Field[float]  = Field(0.0, access=Field.READ, description="Progress of the current state")
    animation:          Field[str]    = Field("", access=Field.READ, description="Current animation")
    animation_progress: Field[float]  = Field(0.0, access=Field.READ, description="Progress of the current animation")

    # the steps of a visitor's cycle in the order they happen, one row each; the row's title names the step
    charge:             Field[float]  = Field(2.0, min=0.1, max=10.0, step=0.1, newline=True, row_label="Face found", label="Charge (s)", description="Time a face must stay qualified until the flash (s)")
    drain:              Field[float]  = Field(4.0, min=0.1, max=10.0, step=0.1, label="Drain (s)", description="Time the charge takes to fall back when the face is lost (s)")
    charge_ease:        Field[Easing] = Field(Easing.LINEAR, label="Ease", description="Easing of the face's turn to colour")
    flash_in:           Field[float]  = Field(0.5, min=0.1, max=6.0, step=0.1, newline=True, row_label="Flash in", label="Duration (s)", description="The camera flashes; the capture is taken halfway (s)")
    flash_in_ease:      Field[Easing] = Field(Easing.QUAD_IN, label="Ease", description="Easing of the flash in")
    flash_hold:         Field[float]  = Field(0.5, min=0.0, max=3.0, step=0.1, label="Hold (s)", description="Hold the full flash at least this long before the capture appears (s)")
    flash_out_auto:     Field[bool]   = Field(True, newline=True, row_label="Flash out", label="Auto", description="Match the flash out to the measured wait for the first portrait step")
    flash_out:          Field[float]  = Field(2.2, min=0.1, max=6.0, step=0.1, label="Duration (s)", description="The capture appears out of the flash (s); set by Auto")
    flash_out_ease:     Field[Easing] = Field(Easing.SINE_OUT, label="Ease", description="Easing of the flash out")
    grin_delay:         Field[float]  = Field(1.0, min=0.0, max=5.0, step=0.1, newline=True, row_label="Grin", label="Delay (s)", description="Shortest hold on the portrait before the grin plays, for the blend (s)")
    grin_fps:           Field[float]  = Field(15.0, min=5.0, max=30.0, step=1.0, label="Fps", description="Grin playback speed (frames per second)")
    grin_made_fps:      Field[float]  = Field(0.0, access=Field.READ, label="Made fps", description="Speed at which ComfyUI made the last grin (frames per second)")
    grin_buffer_wait:   Field[float]  = Field(0.0, access=Field.READ, label="Buffer wait (s)", description="Wait the last grin needed to play without stalling (s)")
    grin_start_wait:    Field[float]  = Field(0.0, access=Field.READ, label="Start wait (s)", description="Wait before the last grin played: the longer of delay and buffer wait (s)")
    timeout:            Field[float]  = Field(5.0, min=0.1, max=6.0, step=0.1, newline=True, label="Timeout (s)", description="The result stays after the grin, before the bloom (s)")
    destroy_in:         Field[float]  = Field(0.4, min=0.1, max=6.0, step=0.1, newline=True, row_label="Bloom", label="Duration (s)", description="The result blooms (s)")
    destroy_in_ease:    Field[Easing] = Field(Easing.QUAD_IN, label="Ease", description="Easing of the bloom")
    destroy_out:        Field[float]  = Field(1.1, min=0.1, max=6.0, step=0.1, newline=True, row_label="Fade", label="Duration (s)", description="The result dissolves into the camera (s)")
    destroy_out_ease:   Field[Easing] = Field(Easing.CUBIC_OUT, label="Ease", description="Easing of the fade")
