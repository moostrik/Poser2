from enum import Enum

from ..credit import LINE as CREDIT_LINE, LINES as CREDIT_LINES, NAME as CREDIT_NAME, URL as CREDIT_URL
from ..credit_scroll import DEFAULT_GAP_PX as CREDIT_SCROLL_GAP, DEFAULT_ROWS as CREDIT_SCROLL_ROWS
from modules.settings import BaseSettings, Field, Group
from modules.utils import Color
from modules.window import WindowManagerSettings, WindowSettings


class WindowsSettings(WindowManagerSettings):
    """The render's windows: window 0 the operator's, windows 1-2 the walls (`sphere.WALLS`)"""
    window_0: Group[WindowSettings]        = Group(WindowSettings)
    window_1: Group[WindowSettings]        = Group(WindowSettings)
    window_2: Group[WindowSettings]        = Group(WindowSettings)


class FaceFill(Enum):
    FADE  = 'Fade'      # color over the whole box
    RISE  = 'Rise'      # color climbs from the bottom
    IRIS  = 'Iris'      # color grows as a circle from the center
    TRACE = 'Trace'     # no color: the outline draws itself around the box
    SCAN  = 'Scan'      # as Rise, with a bright line at the color's edge


class FaceEffect(Enum):
    DARK   = 'Dark'
    PIXELS = 'Pixels'
    BLUR   = 'Blur'
    FOG    = 'Fog'


class RenderSettings(BaseSettings):
    show_camera:   Field[bool]  = Field(False, description="Draw the raw camera image on top; no sequence starts", pinned=True)
    face_marks:    Field[bool]  = Field(False, description="Draw the faces' outlines, dots and reject reasons on the camera")
    face_color:    Field[Color] = Field(Color(0.2, 1.0, 0.2), description="Color of the chosen face's box and dots")
    face_fill:     Field[FaceFill] = Field(FaceFill.FADE, description="How the chosen face's box turns to color as its charge fills")
    face_effect:   Field[FaceEffect] = Field(FaceEffect.DARK, description="How a face's box shows its distance from good")
    face_amount:   Field[float] = Field(0.5, min=0.0, max=1.0, step=0.05, description="Strength of the effect on a face that is far from good")
    freeze:        Field[bool]  = Field(False, description="Hold the current frame")
    trail_active:  Field[bool]  = Field(True, description="Blend new diffusion images over the previous ones")
    trail:         Field[float] = Field(0.995, min=0.98, max=1.0, step=0.001, description="Trail amount")
    loop_trail:    Field[float] = Field(0.5, min=0.0, max=0.99, step=0.01, description="Trail while the grin animation plays (0 = none)")
    fade_blur:     Field[float] = Field(0.9, min=0.0, max=9.0, step=0.1, description="Blur per frame while fading the last image out (pixels)")
    bloom_amount:  Field[float] = Field(0.6, min=0.0, max=1.0, step=0.1, description="Bloom while fading the last image out")
    credit:        Field[bool]  = Field(True, newline=True, row_label="Credit", label="Show", description="QR code in the corner, the caption with the grin and the result")
    credit_name:   Field[str]   = Field(CREDIT_NAME, label="Name", description="The artist's name: scrolling, on the QR cell and the watermark")
    credit_lines:  Field[str]   = Field(CREDIT_LINES, access=Field.INIT, label="Lines file", description="Text file with the lines under SMILE(ai), one per line")
    credit_line:   Field[str]   = Field(CREDIT_LINE, access=Field.READ, label="Line", description="The line under SMILE(ai) for the current visitor")
    credit_url:    Field[str]   = Field(CREDIT_URL, label="URL", description="Page the QR code opens")
    credit_camera: Field[bool]  = Field(False, newline=True, row_label="Caption", label="On camera", description="Also show the caption over the camera view, while waiting for a visitor")
    credit_walls:  Field[bool]  = Field(False, label="On walls", description="Show each clip's caption on the walls (SMILE(ai) top, line bottom)")
    credit_qr_walls: Field[bool] = Field(True, label="QR on walls", description="One cell per wall shows the QR code and the name for one play")
    credit_size:   Field[int]   = Field(200, min=120, max=480, step=10, label="QR size (px)", description="Size of the QR code (px)")
    credit_scroll: Field[bool]  = Field(False, newline=True, row_label="Scroll", label="On walls", description="Each word of the name circles its own ring of the walls, like a ticker")
    credit_scroll_idle: Field[float] = Field(120.0, min=0.0, max=600.0, step=10.0, label="Idle (s)", description="Seconds without a qualifying face before the name goes round")
    credit_scroll_rows: Field[str] = Field(CREDIT_SCROLL_ROWS, label="Rows", description="Rings of the words, 0 (top) to 4, first word on the first")
    credit_scroll_speed_1: Field[float] = Field(1.0, min=0.1, max=5.0, step=0.1, label="Speed 1 (screens/s)", description="How fast the first word moves, in screen widths per second")
    credit_scroll_speed_2: Field[float] = Field(0.8, min=0.1, max=5.0, step=0.1, label="Speed 2 (screens/s)", description="How fast the second word moves, in screen widths per second")
    credit_scroll_size: Field[int] = Field(512, min=64, max=640, step=16, label="Size (px)", description="Height of the letters with their edge, of the screen's 640 px, centered")
    credit_scroll_gap: Field[int] = Field(CREDIT_SCROLL_GAP, min=0, max=2048, step=16, label="Gap (px)", description="The wall between two neighboring screens, as pixels the letters cross unseen")
    credit_scroll_reverse: Field[bool] = Field(False, label="Reverse", description="The words go the other way round the sphere")
    credit_scroll_last: Field[str] = Field('', access=Field.READ, label="Going round", description="Which word goes round which row now")
