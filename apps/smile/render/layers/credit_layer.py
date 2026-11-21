from enum import IntEnum

from OpenGL.GL import glViewport

from ...credit import CreditLines
from modules.gl import Fbo, Texture, shaders
from ...sequence import SequenceType
from ..settings import RenderSettings

from modules.render import LayerBase
from .caption_renderer import CaptionRenderer
from .passes import copy
from .qr_code_layer import QrCodeLayer

FONT_SIZE: int = 96
MARGIN_PX: int = 40             # from the image's edges
OUTLINE_PX: int = 5             # the black edge round the letters, as on a meme
FADE_S: float = 0.5             # the caption fades in and out over this time

# the caption is shown with the visitor's portrait: from the first generated image, while the grin plays and
# while the result holds after it
SHOWN: frozenset[SequenceType] = frozenset({SequenceType.DIFFUSION, SequenceType.DIFF_LOOP, SequenceType.TIMEOUT})
# and, with `credit_camera`, over the camera while the installation waits for a visitor
CAMERA: frozenset[SequenceType] = frozenset({SequenceType.CAMERA_SHOW, SequenceType.CAMERA_DETECT})


class CreditView(IntEnum):
    """Who a pass of the credit is for, each with its own texture."""
    ROOM = 0        # the main screen: the QR code, the title and the line
    CELL = 1        # the online cell: the title and the line
    PHONE = 2       # the phone: the line, its page has the title above the picture


def caption_shown(sequence_type: SequenceType, camera: bool) -> bool:
    """Whether the caption is on in this sequence state (`camera`: the camera view has it too)."""
    return sequence_type in SHOWN or (camera and sequence_type in CAMERA)


def new_line_due(sequence_type: SequenceType, previous: SequenceType) -> bool:
    """A new line is drawn at the capture (the visitor's own), and when the camera view begins after a
    visitor (the waiting screen's), not when a face comes and goes within the camera view."""
    if sequence_type == previous:
        return False
    if sequence_type == SequenceType.CAPTURE:
        return True
    return sequence_type == SequenceType.CAMERA_SHOW and previous not in CAMERA


def fade_step(alpha: float, shown: bool, dt: float) -> float:
    """The caption's opacity one frame of `dt` seconds later: towards 1 while shown, towards 0 after."""
    step: float = dt / FADE_S
    return min(1.0, alpha + step) if shown else max(0.0, alpha - step)


class CreditLayer(LayerBase):
    """A QR code to the project page in the bottom right corner, always, and the machine's joke as a meme
    caption over the top of the image (white with a black edge, centered): the title SMILE(ai), and under it
    the visitor's line, drawn from the lines file at every capture (`new_visitor`), at the opacity `render`
    gets. People film the result, so the title is in their videos. Each `CreditView` is a pass of its own."""

    def __init__(self, settings: RenderSettings) -> None:
        self.settings: RenderSettings = settings
        self._lines: CreditLines = CreditLines(settings.credit_lines)
        self._base: Fbo = Fbo()             # the image with the QR code
        self._card: Fbo = Fbo()             # and with the caption
        self._fbo: Fbo = Fbo()              # the two mixed by the caption's opacity
        self._cell: Fbo = Fbo()             # the same without the QR code: the online cell's copy
        self._clean: Fbo = Fbo()            # and without the title: the phone's copy
        self._targets: dict[CreditView, Fbo] = {CreditView.ROOM: self._fbo, CreditView.CELL: self._cell, CreditView.PHONE: self._clean}
        self._mix: shaders.Mix = shaders.Mix()
        self._caption: CaptionRenderer = CaptionRenderer(FONT_SIZE, OUTLINE_PX, MARGIN_PX)
        self._qr: QrCodeLayer = QrCodeLayer()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._base.allocate(width, height, internal_format)
        self._card.allocate(width, height, internal_format)
        self._fbo.allocate(width, height, internal_format)
        self._cell.allocate(width, height, internal_format)
        self._clean.allocate(width, height, internal_format)
        self._mix.allocate()
        self._caption.allocate()

    def deallocate(self) -> None:
        self._base.deallocate()
        self._card.deallocate()
        self._fbo.deallocate()
        self._cell.deallocate()
        self._clean.deallocate()
        self._mix.deallocate()
        self._caption.deallocate()
        self._qr.deallocate()

    def update(self) -> None:
        self._caption.update()
        self._qr.set(str(self.settings.credit_url), int(self.settings.credit_size))

    @property
    def texture(self) -> Texture:
        return self._fbo

    @property
    def cell(self) -> Texture:
        """The output of the last `render` of `CreditView.CELL` (SMILE Online: the online cell's copy)."""
        return self._cell

    @property
    def clean(self) -> Texture:
        """The output of the last `render` of `CreditView.PHONE` (SMILE Online: the phone's copy)."""
        return self._clean

    def render(self, source: Texture, alpha: float, view: CreditView = CreditView.ROOM) -> None:
        """The credit over source for `view`, the caption at opacity `alpha`: the main screen's into `texture`,
        the online cell's without the QR code into `cell`, the phone's also without the title into `clean`
        (SMILE Online: the code is for the room, and the phone's page has the title above the picture)."""
        if not source.allocated:
            return
        target: Fbo = self._targets[view]
        self._base.begin()
        source.draw()
        qr: int = self._qr.size if view == CreditView.ROOM else 0
        if qr:
            # the viewport's origin is the bottom left
            glViewport(self._width - MARGIN_PX - qr, MARGIN_PX, qr, qr)
            self._qr.texture.draw()
            glViewport(0, 0, self._width, self._height)
        self._base.end()
        if alpha <= 0.0:
            copy(self._base, target)
            return
        self._card.begin()
        self._base.draw()
        self._caption.draw(str(self.settings.credit_line), self._width, self._height, title=view != CreditView.PHONE)
        self._card.end()
        target.begin()
        self._mix.use(self._base, self._card, alpha)
        target.end()

    def new_visitor(self) -> None:
        """Draw the line under the title for the visitor that is captured now."""
        self.settings.credit_line = self._lines.pick()
