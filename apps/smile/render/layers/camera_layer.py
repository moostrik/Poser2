import math
import time

from modules.face import FaceFrame
from modules.gl import Fbo, Image, Style, SwapFbo, Text, Texture, shaders
from ..inputs import Faces
from ..settings import FaceEffect, FaceFill, RenderSettings
from ..shaders import FaceColor

from .layer_base import LayerBase
from .mark_drawer import MarkDrawer
from .face_marks import BOX_PAD_PX, FACES_STALE, LABEL_BG, REJECTED_COLOR, Mark, SubjectMark, build_marks, face_fields
from .passes import blur, copy

FONT_SIZE: int = 18
LABEL_PAD_PX: float = 4.0
BLUR_RADIUS: float = 2.0    # of the blur effect at amount 1 (1 is about 20 pixels each way)
NO_RECT: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

# the shader's numbers for the settings' choices: their order in the enum
EFFECTS: list[FaceEffect] = list(FaceEffect)
FILLS: list[FaceFill] = list(FaceFill)


def warmup_block(width: int, height: int, step: int) -> int:
    """The block size of the warm-up pixels at `step`: the largest block that tiles a width x height
    image whole (their gcd, 256 at 1024 x 1280) halved per step, or when that is no divisor the
    common divisor nearest to it in ratio, so the blocks always come out whole."""
    largest: int = math.gcd(width, height)
    target: float = largest / 2 ** step
    divisors = (d for d in range(1, largest + 1) if largest % d == 0)
    return min(divisors, key=lambda d: abs(math.log(d / target)))


class CameraLayer(LayerBase):
    """The live camera in black and white. Every detected face's box (the portrait crop's) shows how
    far the face is from good (settings.face_effect), and the chosen face's turns to color as its
    charge fills (settings.face_fill). With settings.face_marks every detected face is marked: the chosen one with an
    outline (from grey to the active color with the charge) and grey keypoints, the others in
    white, a rejected one in grey with its reason. A box that reaches past the image is drawn up to a
    padding inside it, so the chosen face's outline and fill stay whole. Keeps its last frame until
    render() is called again (the capture freeze).
    While the generator is not ready (warmup is its step) the whole image is in pixels instead,
    blocks that tile the image whole and halve with every step of the warm-up, and no face is shown."""

    def __init__(self, camera: Image, faces: Faces, settings: RenderSettings) -> None:
        self._camera: Image = camera
        self._faces: Faces = faces
        self.settings: RenderSettings = settings
        self.warmup: int | None = None  # the step the generator's warm-up reached; None is ready (any thread sets it)
        self._subject: SubjectMark = SubjectMark()
        self._fbo: Fbo = Fbo()
        self._face_color: FaceColor = FaceColor()
        self._drawer: MarkDrawer = MarkDrawer()
        self._text: Text = Text()
        # a blurred camera frame, only made for the blur effect
        self._blurred: SwapFbo = SwapFbo()
        self._blur_h: shaders.BlurH = shaders.BlurH()
        self._blur_v: shaders.BlurV = shaders.BlurV()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._fbo.allocate(width, height, internal_format)
        self._face_color.allocate()
        self._drawer.allocate()
        self._text.allocate(font_size=FONT_SIZE)
        self._blurred.allocate(width, height, internal_format)
        self._blur_h.allocate()
        self._blur_v.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._face_color.deallocate()
        self._drawer.deallocate()
        self._text.deallocate()
        self._blurred.deallocate()
        self._blur_h.deallocate()
        self._blur_v.deallocate()

    def update(self) -> None:
        self._face_color.reload()
        self._drawer.reload()
        self._blur_h.reload()
        self._blur_v.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def render(self, charge: float, marks: float = 1.0) -> None:
        """Draw the newest camera frame in black and white; `charge` (0..1) is how far the chosen
        face is toward the flash: the color inside its box.
        `marks` is the opacity of the face marks (the flash fades them out before the capture)."""
        if not self._camera.allocated:
            return
        r, g, b, a = self.settings.face_color
        now: float = time.perf_counter()
        frame: FaceFrame | None = self._faces.get()
        if frame is not None and now - frame.timestamp > FACES_STALE:
            frame = None
        # not ready: the image in pixels that halve per step, and no face (nobody can start)
        warmup: int | None = self.warmup
        mosaic: float = 1.0
        if warmup is not None:
            mosaic = float(warmup_block(self._width, self._height, max(warmup, 0)))
            frame = None
            charge = 0.0
        pad = (BOX_PAD_PX / self._width, BOX_PAD_PX / self._height)
        subject: Mark | None = self._subject.update(frame, now, charge, (r, g, b, a), pad)
        fields = face_fields(frame, pad) if frame is not None else []
        if warmup is not None:
            subject = None      # a mark that was still draining
        rect = (subject.x, subject.y, subject.w, subject.h, subject.far) if subject is not None else NO_RECT
        fill: FaceFill = self.settings.face_fill
        effect: FaceEffect = self.settings.face_effect
        amount: float = self.settings.face_amount
        blurred: Texture | None = None
        if effect is FaceEffect.BLUR and (fields or subject is not None):
            copy(self._camera, self._blurred)
            blur(self._blurred, self._blur_h, self._blur_v, amount * BLUR_RADIUS)
            blurred = self._blurred
        self._fbo.begin()
        self._face_color.use(self._camera, self._width, self._height, fields, EFFECTS.index(effect), amount,
                             rect, charge, FILLS.index(fill), (r, g, b, a), blurred, mosaic)
        if self.settings.face_marks and marks > 0.0 and warmup is None:
            face_marks: list[Mark] = build_marks(frame, (r, g, b, a), charge, pad) if frame is not None else []
            if subject is not None and (frame is None or frame.subject is None):
                face_marks.append(subject)      # the face left: its mark stays while the charge drains
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            for mark in face_marks:
                # the trace is the chosen face's outline
                self._drawer.draw(mark, self._width, self._height, marks, outline=not (mark.subject and fill is FaceFill.TRACE))
            for mark in face_marks:
                if mark.label:
                    self._draw_label(mark, marks)
            Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.end()

    def _draw_label(self, mark: Mark, alpha: float) -> None:
        """Above the box, inside the frame; below its top edge when there is no room above."""
        text_width, text_height = self._text.measure_text(mark.label)
        x: float = mark.x * self._width
        y: float = mark.y * self._height - text_height - LABEL_PAD_PX * 2
        if y < 0.0:
            y = mark.y * self._height + LABEL_PAD_PX
        x = max(LABEL_PAD_PX, min(x, self._width - text_width - LABEL_PAD_PX))
        y = max(LABEL_PAD_PX, min(y, self._height - text_height - LABEL_PAD_PX))
        r, g, b, a = REJECTED_COLOR
        self._text.draw_box_text(x, y, mark.label, (r, g, b, a * alpha), (*LABEL_BG[:3], LABEL_BG[3] * alpha), self._width, self._height)
