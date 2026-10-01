import time

from modules.face.FaceFinder import FaceFrame
from modules.gl import Fbo, Image, Style, Text, Texture
from modules.render.inputs import Faces
from modules.render.settings import FaceFill, RenderSettings
from modules.render.shaders import FaceColor

from .LayerBase import LayerBase
from .MarkDrawer import MarkDrawer
from .face_marks import FACES_STALE, LABEL_BG, REJECTED_COLOR, Mark, SubjectMark, build_marks

FONT_SIZE: int = 18
LABEL_PAD_PX: float = 4.0


class CameraLayer(LayerBase):
    """The live camera in black and white, with the chosen face's box turning to colour as its
    charge fills. With settings.face_marks every detected face is marked: the chosen one with an
    outline (from grey to the active colour with the charge) and grey keypoints, the others in
    white, a rejected one in grey with its reason. Keeps its last frame until render() is called
    again (the capture freeze)."""

    def __init__(self, camera: Image, faces: Faces, settings: RenderSettings) -> None:
        self._camera: Image = camera
        self._faces: Faces = faces
        self.settings: RenderSettings = settings
        self._subject: SubjectMark = SubjectMark()
        self._fbo: Fbo = Fbo()
        self._face_color: FaceColor = FaceColor()
        self._drawer: MarkDrawer = MarkDrawer()
        self._text: Text = Text()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._fbo.allocate(width, height, internal_format)
        self._face_color.allocate()
        self._drawer.allocate()
        self._text.allocate(font_size=FONT_SIZE)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._face_color.deallocate()
        self._drawer.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        self._face_color.reload()
        self._drawer.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def render(self, charge: float, marks: float = 1.0) -> None:
        """Draw the newest camera frame in black and white; `charge` (0..1) is how far the chosen
        face is toward the flash: the colour inside its box (fading in or rising, settings.face_fill).
        `marks` is the opacity of the face marks (the flash fades them out before the capture)."""
        if not self._camera.allocated:
            return
        r, g, b, a = self.settings.face_color
        now: float = time.time()
        frame: FaceFrame | None = self._faces.get()
        if frame is not None and now - frame.time > FACES_STALE:
            frame = None
        subject: Mark | None = self._subject.update(frame, now, charge, (r, g, b, a))
        self._fbo.begin()
        if subject is None:
            self._face_color.use(self._camera, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            rise: bool = self.settings.face_fill is FaceFill.RISE
            self._face_color.use(self._camera, subject.x, subject.y, subject.w, subject.h, charge, rise)
        if self.settings.face_marks and marks > 0.0:
            face_marks: list[Mark] = build_marks(frame, (r, g, b, a), charge) if frame is not None else []
            if subject is not None and (frame is None or frame.subject is None):
                face_marks.append(subject)      # the face left: its mark stays while the charge drains
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            for mark in face_marks:
                self._drawer.draw(mark, self._width, self._height, marks)
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
