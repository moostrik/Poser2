import time

from modules.gl import Fbo, Image, Style, Texture
from modules.render.inputs import Faces
from modules.render.settings import RenderSettings
from modules.render.shaders import FaceColor

from .LayerBase import LayerBase
from .MarkDrawer import MarkDrawer
from .face_marks import Mark, SubjectMark


class CameraLayer(LayerBase):
    """The live camera in black and white, with the chosen face marked: its box turning to colour as
    its charge fills, an outline (the active colour while it qualifies, grey while the charge
    drains), its keypoints and progress bars. Keeps its last frame until render() is called again
    (the capture freeze)."""

    def __init__(self, camera: Image, faces: Faces, settings: RenderSettings) -> None:
        self._camera: Image = camera
        self._faces: Faces = faces
        self.settings: RenderSettings = settings
        self._subject: SubjectMark = SubjectMark()
        self._fbo: Fbo = Fbo()
        self._face_color: FaceColor = FaceColor()
        self._drawer: MarkDrawer = MarkDrawer()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._fbo.allocate(width, height, internal_format)
        self._face_color.allocate()
        self._drawer.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._face_color.deallocate()
        self._drawer.deallocate()

    def update(self) -> None:
        self._face_color.reload()
        self._drawer.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def render(self, charge: float, marks: float = 1.0) -> None:
        """Draw the newest camera frame in black and white; `charge` (0..1) is how far the chosen
        face is toward the flash: the colour inside its box, and its progress bars. `marks` is the
        opacity of the outline, dots and bars (the flash fades them out before the capture)."""
        if not self._camera.allocated:
            return
        r, g, b, a = self.settings.face_color
        mark: Mark | None = self._subject.update(self._faces.get(), time.time(), charge, (r, g, b, a))
        self._fbo.begin()
        if mark is None:
            self._face_color.use(self._camera, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            self._face_color.use(self._camera, mark.x, mark.y, mark.w, mark.h, charge)
            if marks > 0.0:
                Style.set_blend_mode(Style.BlendMode.ALPHA)
                self._drawer.draw(mark, self._width, self._height, marks)
                self._drawer.draw_bar(mark, charge, self._width, self._height, marks)
                Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.end()
