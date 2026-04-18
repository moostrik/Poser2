# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.settings import Field, BaseSettings
from modules.board import HasFrames
from modules.pose.frame import Frame
from modules.pose.features import BBox
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.shaders import DrawRectangleOutline
from modules.utils import Color


class BBoxRendererSettings(BaseSettings):
    stage:      Field[int] = Field(3, description="Pipeline stage for pose data")
    line_width: Field[float] = Field(2.0, min=1.0, max=10.0, description="Bounding box line width in pixels")
    color:      Field[Color] = Field(Color(1.0, 1.0, 1.0), description="Bounding box color")


class BBoxRenderer(LayerBase):
    def __init__(self, cam_id: int, board: HasFrames, settings: BBoxRendererSettings | None = None) -> None:
        self.settings: BBoxRendererSettings = settings or BBoxRendererSettings()
        self._board: HasFrames = board
        self._cam_id: int = cam_id
        self._cam_bbox_rects: list[Rect] = []
        self._width: int = 0
        self._height: int = 0

        self._shader: DrawRectangleOutline = DrawRectangleOutline()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._shader.allocate()
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height

    def deallocate(self) -> None:
        self._shader.deallocate()

    def draw(self) -> None:
        color = self.settings.color.to_tuple()
        for bbox_rect in self._cam_bbox_rects:
            # Convert pixel line width to normalized coordinates relative to the rectangle
            rect_w_pixels = bbox_rect.width * self._width
            rect_h_pixels = bbox_rect.height * self._height
            line_width_x = self.settings.line_width / rect_w_pixels
            line_width_y = self.settings.line_width / rect_h_pixels

            self._shader.use(
                bbox_rect.x, bbox_rect.y, bbox_rect.width, bbox_rect.height,
                *color,
                line_width_x, line_width_y
            )

    def update(self) -> None:
        cam_poses = {p for p in self._board.get_frames(self.settings.stage).values() if p.cam_id == self._cam_id}
        self._cam_bbox_rects = []
        for pose in cam_poses:
            self._cam_bbox_rects.append(pose[BBox].to_rect())
