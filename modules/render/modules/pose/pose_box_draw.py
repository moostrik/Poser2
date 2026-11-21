# Local application imports
from modules.board import HasFrames
from modules.pose import BBox
from modules.gl import shaders
from ...layer_base import LayerBase
from ...color_settings import ColorSettings
from modules.utils import Color, Rect


class PoseBoxDraw(LayerBase):
    """The boxes of a camera's tracks, each in its track's color from `color_settings`, or in `color` without them.
    `stage`, `line_width` (pixels) and `color` are inputs its owner sets."""

    def __init__(self, cam_id: int, board: HasFrames, color_settings: ColorSettings | None = None, *, stage: int = 0,
                 line_width: float = 2.0, color: Color = Color(1.0, 1.0, 1.0)) -> None:
        self.stage: int = stage
        self.line_width: float = line_width
        self.color: Color = color
        self._board: HasFrames = board
        self._cam_id: int = cam_id
        self._color_settings: ColorSettings | None = color_settings
        self._cam_bbox_rects: list[tuple[Rect, int]] = []
        self._width: int = 0
        self._height: int = 0

        self._shader: shaders.StrokeRect = shaders.StrokeRect()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._shader.allocate()
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height

    def deallocate(self) -> None:
        self._shader.deallocate()

    def draw(self) -> None:
        track_colors = self._color_settings.track_colors if self._color_settings else None
        for bbox_rect, track_id in self._cam_bbox_rects:
            color: Color = track_colors[track_id % len(track_colors)] if track_colors else self.color
            # Convert pixel line width to normalized coordinates relative to the rectangle
            rect_w_pixels = bbox_rect.width * self._width
            rect_h_pixels = bbox_rect.height * self._height
            line_width_x = self.line_width / rect_w_pixels
            line_width_y = self.line_width / rect_h_pixels

            self._shader.use(bbox_rect, color, line_width_x, line_width_y)

    def update(self) -> None:
        self._cam_bbox_rects = []
        for track_id, pose in self._board.get_frames(self.stage).items():
            if pose.cam_id == self._cam_id:
                self._cam_bbox_rects.append((pose[BBox].to_rect(), track_id))
