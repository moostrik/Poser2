# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.shaders import DrawRectangleOutline


class BBoxCamRenderer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, data_type: PoseDataHubTypes, line_width: int = 2,
                 bbox_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._cam_bbox_rects: list[Rect] = []
        self._width: int = 0
        self._height: int = 0

        self.data_type: PoseDataHubTypes = data_type
        self.line_width: int = line_width  # Line width in pixels
        self.bbox_color: tuple[float, float, float, float] = bbox_color

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
        for bbox_rect in self._cam_bbox_rects:
            # Convert pixel line width to normalized coordinates relative to the rectangle
            rect_w_pixels = bbox_rect.width * self._width
            rect_h_pixels = bbox_rect.height * self._height
            line_width_x = self.line_width / rect_w_pixels
            line_width_y = self.line_width / rect_h_pixels

            self._shader.use(
                bbox_rect.x, bbox_rect.y, bbox_rect.width, bbox_rect.height,
                *self.bbox_color,
                line_width_x, line_width_y
            )

    def update(self) -> None:
        cam_poses: set[Frame] = self._data.get_items_for_cam(DataHubType(self.data_type), self._cam_id)
        self._cam_bbox_rects = []
        for pose in cam_poses:
            self._cam_bbox_rects.append(pose.bbox.to_rect())
