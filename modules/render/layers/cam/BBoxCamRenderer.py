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

        self.data_type: PoseDataHubTypes = data_type
        self.line_width: float = 0.05  # Fixed normalized line width for visibility
        self.bbox_color: tuple[float, float, float, float] = bbox_color

        self._shader: DrawRectangleOutline = DrawRectangleOutline()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._shader.allocate()

    def deallocate(self) -> None:
        self._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        for bbox_rect in self._cam_bbox_rects:
            draw_rect: Rect = bbox_rect.affine_transform(rect)
            self._shader.use(
                draw_rect.x, draw_rect.y, draw_rect.width, draw_rect.height,
                *self.bbox_color,
                self.line_width
            )

    def update(self) -> None:
        cam_poses: set[Frame] = self._data.get_items_for_cam(DataHubType(self.data_type), self._cam_id)
        self._cam_bbox_rects = []
        for pose in cam_poses:
            self._cam_bbox_rects.append(pose.bbox.to_rect())
