# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataType, PoseDataTypes
from modules.pose.Pose import Pose
from modules.render.renderers.RendererBase import RendererBase
from modules.utils.PointsAndRects import Rect


class CamBBoxRenderer(RendererBase):
    def __init__(self, cam_id: int, data: DataHub, data_type: PoseDataTypes, line_width: int = 2,
                 bbox_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        # for now make sure the pose meshes are for the correct data type
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._cam_bbox_rects: list[Rect] = []

        self.data_type: PoseDataTypes = data_type
        self.line_width: int = int(line_width)
        self.bbox_color: tuple[float, float, float, float] = bbox_color

    def allocate(self) -> None:
        pass

    def deallocate(self) -> None:
        pass

    def draw(self, rect: Rect) -> None:
        glLineWidth(self.line_width)
        glColor4f(*self.bbox_color)

        for bbox_rect in self._cam_bbox_rects:

            draw_rect: Rect = bbox_rect.affine_transform(rect)

            glBegin(GL_LINE_LOOP)
            glVertex2f(draw_rect.x, draw_rect.y)  # Bottom left
            glVertex2f(draw_rect.x + draw_rect.width, draw_rect.y)  # Bottom right
            glVertex2f(draw_rect.x + draw_rect.width, draw_rect.y + draw_rect.height)  # Top right
            glVertex2f(draw_rect.x, draw_rect.y + draw_rect.height)  # Top left
            glEnd()

        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

    def update(self) -> None:
        cam_poses: set[Pose] = self._data.get_items_for_cam(DataType(self.data_type), self._cam_id)
        self._cam_bbox_rects = []
        for pose in cam_poses:
            self._cam_bbox_rects.append(pose.bbox.to_rect())
