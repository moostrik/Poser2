# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.gl.LayerBase import LayerBase, Rect


class CamBBoxRenderer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, data_type: PoseDataHubTypes, line_width: int = 2,
                 bbox_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        # for now make sure the pose meshes are for the correct data type
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._cam_bbox_rects: list[Rect] = []

        self.data_type: PoseDataHubTypes = data_type
        self.line_width: int = int(line_width)
        self.bbox_color: tuple[float, float, float, float] = bbox_color

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
        cam_poses: set[Frame] = self._data.get_items_for_cam(DataHubType(self.data_type), self._cam_id)
        self._cam_bbox_rects = []
        for pose in cam_poses:
            self._cam_bbox_rects.append(pose.bbox.to_rect())
