
# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports

from modules.DataHub import DataHub, DataType, PoseDataTypes
from modules.pose.Pose import Pose
from modules.render.renderers.RendererBase import RendererBase
from modules.utils.PointsAndRects import Rect


class PoseBBoxRenderer(RendererBase):
    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataTypes, line_width: float = 2.0,
                 bbox_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self._data: DataHub = data
        self._track_id: int = track_id
        self._bbox_rect: Rect | None = None

        self.data_type: PoseDataTypes = data_type
        self.bbox_color: tuple[float, float, float, float] = bbox_color
        self.line_width: float = line_width

    def allocate(self) -> None:
        pass

    def deallocate(self) -> None:
        pass

    def draw(self, rect: Rect) -> None:
        if self._bbox_rect is None:
            return
        glLineWidth(self.line_width)
        glColor4f(*self.bbox_color)

        draw_rect: Rect = self._bbox_rect.affine_transform(rect)

        glBegin(GL_LINE_LOOP)
        glVertex2f(draw_rect.x, draw_rect.y)  # Bottom left
        glVertex2f(draw_rect.x + draw_rect.width, draw_rect.y)  # Bottom right
        glVertex2f(draw_rect.x + draw_rect.width, draw_rect.y + draw_rect.height)  # Top right
        glVertex2f(draw_rect.x, draw_rect.y + draw_rect.height)  # Top left
        glEnd()
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color


    def update(self) -> None:
        pose: Pose | None = self._data.get_item(DataType(self.data_type), self._track_id)

        if pose is None:
            self._bbox_rect = None
        else:
            self._bbox_rect = pose.bbox.to_rect()

