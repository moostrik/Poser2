""" Renders pose keypoints as lines into an offscreen buffer """

# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl import Fbo, Texture, Blit, viewport_rect
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.shaders import PosePointLines as shader

from modules.utils.HotReloadMethods import HotReloadMethods


class PoseLineLayer(LayerBase):

    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes,
                 line_width: float = 4.0, line_smooth: float = 2.0, use_scores: bool = True, use_bbox: bool = False,
                 color: tuple[float, float, float, float] | None = None) -> None:
        self._track_id: int = track_id
        self._data: DataHub = data
        self._fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type
        self.line_width: float = line_width
        self.line_smooth: float = line_smooth
        self.use_scores: bool = use_scores
        self.use_bbox: bool = use_bbox
        self.color: tuple[float, float, float, float] | None = color

        self._shader: shader = shader()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        if self.use_bbox and self._p_pose is not None:
            box_rect: Rect = self._p_pose.bbox.to_rect()
            rect = box_rect.affine_transform(rect)

        if self._fbo.allocated:
            viewport_rect(*rect)
            Blit.use(self._fbo.texture)

    def update(self) -> None:

        pose: Frame | None = self._data.get_item(DataHubType(self.data_type), self._track_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        line_width: float = 1.0 / self._fbo.height * self.line_width * 2
        line_smooth: float = 1.0 / self._fbo.height * self.line_smooth * 2

        self._fbo.begin()
        self._shader.use(pose.points, line_width=line_width, line_smooth=line_smooth, color=self.color, use_scores=self.use_scores)
        self._fbo.end()