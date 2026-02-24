""" Renders pose keypoints as lines into an offscreen buffer """

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.settings import Setting, BaseSettings
from modules.DataHub import DataHub, Stage
from modules.gl import Fbo, Texture, Blit, clear_color
from modules.pose.Frame import Frame
from modules.pose.features.Points2D import Points2D
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import PosePointLines as shader
from modules.utils import Color


class PoseLineSettings(BaseSettings):
    stage:      Setting[Stage] = Setting(Stage.LERP, access=Setting.INIT, description="Pipeline stage for pose data")
    line_width: Setting[float] = Setting(4.0, min=0.5, max=20.0, description="Line width in pixels")
    line_smooth:Setting[float] = Setting(2.0, min=0.0, max=10.0, description="Line smoothing/antialiasing width")
    use_scores: Setting[bool]  = Setting(True, description="Use confidence scores for line opacity")
    use_bbox:   Setting[bool]  = Setting(False, description="Transform points to image space using bbox")
    color:      Setting[Color] = Setting(Color(1.0, 1.0, 1.0), description="Line color")


class PoseLineLayer(LayerBase):

    def __init__(self, track_id: int, data: DataHub, config: PoseLineSettings | None = None) -> None:
        self._config: PoseLineSettings = config or PoseLineSettings()
        self._track_id: int = track_id
        self._data_hub: DataHub = data
        self._fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        self._shader: shader = shader()

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def draw(self) -> None:
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)

    def update(self) -> None:
        pose: Frame | None = self._data_hub.get_pose(self._config.stage, self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        # Transform points to image space if use_bbox is enabled
        points = pose.points
        if self._config.use_bbox:
            points = PoseLineLayer._transform_to_image_space(points, pose.bbox.to_rect())

        line_width: float = 1.0 / self._fbo.height * self._config.line_width
        line_smooth: float = 1.0 / self._fbo.height * self._config.line_smooth

        self._fbo.begin()
        clear_color()
        self._shader.use(points, line_width=line_width, line_smooth=line_smooth, color=self._config.color.to_tuple(), use_scores=self._config.use_scores)
        self._fbo.end()

    @staticmethod
    def _transform_to_image_space(points: Points2D, bbox: Rect) -> Points2D:
        """Transform pose points from bbox-relative [0,1] to image space."""
        x_bbox, y_bbox = points.get_xy_arrays()
        x_image = x_bbox * bbox.width + bbox.x
        y_image = y_bbox * bbox.height + bbox.y
        return Points2D.from_xy_arrays(x_image, y_image, points.scores)