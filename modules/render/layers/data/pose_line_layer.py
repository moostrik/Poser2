""" Renders pose keypoints as lines into an offscreen buffer """

# Local application imports
from modules.settings import Field, BaseSettings
from modules.board import HasFrames
from modules.gl import Fbo, Texture, Blit, clear_color
from modules.pose import Frame, Points2D, BBox
from ..layer_base import LayerBase, DataCache
from ...shaders import PosePointLines as shader
from modules.utils import Color, Rect


class PoseLineLayerSettings(BaseSettings):
    stage:      Field[int] = Field(0, access=Field.INIT, description="Pipeline stage for pose data")
    line_width: Field[float] = Field(4.0, min=0.5, max=20.0, description="Line width in pixels")
    line_smooth:Field[float] = Field(2.0, min=0.0, max=10.0, description="Line smoothing/antialiasing width")
    use_scores: Field[bool]  = Field(True, description="Use confidence scores for line opacity")
    use_bbox:   Field[bool]  = Field(False, description="Transform points to image space using bbox")
    color:      Field[Color] = Field(Color(1.0, 1.0, 1.0), description="Line color")


class PoseLineLayer(LayerBase):

    def __init__(self, track_id: int, board: HasFrames, settings: PoseLineLayerSettings | None = None) -> None:
        self.settings: PoseLineLayerSettings = settings or PoseLineLayerSettings()
        self._track_id: int = track_id
        self._board: HasFrames = board
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
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._track_id)
        self._data_cache.set(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        # Transform points to image space if use_bbox is enabled
        points = pose[Points2D]
        if self.settings.use_bbox:
            points = PoseLineLayer._transform_to_image_space(points, pose[BBox].to_rect())

        line_width: float = 1.0 / self._fbo.height * self.settings.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.settings.line_smooth

        self._fbo.begin()
        clear_color()
        self._shader.use(points, line_width=line_width, line_smooth=line_smooth, color=self.settings.color.to_tuple(), use_scores=self.settings.use_scores)
        self._fbo.end()

    @staticmethod
    def _transform_to_image_space(points: Points2D, bbox: Rect) -> Points2D:
        """Transform pose points from bbox-relative [0,1] to image space."""
        x_bbox, y_bbox = points.get_xy_arrays()
        x_image = x_bbox * bbox.width + bbox.x
        y_image = y_bbox * bbox.height + bbox.y
        return Points2D.from_xy_arrays(x_image, y_image, points.scores)