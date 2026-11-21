""" Renders pose keypoints as lines into an offscreen buffer """

# Local application imports
from modules.settings import Field, BaseSettings
from modules.board import HasFrames
from modules.gl import Fbo, Texture, clear_color
from modules.pose import Frame, Points2D, BBox
from ....layer_base import LayerBase
from ....data_cache import DataCache
from ..shaders import PosePointLines as shader
from modules.utils import Color


class PoseLineLayerSettings(BaseSettings):
    use_scores: Field[bool]  = Field(True, description="Use confidence scores for line opacity")
    use_bbox:   Field[bool]  = Field(False, description="Transform points to image space using bbox")
    color:      Field[Color] = Field(Color(1.0, 1.0, 1.0), description="Line color")


class PoseLineLayer(LayerBase):
    """A track's pose lines. `stage`, `line_width` and `line_smooth` (pixels) are inputs its owner sets."""

    def __init__(self, track_id: int, board: HasFrames, settings: PoseLineLayerSettings, *, stage: int = 0,
                 line_width: float = 4.0, line_smooth: float = 2.0) -> None:
        self.settings: PoseLineLayerSettings = settings
        self.stage: int = stage
        self.line_width: float = line_width
        self.line_smooth: float = line_smooth
        self._track_id: int = track_id
        self._board: HasFrames = board
        self._fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        self._shader: shader = shader()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def draw(self) -> None:
        if self._fbo.allocated:
            self._fbo.draw()

    def update(self) -> None:
        pose: Frame | None = self._board.get_frame(self.stage, self._track_id)
        self._data_cache.set(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        # Transform points to image space if use_bbox is enabled
        points = pose[Points2D]
        if self.settings.use_bbox:
            points = points.to_image(pose[BBox].to_rect())

        line_width: float = 1.0 / self._fbo.height * self.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.line_smooth

        self._fbo.begin()
        clear_color()
        self._shader.use(points, line_width=line_width, line_smooth=line_smooth, color=self.settings.color.to_tuple(), use_scores=self.settings.use_scores)
        self._fbo.end()
