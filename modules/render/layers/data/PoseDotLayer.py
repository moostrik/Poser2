""" Renders pose keypoints as dots into an offscreen buffer """

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.settings import Field, BaseSettings
from modules.board import HasFrames
from modules.gl import Fbo, Texture, clear_color
from modules.pose.frame import Frame
from modules.pose.features import Points2D
from ..LayerBase import LayerBase, DataCache
from ...shaders import PosePointDots as shader


class PoseDotSettings(BaseSettings):
    stage:      Field[int] = Field(0, access=Field.INIT, description="Pipeline stage for pose data")
    dot_size:   Field[float] = Field(4.0, min=1.0, max=20.0, description="Dot size in pixels")
    dot_smooth: Field[float] = Field(2.0, min=0.0, max=10.0, description="Dot smoothing/antialiasing width")


class PoseDotLayer(LayerBase):
    def __init__(self, track_id: int, board: HasFrames, color: tuple[float, float, float, float] | None = None, config: PoseDotSettings | None = None) -> None:
        self._config: PoseDotSettings = config or PoseDotSettings()
        self._track_id: int = track_id
        self._board: HasFrames = board
        self._fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        self.color: tuple[float, float, float, float] | None = color

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

    def update(self) -> None:
        pose: Frame | None = self._board.get_frame(self._config.stage, self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        dot_size: float = 1.0 / self._fbo.height * self._config.dot_size * 2
        dot_smooth: float = 1.0 / self._fbo.height * self._config.dot_smooth * 2

        self._fbo.begin()
        clear_color()
        self._shader.use(pose[Points2D], dot_size=dot_size, dot_smooth=dot_smooth, color=self.color)
        self._fbo.end()