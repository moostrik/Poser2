""" Renders pose keypoints as dots into an offscreen buffer """

# Standard library imports
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.DataHub import DataHub, Stage
from modules.gl import Fbo, Texture, clear_color
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import PosePointDots as shader
from modules.utils.PointsAndRects import Rect

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class PoseDotConfig(ConfigBase):
    stage: Stage = config_field(Stage.LERP, description="Pipeline stage for pose data", fixed=True)
    dot_size: float = config_field(4.0, min=1.0, max=20.0, description="Dot size in pixels")
    dot_smooth: float = config_field(2.0, min=0.0, max=10.0, description="Dot smoothing/antialiasing width")


class PoseDotLayer(LayerBase):
    def __init__(self, track_id: int, data: DataHub, color: tuple[float, float, float, float] | None = None, config: PoseDotConfig | None = None) -> None:
        self._config: PoseDotConfig = config or PoseDotConfig()
        self._track_id: int = track_id
        self._data_hub: DataHub = data
        self._fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

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

    def update(self) -> None:
        pose: Frame | None = self._data_hub.get_pose(self._config.stage, self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        dot_size: float = 1.0 / self._fbo.height * self._config.dot_size * 2
        dot_smooth: float = 1.0 / self._fbo.height * self._config.dot_smooth * 2

        self._fbo.begin()
        clear_color()
        self._shader.use(pose.points, dot_size=dot_size, dot_smooth=dot_smooth)
        self._fbo.end()