""" Renders pose keypoints as dots into an offscreen buffer """

# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl import Fbo, Texture, clear_color
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import PosePointDots as shader
from modules.utils.PointsAndRects import Rect

from modules.utils.HotReloadMethods import HotReloadMethods


class PoseDotLayer(LayerBase):
    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes,
                 dot_size: float = 4.0, dot_smooth: float = 2.0,
                 color: tuple[float, float, float, float] | None = None) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data
        self._fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        self.data_type: PoseDataHubTypes = data_type
        self.dot_size: float = dot_size
        self.dot_smooth: float = dot_smooth
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
        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        dot_size: float = 1.0 / self._fbo.height * self.dot_size * 2
        dot_smooth: float = 1.0 / self._fbo.height * self.dot_smooth * 2

        self._fbo.begin()
        clear_color()
        self._shader.use(pose.points, dot_size=dot_size, dot_smooth=dot_smooth)
        self._fbo.end()