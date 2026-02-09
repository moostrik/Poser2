""" Draws camera image roi for a given pose frame """

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
from modules.render.layers.source import ImageSourceLayer
from modules.render.shaders import DrawRoi

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class CropConfig(ConfigBase):
    stage: Stage = config_field(Stage.LERP, description="Pipeline stage for pose data", fixed=True)


class CropLayer(LayerBase):
    def __init__(self, track_id: int, data_hub: DataHub, cam_texture: Texture, config: CropConfig | None = None) -> None:
        self._config: CropConfig = config or CropConfig()
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._cam_texture: Texture = cam_texture
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        # Add DrawRoi shader
        self._roi_shader = DrawRoi()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._roi_shader.deallocate()

    def update(self) -> None:
        pose: Frame | None = self._data_hub.get_pose(self._config.stage, self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        pose_rect: Rect = pose.bbox.to_rect()
        # convert to texture space
        pose_rect = pose_rect.flip_y(0.5)

        self._fbo.begin()
        clear_color()
        self._roi_shader.use(self._cam_texture, pose_rect)
        self._fbo.end()