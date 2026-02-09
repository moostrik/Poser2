""" Composite layer showing track camera crop with RAW, SMOOTH, and LERP pose overlays """

# Standard library imports
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.gl import Fbo, Texture, Blit
from modules.DataHub import DataHub, Stage
from modules.render.layers.LayerBase import LayerBase

from modules.render.layers.cam.CropLayer import CropLayer, CropConfig
from modules.render.layers.source.CropSourceLayer import CropSourceLayer
from modules.render.layers.data.PoseLineLayer import PoseLineLayer, PoseLineConfig

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class PoseCompositorConfig(ConfigBase):
    stage: Stage = config_field(Stage.LERP, description="Pipeline stage for camera crop", fixed=True)
    line_width: float = config_field(3.0, min=0.5, max=20.0, description="Base line width (multiplied per stage)")
    line_smooth: float = config_field(0.0, min=0.0, max=10.0, description="Base line smoothing (multiplied per stage)")
    use_gpu_crop: bool = config_field(False, description="Use GPU crop source (model input) instead of camera crop")


class PoseCompositor(LayerBase):
    def __init__(self, track_id: int, data: DataHub, cam_texture: Texture, track_color: tuple[float, float, float, float], config: PoseCompositorConfig | None = None) -> None:
        self._config: PoseCompositorConfig = config or PoseCompositorConfig()
        self._track_id: int = track_id
        self._fbo: Fbo = Fbo()

        # Camera crop layers (one of these will be used based on config)
        crop_config: CropConfig = CropConfig(stage=self._config.stage)
        self._cam_crop_layer: CropLayer = CropLayer(track_id, data, cam_texture, crop_config)
        self._gpu_crop_layer: CropSourceLayer = CropSourceLayer(track_id, data)

        # Pose line layers - RAW (black, thickest), SMOOTH (white, medium), LERP (color, thin)
        # RAW = 3x thickness, drawn first (back layer)
        pose_raw_config: PoseLineConfig = PoseLineConfig(
            stage=Stage.RAW,
            line_width=self._config.line_width * 3.0,
            line_smooth=self._config.line_smooth * 3.0,
            use_scores=True,
            use_bbox=False
        )
        self._pose_raw_layer: PoseLineLayer = PoseLineLayer(track_id, data, (0.0, 0.0, 0.0, 1.0), pose_raw_config)

        # SMOOTH = 2x thickness, drawn second (middle layer)
        pose_smooth_config: PoseLineConfig = PoseLineConfig(
            stage=Stage.SMOOTH,
            line_width=self._config.line_width * 2.0,
            line_smooth=self._config.line_smooth * 2.0,
            use_scores=True,
            use_bbox=False
        )
        self._pose_smooth_layer: PoseLineLayer = PoseLineLayer(track_id, data, (1.0, 1.0, 1.0, 1.0), pose_smooth_config)

        # LERP = 1x thickness, drawn last (top layer)
        pose_lerp_config: PoseLineConfig = PoseLineConfig(
            stage=Stage.LERP,
            line_width=self._config.line_width,
            line_smooth=self._config.line_smooth,
            use_scores=True,
            use_bbox=False
        )
        self._pose_lerp_layer: PoseLineLayer = PoseLineLayer(track_id, data, track_color, pose_lerp_config)

        self.hot_reload = HotReloadMethods(self.__class__)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._cam_crop_layer.allocate(width, height, internal_format)
        self._gpu_crop_layer.allocate(width, height, internal_format)
        self._pose_raw_layer.allocate(width, height, internal_format)
        self._pose_smooth_layer.allocate(width, height, internal_format)
        self._pose_lerp_layer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._cam_crop_layer.deallocate()
        self._gpu_crop_layer.deallocate()
        self._pose_raw_layer.deallocate()
        self._pose_smooth_layer.deallocate()
        self._pose_lerp_layer.deallocate()

    def update(self) -> None:
        # Update all layers
        crop_layer = self._gpu_crop_layer if self._config.use_gpu_crop else self._cam_crop_layer
        crop_layer.update()
        self._pose_raw_layer.update()
        self._pose_smooth_layer.update()
        self._pose_lerp_layer.update()

        # Composite: crop background + pose overlays (RAW → SMOOTH → LERP)
        self._fbo.begin()
        Blit.use(crop_layer.texture)
        Blit.use(self._pose_raw_layer.texture)
        Blit.use(self._pose_smooth_layer.texture)
        Blit.use(self._pose_lerp_layer.texture)
        self._fbo.end()
