""" Composite layer showing track camera crop with RAW, SMOOTH, and LERP pose overlays """

# Third-party imports
from OpenGL.GL import * # type: ignore

from typing import Any

# Local application imports
from modules.settings import Field, BaseSettings
from modules.gl import Fbo, Texture, Blit
from ..LayerBase import LayerBase

from .CropLayer import CropLayer, CropSettings
from ..source import CropSourceLayer
from ..data import PoseLineLayer, PoseLineSettings
from ...color_settings import ColorSettings
from modules.utils import Color


class PoseCompSettings(BaseSettings):
    stage:      Field[int] = Field(0, description="Pipeline stage for camera crop")
    line_width: Field[float] = Field(2.0, min=0.5, max=20.0, description="Base line width (multiplied per stage)")
    line_smooth:Field[float] = Field(0.0, min=0.0, max=10.0, description="Base line smoothing (multiplied per stage)")
    use_gpu_crop:Field[bool] = Field(True, description="Use GPU crop source (model input) instead of camera crop")


class PoseCompositor(LayerBase):
    def __init__(self, track_id: int, board: Any, cam_texture: Texture, settings: PoseCompSettings, color_settings: ColorSettings) -> None:
        self._track_id: int = track_id
        self.settings: PoseCompSettings = settings or PoseCompSettings()
        self._fbo: Fbo = Fbo()

        # Camera crop layers (one of these will be used based on config)
        crop_config: CropSettings = CropSettings(stage=self.settings.stage)
        self._cam_crop_layer: CropLayer = CropLayer(track_id, board, cam_texture, crop_config)
        self._gpu_crop_layer: CropSourceLayer = CropSourceLayer(track_id, board)

        # Pose line layers - RAW (black, thickest), SMOOTH (white, medium), LERP (color, thin)
        # RAW = 3x thickness, drawn first (back layer)
        pose_raw_config: PoseLineSettings = PoseLineSettings(
            stage=1,
            line_width=self.settings.line_width * 3.0,
            line_smooth=self.settings.line_smooth * 3.0,
            use_scores=True,
            use_bbox=False,
            color=Color(0.0, 0.0, 0.0)
        )
        self._pose_raw_layer: PoseLineLayer = PoseLineLayer(track_id, board, pose_raw_config)

        # SMOOTH = 2x thickness, drawn second (middle layer)
        pose_smooth_config: PoseLineSettings = PoseLineSettings(
            stage=2,
            line_width=self.settings.line_width * 2.0,
            line_smooth=self.settings.line_smooth * 2.0,
            use_scores=True,
            use_bbox=False,
            color=Color(1.0, 1.0, 1.0)
        )
        self._pose_smooth_layer: PoseLineLayer = PoseLineLayer(track_id, board, pose_smooth_config)

        # LERP = 1x thickness, drawn last (top layer)
        pose_lerp_config: PoseLineSettings = PoseLineSettings(
            stage=3,
            line_width=self.settings.line_width,
            line_smooth=self.settings.line_smooth,
            use_scores=True,
            use_bbox=False,
            color=color_settings.track_colors[track_id]
        )
        self._pose_lerp_layer: PoseLineLayer = PoseLineLayer(track_id, board, pose_lerp_config)

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
        crop_layer = self._gpu_crop_layer if self.settings.use_gpu_crop else self._cam_crop_layer
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
