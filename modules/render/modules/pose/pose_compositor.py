""" Composite layer showing track camera crop with RAW, SMOOTH, and LERP pose overlays """

from typing import Any

# Local application imports
from modules.settings import Field, BaseSettings
from modules.gl import Fbo, Texture
from ...layer_base import LayerBase

from ..board import BoardImageLayer
from ..inference import CropLayer
from .lines import PoseLineLayer, PoseLineLayerSettings
from ...color_settings import ColorSettings
from modules.utils import Color


class PoseCompositorSettings(BaseSettings):
    stage:      Field[int] = Field(0, description="Pipeline stage for camera crop")
    back_stage: Field[int] = Field(1, description="Pipeline stage of the back line (black, 3x wide)")
    middle_stage:Field[int] = Field(2, description="Pipeline stage of the middle line (white, 2x wide)")
    front_stage:Field[int] = Field(3, description="Pipeline stage of the front line (track color)")
    line_width: Field[float] = Field(2.0, min=0.5, max=20.0, description="Base line width (multiplied per line)")
    line_smooth:Field[float] = Field(0.0, min=0.0, max=10.0, description="Base line smoothing (multiplied per line)")
    use_gpu_crop:Field[bool] = Field(True, description="Use GPU crop source (model input) instead of camera crop")


class PoseCompositor(LayerBase):
    """The crop with the pose lines over it. `gpu_crop` is the app's crop source layer (the model's
    input), shared rather than uploaded a second time; the app updates it before this layer."""

    def __init__(self, track_id: int, board: Any, cam_texture: Texture, gpu_crop: BoardImageLayer, settings: PoseCompositorSettings,
                 color_settings: ColorSettings) -> None:
        self._track_id: int = track_id
        self.settings: PoseCompositorSettings = settings or PoseCompositorSettings()
        self._fbo: Fbo = Fbo()

        # Camera crop layers (one of these will be used based on settings)
        self._cam_crop_layer: CropLayer = CropLayer(track_id, board, cam_texture)
        self._gpu_crop_layer: BoardImageLayer = gpu_crop

        # Pose lines, back to front: black 3x wide, white 2x, the track color 1x. Their stages and widths are
        # inputs, handed on from `settings` each update (`_set_inputs`).
        self._pose_back_layer: PoseLineLayer = PoseLineLayer(track_id, board, PoseLineLayerSettings(
            use_scores=True, use_bbox=False, color=Color(0.0, 0.0, 0.0)))
        self._pose_middle_layer: PoseLineLayer = PoseLineLayer(track_id, board, PoseLineLayerSettings(
            use_scores=True, use_bbox=False, color=Color(1.0, 1.0, 1.0)))
        self._pose_front_layer: PoseLineLayer = PoseLineLayer(track_id, board, PoseLineLayerSettings(
            use_scores=True, use_bbox=False, color=color_settings.track_colors[track_id]))
        self._set_inputs()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._cam_crop_layer.allocate(width, height, internal_format)
        self._pose_back_layer.allocate(width, height, internal_format)
        self._pose_middle_layer.allocate(width, height, internal_format)
        self._pose_front_layer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._cam_crop_layer.deallocate()
        self._pose_back_layer.deallocate()
        self._pose_middle_layer.deallocate()
        self._pose_front_layer.deallocate()

    def _set_inputs(self) -> None:
        """The crop's and the lines' stages and widths, from `settings` as the panel has them now."""
        s: PoseCompositorSettings = self.settings
        self._cam_crop_layer.stage = s.stage
        for layer, stage, scale in ((self._pose_back_layer, s.back_stage, 3.0),
                                    (self._pose_middle_layer, s.middle_stage, 2.0),
                                    (self._pose_front_layer, s.front_stage, 1.0)):
            layer.stage = stage
            layer.line_width = s.line_width * scale
            layer.line_smooth = s.line_smooth * scale

    def update(self) -> None:
        self._set_inputs()
        crop_layer = self._gpu_crop_layer if self.settings.use_gpu_crop else self._cam_crop_layer
        if crop_layer is self._cam_crop_layer:
            crop_layer.update()     # its own; the shared GPU crop is updated by the app
        self._pose_back_layer.update()
        self._pose_middle_layer.update()
        self._pose_front_layer.update()

        # Composite: crop background, then the lines back to front
        self._fbo.begin()
        crop_layer.texture.draw()
        self._pose_back_layer.texture.draw()
        self._pose_middle_layer.texture.draw()
        self._pose_front_layer.texture.draw()
        self._fbo.end()
