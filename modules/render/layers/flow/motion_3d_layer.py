"""The 3D motion layer: the Lucas–Kanade flow, x/y from the image and z from the depth."""

# Standard library imports
import logging

# Local application imports
from modules.flow import LucasKanadeFlow3D, LucasKanadeFlow3DSettings, DepthView
from modules.utils import HotReloadMethods
from .motion_layer import MotionLayer, MotionLayerSettings, MotionInput, MotionOutput
from .port import Port, TextureKind

logger = logging.getLogger(__name__)


class Motion3DLayerSettings(LucasKanadeFlow3DSettings, MotionLayerSettings):
    """Configuration for Motion3DLayer: the stage's settings and its flow's own, one panel, the stage's first."""


class Motion3DLayer(MotionLayer):
    """The 3D motion layer: `LucasKanadeFlow3D` on the input's image, z from its depth, gated by its mask."""

    def __init__(self, cam_id: int, settings: Motion3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Motion3DLayerSettings = settings
        self._cam_id: int = cam_id
        self._flow: LucasKanadeFlow3D = LucasKanadeFlow3D(settings)
        self._warned_view_span: bool = False

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def output(self) -> MotionOutput:
        textures: dict[str, Port] = {}
        input: MotionInput | None = self._input
        if input is not None:
            textures['image'] = Port(input.image, TextureKind.IMAGE)
            if input.depth is not None:
                textures['depth'] = Port(input.depth.texture, TextureKind.SCALAR)
            if input.mask is not None:
                textures['mask'] = Port(input.mask, TextureKind.SCALAR)
        textures['velocity'] = Port(self._flow.velocity, TextureKind.VELOCITY)
        return MotionOutput(self._flow.velocity, textures=textures)

    # ========== The flow ==========

    def _allocate_stages(self, width: int, height: int) -> None:
        self._flow.allocate(width, height)

    def _deallocate_stages(self) -> None:
        self._flow.deallocate()

    def _reset_stages(self) -> None:
        self._flow.reset()

    def _measure(self, input: MotionInput, interval: float) -> None:
        flow: LucasKanadeFlow3D = self._flow
        flow.set_input(input.image)
        flow.set_mask(input.mask)
        depth: DepthView | None = input.depth
        if depth is not None and not depth.texture.allocated:
            depth = None
        if depth is not None and depth.view_span <= 0.0 and not self._warned_view_span:
            self._warned_view_span = True
            logger.warning("Motion3DLayer %d: depth is on but its view spans nothing, so z reads no motion", self._cam_id)
        flow.update(depth.texture if depth is not None else None, interval,
                    depth.view_span if depth is not None else 0.0,
                    depth.depth_step if depth is not None else (0.0, 0.0))
