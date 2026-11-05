"""The 3D motion layer: the Lucas–Kanade flow, x/y from the image and z from the depth."""

# Standard library imports
import logging
from enum import IntEnum, auto

# Local application imports
from modules.flow import LucasKanadeFlow3D, LucasKanadeFlow3DSettings, DepthView
from modules.settings import Field, Group
from modules.utils import HotReloadMethods
from .motion_layer import MotionLayer, MotionLayerSettings, MotionInput, MotionOutput

logger = logging.getLogger(__name__)


class Motion3DDrawMode(IntEnum):
    """Draw modes for Motion3DLayer. The 3D velocity (xyz) draws as white x/y arrows with a ring of radius |z|
    on the arrow's scale, red toward the camera and green away.

    IMAGE - The picture the flow measures on
    DEPTH - The depth, metres
    MASK - The player's mask
    VELOCITY - The flow's 3D velocity, times its gains and the mask
    """
    IMAGE = 0
    DEPTH = auto()
    MASK = auto()
    VELOCITY = auto()


class Motion3DLayerSettings(MotionLayerSettings):
    """Configuration for Motion3DLayer."""
    draw_mode: Field[Motion3DDrawMode] =    Field(Motion3DDrawMode.VELOCITY)

    lucas_kanade = Group(LucasKanadeFlow3DSettings)


class Motion3DLayer(MotionLayer):
    """The 3D motion layer: `LucasKanadeFlow3D` on the input's image, z from its depth, gated by its mask."""

    def __init__(self, cam_id: int, settings: Motion3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Motion3DLayerSettings = settings
        self._cam_id: int = cam_id
        self._flow: LucasKanadeFlow3D = LucasKanadeFlow3D(settings.lucas_kanade)
        self._warned_view_span: bool = False

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def output(self) -> MotionOutput:
        return MotionOutput(self._flow.velocity)

    # ========== The flow ==========

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
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

    def _visualize(self, input: MotionInput) -> None:
        mode: Motion3DDrawMode = self.settings.draw_mode
        if mode == Motion3DDrawMode.DEPTH and input.depth is not None and input.depth.texture.allocated:
            self._visualizer.process_signed(input.depth.texture)
        elif mode == Motion3DDrawMode.MASK and input.mask is not None and input.mask.allocated:
            self._visualizer.process_signed(input.mask)
        elif mode == Motion3DDrawMode.VELOCITY:
            self._visualizer.process(self._flow.velocity, True)
        else:
            self._visualizer.process(input.image)
