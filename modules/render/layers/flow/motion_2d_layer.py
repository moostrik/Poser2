"""The 2D motion layer: the gradient flow, x/y from the image, as on main."""

# Local application imports
from modules.flow import GradientFlow, GradientFlowSettings
from modules.settings import Group
from modules.utils import HotReloadMethods
from .motion_layer import MotionLayer, MotionLayerSettings, MotionInput, MotionOutput
from .port import Port, TextureKind


class Motion2DLayerSettings(MotionLayerSettings):
    """Configuration for Motion2DLayer."""
    gradient = Group(GradientFlowSettings)


class Motion2DLayer(MotionLayer):
    """The 2D motion layer: `GradientFlow` on the input's image. Its velocity is per frame, not per
    second, and it reads no mask, until the 2D flow moves to the Lucas–Kanade solve."""

    def __init__(self, settings: Motion2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Motion2DLayerSettings = settings
        self._flow: GradientFlow = GradientFlow(settings.gradient)

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def output(self) -> MotionOutput:
        return MotionOutput(self._flow.velocity, textures={
            'image': Port(self._flow.color_input, TextureKind.IMAGE),
            'velocity': Port(self._flow.velocity, TextureKind.VELOCITY),
        })

    # ========== The flow ==========

    def _allocate_stages(self, width: int, height: int) -> None:
        self._flow.allocate(width, height)

    def _deallocate_stages(self) -> None:
        self._flow.deallocate()

    def _reset_stages(self) -> None:
        self._flow.reset()

    def _measure(self, input: MotionInput, interval: float) -> None:
        self._flow.set_input(input.image)
        self._flow.update()
