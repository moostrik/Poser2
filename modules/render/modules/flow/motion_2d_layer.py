"""The 2D motion layer: the Lucas–Kanade flow, x/y from the image."""

# Local application imports
from modules.gl import GpuTimer
from modules.flow import LucasKanadeFlow2D, LucasKanadeFlow2DSettings, HasPorts
from .motion_layer import MotionLayer, MotionLayerSettings, MotionInput, MotionOutput


class Motion2DLayerSettings(LucasKanadeFlow2DSettings, MotionLayerSettings):
    """Configuration for Motion2DLayer: the stage's settings and its flow's own, one panel, the stage's first."""


class Motion2DLayer(MotionLayer):
    """The 2D motion layer: `LucasKanadeFlow2D` on the input's image, gated by its mask."""

    def __init__(self, settings: Motion2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Motion2DLayerSettings = settings
        self._flow: LucasKanadeFlow2D = LucasKanadeFlow2D(settings)
        self._gpu_timer: GpuTimer = GpuTimer("GPU motion")     # a measure, on each new camera frame

    @property
    def output(self) -> MotionOutput:
        return MotionOutput(self._flow.velocity)

    def _components(self) -> dict[str, HasPorts]:
        return {'': self._flow}     # its own part carries its group's name, xy

    # ========== The flow ==========

    def _allocate_stages(self, width: int, height: int) -> None:
        self._flow.allocate(width, height)

    def _deallocate_stages(self) -> None:
        self._flow.deallocate()
        self._gpu_timer.deallocate()

    def _reset_stages(self) -> None:
        self._flow.reset()

    def _measure(self, input: MotionInput, interval: float) -> None:
        self._flow.set_input(input.image)
        self._flow.set_mask(input.mask)
        self._gpu_timer.begin()
        self._flow.update(interval, guarded=self.settings.guard)
        self._gpu_timer.end()
