"""The motion stage: the layer that measures a velocity field from two camera frames, the optical flow.

`MotionInput` is what the app hands it each tick, `MotionOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it, so an input built before the render loop names the object the stage before writes into.

`MotionLayer` is the shape the 2D and the 3D motion layer share: it measures once per camera frame,
when the input's `timestamp` differs from the one it measured last, over the interval the stamps
give (`frame_interval`). It draws nothing; the monitors draw its output's `textures`."""

from abc import abstractmethod
from dataclasses import dataclass
import math

from modules.gl import Texture, Style
from modules.flow import DepthView
from modules.settings import Field
from .stage_layer import StageLayer, StageLayerSettings, StageOutput

MAX_FRAME_INTERVAL: float = 0.5     # s: a longer gap between two captures is a pause or a loop, not a frame's
FRAME_PERIOD_BAND: tuple[float, float] = (0.5, 2.0)    # a capture gap is held to this share of the frame interval
FRAME_PERIOD_WEIGHT: float = 0.05   # weight of a new gap on the frame interval's Ema


def frame_interval(timestamp: float | None, last: float | None, period: float | None, ticks: float,
                   time_step: float) -> tuple[float, float | None]:
    """The interval (s) a flow frame divides its motion by, and the frame period one frame on: the
    gap between the capture times `last` and `timestamp`, held to FRAME_PERIOD_BAND of `period` (an
    Ema of the gaps); a gap that is not a frame's (none, not finite, 0 or less, a pause or a loop)
    takes `period`, else `ticks` (the time counted in ticks), else `time_step`. Never 0, negative or
    not finite."""
    if period is not None and not (math.isfinite(period) and period > 0.0):
        period = None
    gap: float = timestamp - last if timestamp is not None and last is not None else math.nan
    if not (math.isfinite(gap) and 0.0 < gap < MAX_FRAME_INTERVAL):
        if period is not None:
            return period, period
        return (ticks if math.isfinite(ticks) and ticks > 0.0 else time_step), None
    if period is None:
        return gap, gap
    low, high = FRAME_PERIOD_BAND
    gap = min(max(gap, low * period), high * period)
    return gap, period + FRAME_PERIOD_WEIGHT * (gap - period)


@dataclass
class MotionInput:
    """One camera's picture for the motion stage, built by the app each tick. `image` is the picture
    the velocity is measured on by its brightness; `timestamp` the capture time of the camera frame it
    holds, the frame's identity: the stage measures when it differs from the one it measured last
    (None: every tick); `dt` the render tick's; `depth` the camera's depth on the same view, or None;
    `mask` the player's mask on the view, or None."""
    image: Texture
    timestamp: float | None
    dt: float
    depth: DepthView | None = None
    mask: Texture | None = None


@dataclass
class MotionOutput(StageOutput):
    """The motion stage's textures: `velocity` (x, y, z) in view widths per second, gated by the mask
    with the flow's `use_mask`, z 0 without depth; its ports are the velocity and what it measured on."""
    velocity: Texture


class MotionLayerSettings(StageLayerSettings):
    """What the 2D and the 3D motion layer share; each adds its optical flow's group."""
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")


class MotionLayer(StageLayer[MotionInput, MotionOutput]):
    """The motion stage's layer. `update` measures on a new camera frame (`_measure`). The subclass
    holds its optical flow in `_allocate_stages`, `_deallocate_stages`, `_reset_stages` and `_measure`;
    the flow runs at `settings.width` by `settings.height` and follows a change of them. The layer draws nothing."""

    def __init__(self, settings: MotionLayerSettings) -> None:
        super().__init__(settings)
        self.settings: MotionLayerSettings = settings
        self._timestamp: float | None = None        # the capture time of the frame measured last
        self._frame_period: float | None = None     # the frame interval, an Ema of the capture gaps (s)
        self._ticks: float = 0.0                    # time counted in ticks since the last measurement (s)
        self._flow_size: tuple[int, int] = (0, 0)   # the size the flow runs at; follows settings.width/height

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._allocate_flow()

    def deallocate(self) -> None:
        self._deallocate_flow()
        super().deallocate()

    def reset(self) -> None:
        """Start the flow from rest: the next frame is a first frame."""
        self._reset_stages()
        self._timestamp = None
        self._frame_period = None
        self._ticks = 0.0

    def _allocate_flow(self) -> None:
        flow_width, flow_height = self._flow_size = (self.settings.width, self.settings.height)
        self._allocate_stages(flow_width, flow_height)

    def _deallocate_flow(self) -> None:
        self._deallocate_stages()
        self._flow_size = (0, 0)

    def _follow_flow_size(self) -> None:
        """Reallocate the flow when settings.width/height changed, and start it from rest."""
        if self._flow_size == (0, 0) or self._flow_size == (self.settings.width, self.settings.height):
            return
        self._deallocate_flow()
        self._allocate_flow()
        self.reset()

    @abstractmethod
    def _allocate_stages(self, width: int, height: int) -> None:
        """Allocate the optical flow at the flow's size."""

    @abstractmethod
    def _deallocate_stages(self) -> None: ...

    @abstractmethod
    def _reset_stages(self) -> None: ...

    @abstractmethod
    def _measure(self, input: MotionInput, interval: float) -> None:
        """Measure the velocity between the frame measured last and `input.image`, `interval` seconds apart."""

    # ========== Processing ==========

    def _update(self) -> None:
        input: MotionInput | None = self._input
        if input is None:
            return
        self._follow_flow_size()

        self._ticks += input.dt
        if input.timestamp is None or input.timestamp != self._timestamp:
            Style.push_style()
            Style.set_blend_mode(Style.BlendMode.DISABLED)
            interval, self._frame_period = frame_interval(input.timestamp, self._timestamp, self._frame_period, self._ticks, input.dt)
            if input.timestamp is not None and math.isfinite(input.timestamp):
                self._timestamp = input.timestamp
            self._measure(input, interval)
            self._ticks = 0.0
            Style.pop_style()

    def draw(self) -> None:
        """A stage draws nothing; a monitor draws its textures."""
