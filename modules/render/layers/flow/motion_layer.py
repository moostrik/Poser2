"""The motion stage: the layer that measures a velocity field from two camera frames, the optical flow.

`MotionInput` is what the app hands it each tick, `MotionOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it, so an input built before the render loop names the object the stage before writes into.

`MotionLayer` is the shape the 2D and the 3D motion layer share: it measures once per camera frame,
when the input's `timestamp` differs from the one it measured last, over the interval the stamps
give (`frame_interval`), and visualizes every tick."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from modules.gl import Texture, Style, Blit
from modules.flow import DepthView, Visualizer, VisualizationFieldSettings
from modules.settings import Field, BaseSettings, Group
from ..layer_base import LayerBase

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


@runtime_checkable
class HasCaptureTime(Protocol):
    """A layer that keeps the capture time of the camera frame it holds."""
    @property
    def timestamp(self) -> float | None: ...


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
class MotionOutput:
    """The motion stage's textures: `velocity` (x, y, z) in view widths per second, gated by the mask
    with the flow's `use_mask`; z is 0 without depth."""
    velocity: Texture


class MotionLayerSettings(BaseSettings):
    """What the 2D and the 3D motion layer share; each adds its `draw_mode` and its optical flow's group."""
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")

    visualization = Group(VisualizationFieldSettings)


class MotionLayer(LayerBase, ABC):
    """The motion stage's layer. `set_input` before the render loop; `update` measures on a new camera
    frame (`_measure`) and visualizes (`_visualize`); `output` is the stage's textures. The subclass
    holds its optical flow in `_allocate_stages`, `_deallocate_stages`, `_reset_stages`, `_measure`
    and `_visualize`; the flow runs at `settings.width` by `settings.height` and follows a change of
    them, the visualization at the layer's own size."""

    def __init__(self, settings: MotionLayerSettings) -> None:
        self.settings: MotionLayerSettings = settings
        self._input: MotionInput | None = None
        self._visualizer: Visualizer = Visualizer(settings.visualization)
        self._timestamp: float | None = None        # the capture time of the frame measured last
        self._frame_period: float | None = None     # the frame interval, an Ema of the capture gaps (s)
        self._ticks: float = 0.0                    # time counted in ticks since the last measurement (s)
        self._flow_size: tuple[int, int] = (0, 0)   # the size the flow runs at; follows settings.width/height
        self._output_size: tuple[int, int] = (0, 0) # the layer's allocation size

    def set_input(self, input: MotionInput) -> None:
        """What the layer measures on at its next update."""
        self._input = input

    @property
    @abstractmethod
    def output(self) -> MotionOutput:
        """The stage's textures, allocated once; what the bridge's input is built from."""
        ...

    @property
    def texture(self) -> Texture:
        return self._visualizer.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._output_size = (width, height)
        self._allocate_flow()
        self._visualizer.allocate(width, height)

    def deallocate(self) -> None:
        self._deallocate_flow()
        self._visualizer.deallocate()

    def reset(self) -> None:
        """Start the flow from rest: the next frame is a first frame."""
        self._reset_stages()
        self._timestamp = None
        self._frame_period = None
        self._ticks = 0.0

    def _allocate_flow(self) -> None:
        flow_width, flow_height = self._flow_size = (self.settings.width, self.settings.height)
        self._allocate_stages(flow_width, flow_height, *self._output_size)

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
    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        """Allocate the optical flow at the flow's size; `output_width` by `output_height` is the layer's."""

    @abstractmethod
    def _deallocate_stages(self) -> None: ...

    @abstractmethod
    def _reset_stages(self) -> None: ...

    @abstractmethod
    def _measure(self, input: MotionInput, interval: float) -> None:
        """Measure the velocity between the frame measured last and `input.image`, `interval` seconds apart."""

    @abstractmethod
    def _visualize(self, input: MotionInput) -> None:
        """Draw the texture `settings.draw_mode` names through the visualizer."""

    # ========== Processing ==========

    def update(self) -> None:
        input: MotionInput | None = self._input
        if input is None:
            return
        self._follow_flow_size()

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._ticks += input.dt
        if input.timestamp is None or input.timestamp != self._timestamp:
            interval, self._frame_period = frame_interval(input.timestamp, self._timestamp, self._frame_period, self._ticks, input.dt)
            if input.timestamp is not None and math.isfinite(input.timestamp):
                self._timestamp = input.timestamp
            self._measure(input, interval)
            self._ticks = 0.0

        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self._visualize(input)
        Style.pop_style()

    # ========== Rendering ==========

    def draw(self) -> None:
        Style.push_style()
        Style.set_blend_mode(self.settings.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        Style.pop_style()
