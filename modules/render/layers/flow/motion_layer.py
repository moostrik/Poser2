"""The motion stage: the layer that measures a velocity field from two camera frames, the optical flow.

`MotionInput` is what the app hands it each tick, `MotionOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it, so an input built before the render loop names the object the stage before writes into.

`MotionLayer` is the shape the 2D and the 3D motion layer share: it measures once per camera frame,
when the input's `timestamp` differs from the one it measured last, over the interval the stamps
give (`frame_interval`). A frame is a picture with a capture time: without one (`timestamp` None) no
camera frame is there yet, only a texture, so the motion stays at rest and counts nothing, and after a
frame a None starts it over. It draws nothing; the monitors draw its output's `textures`."""

from abc import abstractmethod
from dataclasses import dataclass
import math
from typing import Annotated

from modules.gl import Texture, Style
from modules.flow import DepthView, TextureKind
from modules.settings import Field
from .stage_layer import StageLayer, StageLayerSettings, StageOutput

MAX_FRAME_INTERVAL: float = 0.5    # s: a longer gap between two captures is a pause or a loop, not a frame's
FRAME_INTERVAL_BAND: tuple[float, float] = (0.5, 2.0)  # a capture gap is held to this share of the expected frame interval


def frame_interval(timestamp: float | None, last: float | None, fps: float, ticks: float, time_step: float) -> float:
    """The interval (s) a flow frame divides its motion by: the gap between the capture times `last`
    and `timestamp`, held to FRAME_INTERVAL_BAND of the expected frame interval, 1 / `fps`, the
    camera's set frame rate; a gap that is not a frame's (none, not finite, 0 or less, a pause or a
    loop) takes the expected interval. Without a frame rate (not finite, 0 or less) the gap is taken
    as it is, and one that is not a frame's takes `ticks` (the time counted in ticks), else
    `time_step`. Never 0, negative or not finite."""
    expected: float | None = 1.0 / fps if math.isfinite(fps) and fps > 0.0 else None
    gap: float = timestamp - last if timestamp is not None and last is not None else math.nan
    if not (math.isfinite(gap) and 0.0 < gap < MAX_FRAME_INTERVAL):
        if expected is not None:
            return expected
        return ticks if math.isfinite(ticks) and ticks > 0.0 else time_step
    if expected is None:
        return gap
    low, high = FRAME_INTERVAL_BAND
    return min(max(gap, low * expected), high * expected)


@dataclass
class MotionInput:
    """One camera's picture for the motion stage, built by the app each tick. `image` is the picture
    the velocity is measured on by its brightness; `timestamp` the capture time of the camera frame it
    holds, the frame's identity: the stage measures when it differs from the one it measured last
    (None: no camera frame yet, nothing to measure); `dt` the render tick's; `depth` the camera's depth on the same view, or None;
    `mask` the player's mask on the view, or None."""
    image: Annotated[Texture, TextureKind.IMAGE]
    timestamp: float | None
    dt: float
    depth: Annotated[DepthView | None, TextureKind.SCALAR] = None
    mask: Annotated[Texture | None, TextureKind.SCALAR] = None


@dataclass
class MotionOutput(StageOutput):
    """The motion stage's textures: `velocity` (x, y, z) in view widths per second, gated by the mask
    with the flow's `use_mask`, z 0 without depth; `z_position`, the body's now in the player's depth range,
    the 3D motion's only, None without."""
    velocity: Annotated[Texture, TextureKind.VELOCITY]
    z_position: Annotated[Texture | None, TextureKind.Z_POSITION] = None


class MotionLayerSettings(StageLayerSettings):
    """What the 2D and the 3D motion layer share; each adds its optical flow's group."""
    guard: Field[bool] =                    Field(True, description="Set non-finite values to 0, count and log them; off once the motion is stable")
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")
    fps: Field[float] =                     Field(30.0, min=1.0, max=120.0, access=Field.INIT, visible=False, description="Camera frame rate the capture gaps are held around")


class MotionLayer(StageLayer[MotionInput, MotionOutput]):
    """The motion stage's layer. `update` measures on a new camera frame (`_measure`). The subclass
    holds its optical flow in `_allocate_stages`, `_deallocate_stages`, `_reset_stages` and `_measure`;
    the flow runs at `settings.width` by `settings.height` and follows a change of them. The layer draws nothing."""

    def __init__(self, settings: MotionLayerSettings) -> None:
        super().__init__(settings)
        self.settings: MotionLayerSettings = settings
        self._timestamp: float | None = None        # the capture time of the frame measured last
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

        # no capture time is no camera frame, only a texture: nothing to measure, and the next frame starts over
        if input.timestamp is None or not math.isfinite(input.timestamp):
            if self._timestamp is not None:
                self.reset()
            return

        self._ticks += input.dt
        if input.timestamp != self._timestamp:
            Style.push_style()
            Style.set_blend_mode(Style.BlendMode.DISABLED)
            interval: float = frame_interval(input.timestamp, self._timestamp, self.settings.fps, self._ticks, input.dt)
            self._timestamp = input.timestamp
            self._measure(input, interval)
            self._ticks = 0.0
            Style.pop_style()

    def draw(self) -> None:
        """A stage draws nothing; a monitor draws its textures."""
