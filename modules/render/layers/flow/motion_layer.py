"""The motion stage: the layer that measures a velocity field from two camera frames, the optical flow.

`MotionInput` is what the app hands it each tick, `MotionOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it, so an input built before the render loop names the object the stage before writes into."""

from dataclasses import dataclass

from modules.gl import Texture
from modules.flow import DepthView


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
