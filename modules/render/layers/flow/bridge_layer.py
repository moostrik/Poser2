"""The bridge stage: the layer that turns the measured velocity into what the fluid injects.

`BridgeInput` is what the app hands it each tick, `BridgeOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it."""

from dataclasses import dataclass

from modules.gl import Texture


@dataclass
class BridgeInput:
    """One camera's velocity for the bridge stage, built by the app each tick. `velocity` is the
    motion output; `dt` the render tick's; `image`, or None, the picture whose RGB colors the
    density; `color` an RGB tint on it, white by default, the color alone without an image; `mask`
    the player's mask on the view, or None, which gates the image."""
    velocity: Texture
    dt: float
    image: Texture | None = None
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mask: Texture | None = None


@dataclass
class BridgeOutput:
    """The bridge stage's textures, what the fluid injects: `velocity`; `density`, RGBA, the amount
    times the color, with the amount as alpha; `temperature`; `onset`, a second velocity, or None;
    `spin`, a curl at its `spin` setting, or None."""
    velocity: Texture
    density: Texture
    temperature: Texture
    onset: Texture | None = None
    spin: Texture | None = None
