"""The fluid stage: the layer that runs the Navier–Stokes simulation on what the bridges give it.

`FluidInput` is what the app hands it each tick: `dt` and one `Injection` per bridge output the
fluid takes. Both hold values only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object
itself, never a buffer taken from it. The fluid knows injections, not cameras."""

from dataclasses import dataclass, field

from modules.gl import Texture


@dataclass
class Injection:
    """One bridge output as one fluid takes it: `velocity`, `density` (RGBA), `temperature`, `onset`
    and `spin` from the bridge output, at `weight`, 0 to 1, the injection's share of the fluid's
    velocity, density and temperature, as a layer's weight in a mix. The app sets the weight from
    whatever it reads; the fluid never knows why."""
    velocity: Texture
    density: Texture
    temperature: Texture
    onset: Texture | None = None
    spin: Texture | None = None
    weight: float = 1.0


@dataclass
class FluidInput:
    """What a fluid takes this tick: `dt`, the render tick's, and its injections."""
    dt: float
    injections: list[Injection] = field(default_factory=list)
