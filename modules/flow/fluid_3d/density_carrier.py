"""What carries the 3D fluid's density: the density (`Density`, color × amount at its own size), the fine density (`FineDensity`, its amount finer still, its color the density's) and particles (`ParticleDensity`); the first always, the others where enabled.

The fluid's solver owns the velocity, the pressure and the temperature; the carrier owns the density.
Each step the fluid hands it its sources (`add`), moves it along the solver's velocity (`update`) and
has it draw itself into the composited output (`composite`). Its `weight` is what the buoyancy reads as
the density's weight, None while it gives none."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from modules.gl import Texture, Texture3D
from ..finite_guard import GuardSlot
from ..port import Port
from ..density_blend import DensityBlend
from .fluid_settings import Advection, Limiter
from .shaders import CompositeLight, MarchLook


@dataclass
class CarrierStep:
    """One step of the solver, what a carrier moves its density by."""
    velocity: Texture3D
    obstacle: Texture3D
    has_obstacles: bool
    aspect: float           # the grid's width over its height
    z_extent: float         # the slab's depth in widths
    dt: float               # s
    step: float             # dt × speed, the advection's distance per unit of velocity
    dissipation: float      # the fade's multiplier this step
    evaporation: float      # dt / evaporation_time, a saturating loss's share this step; 0 off
    evaporation_knee: float # the amount below which the loss turns from a fixed amount to a share
    advection: Advection
    curl: Texture3D | None = None   # the velocity's curl (1/s), while the turbulence needs it
    cell: float = 0.0               # the grid's cell across, in x and y (widths)
    guard: GuardSlot | None = None  # the density's slot in the fluid's FiniteGuard
    limiter: Limiter = Limiter.CLAMP    # MacCormack where its correction overshoots


class DensityCarrier(ABC):
    """The density's carrier: allocated on the simulation's grid, fed sources, moved, drawn."""

    @abstractmethod
    def allocate(self, width: int, height: int, layers: int, extent: float, output_width: int, output_height: int) -> None:
        """Allocate, at its own size, for the simulation's grid (`width` × `height` × `layers`, a box `extent`
        widths deep) it is carried through and an output of the given size."""

    @abstractmethod
    def deallocate(self) -> None: ...

    @property
    @abstractmethod
    def allocated(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None:
        """No density anywhere."""

    @abstractmethod
    def add(self, amount: Texture, color: Texture, strength: float, dt: float, spread: float, blend: DensityBlend,
            gain: float, release: float, z_position: Texture | None = None) -> None:
        """A source, the bridge's `amount` (red) times `gain` times its `color` (RGBA, premultiplied by its alpha,
        the coverage), the dye, into the gaussian slab at each pixel's `z_position` (red over green), nothing
        where it has none or without one, with sigma `spread`, by `blend` (`density_blend.glsl`): over the water's
        by the share `strength`, the amount capped at 1, added × `strength` (dt), or drawn toward it, at most
        fully dyed, by the share `strength`; where the water holds more than the body, down by the share
        `release`; `dt` the step."""

    @abstractmethod
    def update(self, step: CarrierStep) -> None:
        """Move the density one step along the solver's velocity, faded."""

    @abstractmethod
    def composite(self, output: Texture, look: MarchLook, light: CompositeLight | None = None) -> None:
        """Draw the density into the 2D output (RGBA16F) by the march's `look`, lit by `light` where it is given."""

    @property
    @abstractmethod
    def weight(self) -> Texture3D | None:
        """The density on the grid, as the buoyancy reads its weight; None while the carrier gives none."""

    @property
    @abstractmethod
    def ports(self) -> dict[str, Port]: ...

    def set_walls(self, xy: tuple[int, tuple[float, float, float, float]], z: tuple[int, tuple[float, float, float, float]]) -> None:
        """The walls' wrap modes, as GL wrap and border color, for a carrier on textures."""
