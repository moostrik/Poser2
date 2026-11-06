"""What carries the 3D fluid's density: the grid (`GridDensity`) or particles (`ParticleDensity`).

The fluid's solver owns the velocity, the pressure and the temperature; the carrier owns the density.
Each step the fluid hands it its sources (`add`), moves it along the solver's velocity (`update`) and
has it draw itself into the composited output (`composite`). Its `weight` is what the buoyancy reads as
the density's weight, None while it gives none."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from modules.gl import Texture, Texture3D
from ..port import Port
from .fluid_settings import Advection


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
    evaporation: float      # subtracted this step
    advection: Advection


class DensityCarrier(ABC):
    """The density's carrier: allocated on the simulation's grid, fed sources, moved, drawn."""

    @abstractmethod
    def allocate(self, width: int, height: int, layers: int, output_width: int, output_height: int) -> None:
        """Allocate on the simulation's grid (`width` × `height` × `layers`) for an output of the given size."""

    @abstractmethod
    def deallocate(self) -> None: ...

    @property
    @abstractmethod
    def allocated(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None:
        """No density anywhere."""

    @abstractmethod
    def add(self, texture: Texture, gain: float, dt: float, layer: float, spread: float) -> None:
        """A source: `texture` (RGBA: the amount times the color, the amount as alpha) × gain × dt into the
        gaussian slab at `layer` (0 to 1) with sigma `spread`."""

    @abstractmethod
    def update(self, step: CarrierStep) -> None:
        """Move the density one step along the solver's velocity, faded."""

    @abstractmethod
    def composite(self, output: Texture) -> None:
        """Draw the density into the 2D output (RGBA16F)."""

    @property
    @abstractmethod
    def weight(self) -> Texture3D | None:
        """The density on the grid, as the buoyancy reads its weight; None while the carrier gives none."""

    @property
    @abstractmethod
    def ports(self) -> dict[str, Port]: ...

    def reload(self) -> None:
        """Hot-reload its shaders."""

    def set_walls(self, xy: tuple[int, tuple[float, float, float, float]], z: tuple[int, tuple[float, float, float, float]]) -> None:
        """The walls' wrap modes, as GL wrap and border color, for a carrier on textures."""
