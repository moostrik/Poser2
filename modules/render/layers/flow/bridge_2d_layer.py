"""The 2D bridge layer: main's smooth trail, density and temperature on a 2D velocity (xy)."""

# Standard library imports
from enum import IntEnum, auto

# Local application imports
from modules.gl import Texture
from modules.flow import bridge_2d
from modules.settings import Field, Group
from modules.utils import HotReloadMethods
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class Bridge2DDrawMode(IntEnum):
    """Draw modes for Bridge2DLayer, in pipeline order.

    VELOCITY_INPUT - The velocity the trail reads
    VELOCITY - The smoothed velocity (RG), what the fluid gets
    SPEED - The smoothed velocity's magnitude, the density's amount, what the fluid gets
    COLOR - The density's color
    DENSITY - The density bridge's colored density (RGBA), a view
    TEMPERATURE - The temperature field, what the fluid gets
    """
    VELOCITY_INPUT = 0
    VELOCITY = auto()
    SPEED = auto()
    COLOR = auto()
    DENSITY = auto()
    TEMPERATURE = auto()


class Bridge2DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge2DLayer: each bridge's group."""
    draw_mode: Field[Bridge2DDrawMode] =    Field(Bridge2DDrawMode.VELOCITY)

    trail       = Group(bridge_2d.SmoothTrailSettings)
    density     = Group(bridge_2d.DensityBridgeSettings)
    temperature = Group(bridge_2d.TemperatureBridgeSettings)


class Bridge2DLayer(BridgeLayer):
    """The 2D bridge layer: the smooth trail for the fluid's velocity, its speed as the density's
    amount, the temperature from the density's color where there is speed; the density bridge colors
    the amount for its view only."""

    def __init__(self, settings: Bridge2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge2DLayerSettings = settings
        self._trail: bridge_2d.VelocitySmoothTrail = bridge_2d.VelocitySmoothTrail(settings.trail)
        self._velocity_magnitude: bridge_2d.VelocityMagnitude = bridge_2d.VelocityMagnitude()
        self._density_bridge: bridge_2d.DensityBridge = bridge_2d.DensityBridge(settings.density)
        self._temperature_bridge: bridge_2d.TemperatureBridge = bridge_2d.TemperatureBridge(settings.temperature)

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def output(self) -> BridgeOutput:
        """The density is the amount alone, the smoothed velocity's speed (R); the 2D fluid colors it at its output."""
        return BridgeOutput(self._trail.velocity, self._velocity_magnitude.magnitude, self._temperature_bridge.temperature)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        super().allocate(width, height, internal_format)
        self._density_bridge.allocate(width, height)

    def deallocate(self) -> None:
        super().deallocate()
        self._density_bridge.deallocate()

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._trail.allocate(width, height, output_width, output_height)
        self._velocity_magnitude.allocate(width, height)
        self._temperature_bridge.allocate(width, height)

    def _deallocate_stages(self) -> None:
        self._trail.deallocate()
        self._velocity_magnitude.deallocate()
        self._temperature_bridge.deallocate()

    def _reset_stages(self) -> None:
        self._trail.reset()
        self._density_bridge.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._trail.set_velocity(input.velocity)
        self._trail.update(input.dt)

        self._velocity_magnitude.set_input(self._trail.velocity)
        self._velocity_magnitude.update()

        self._density_bridge.set_color(color)
        self._density_bridge.set_velocity(self._trail.velocity)
        self._density_bridge.update()

        self._temperature_bridge.set_color(color)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()

    def _visualize(self, input: BridgeInput) -> None:
        textures: dict[Bridge2DDrawMode, Texture] = {
            Bridge2DDrawMode.VELOCITY_INPUT: self._trail.velocity_input,
            Bridge2DDrawMode.VELOCITY: self._trail.velocity,
            Bridge2DDrawMode.SPEED: self._velocity_magnitude.magnitude,
            Bridge2DDrawMode.COLOR: self._color,
            Bridge2DDrawMode.DENSITY: self._density_bridge.density,
            Bridge2DDrawMode.TEMPERATURE: self._temperature_bridge.temperature,
        }
        self._visualizer.process(textures.get(self.settings.draw_mode, self._trail.velocity))
