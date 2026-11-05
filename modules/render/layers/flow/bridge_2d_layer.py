"""The 2D bridge layer: main's smooth trail, density and temperature on a 2D velocity (xy)."""

# Local application imports
from modules.gl import Texture
from modules.flow import bridge_2d
from modules.settings import Group
from modules.utils import HotReloadMethods
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput
from .port import Port, TextureKind


class Bridge2DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge2DLayer: each bridge's group."""
    trail       = Group(bridge_2d.SmoothTrailSettings)
    density     = Group(bridge_2d.DensityBridgeSettings)
    temperature = Group(bridge_2d.TemperatureBridgeSettings)


class Bridge2DLayer(BridgeLayer):
    """The 2D bridge layer: the smooth trail for the fluid's velocity, its speed as the density's
    amount, the temperature from the density's color where there is speed; the density bridge colors
    the amount for its port only."""

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
        return BridgeOutput(self._trail.velocity, self._velocity_magnitude.magnitude, self._temperature_bridge.temperature, diagnostics={
            'color': Port(self._color, TextureKind.IMAGE),
            'density_bridge': Port(self._density_bridge.density, TextureKind.IMAGE),
        })

    # ========== Lifecycle ==========

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._trail.allocate(width, height, output_width, output_height)
        self._velocity_magnitude.allocate(width, height)
        self._temperature_bridge.allocate(width, height)
        self._density_bridge.allocate(output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._trail.deallocate()
        self._velocity_magnitude.deallocate()
        self._temperature_bridge.deallocate()
        self._density_bridge.deallocate()

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
