"""The 3D bridge layer: the velocity trail and the velocity event bridge on a 3D velocity (xyz), the
density with its color and the temperature."""

# Local application imports
from modules.gl import GpuTimer, Texture
from modules.flow import (
    VelocityTrailBridge, VelocityTrailBridgeSettings,
    VelocityEventBridge, VelocityEventBridgeSettings,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    HasPorts, Port, Ports,
)
from modules.settings import Group
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class Bridge3DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge3DLayer: each bridge's group."""
    trail       = Group(VelocityTrailBridgeSettings)
    event       = Group(VelocityEventBridgeSettings)
    density     = Group(DensityBridgeSettings)
    temperature = Group(TemperatureBridgeSettings)


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer. The velocity trail is the fluid's velocity and its speed the density's
    amount; the density bridge colors the amount with the density's color, the temperature bridge
    reads the color where there is speed.

    Next to the trail, the velocity event bridge reads where a move starts and stops (`event.*`). It
    runs while a monitor shows one of its ports, from rest when it starts: a port's `prepare` says it
    is shown, and the next tick runs it. Nothing it makes reaches the fluid yet."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._trail_bridge: VelocityTrailBridge = VelocityTrailBridge(settings.trail)
        self._event_bridge: VelocityEventBridge = VelocityEventBridge(settings.event)
        self._event_shown: bool = False     # a monitor read one of its ports since the last update
        self._event_running: bool = False   # it ran the tick before
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._density_bridge: DensityBridge = DensityBridge(settings.density)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(settings.temperature)
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")

    @property
    def output(self) -> BridgeOutput:
        return BridgeOutput(self._trail_bridge.smooth_velocity, self._density_bridge.density, self._temperature_bridge.temperature)

    def _bridges(self) -> dict[str, HasPorts]:
        """The trail, the event bridge, then the speed, the density and the temperature."""
        return {'trail': self._trail_bridge, 'event': self._event_ports(), 'speed': self._velocity_magnitude,
                'density': self._density_bridge, 'temperature': self._temperature_bridge}

    def _event_ports(self) -> Ports:
        """The event bridge's ports, each telling it that it is shown when a view reads it."""
        return Ports({name: Port(port.texture, port.kind, port.extent, self._show_event)
                      for name, port in self._event_bridge.ports.items()})

    def _show_event(self) -> None:
        self._event_shown = True

    # ========== Lifecycle ==========

    def deallocate(self) -> None:
        super().deallocate()
        self._gpu_timer.deallocate()

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._trail_bridge.allocate(width, height, output_width, output_height)
        self._event_bridge.allocate(width, height)
        self._velocity_magnitude.allocate(width, height)
        self._temperature_bridge.allocate(width, height)
        self._density_bridge.allocate(output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._trail_bridge.deallocate()
        self._event_bridge.deallocate()
        self._velocity_magnitude.deallocate()
        self._temperature_bridge.deallocate()
        self._density_bridge.deallocate()

    def _reset_stages(self) -> None:
        self._trail_bridge.reset()
        self._event_bridge.reset()
        self._density_bridge.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._gpu_timer.begin()
        self._trail_bridge.set_velocity(input.velocity)
        self._trail_bridge.update(input.dt)
        self._update_event(input, self._event_shown)
        self._event_shown = False

        self._velocity_magnitude.set_input(self._trail_bridge.smooth_velocity)
        self._velocity_magnitude.update()

        # The amount the density bridge makes its alpha from: the trail's speed
        self._density_bridge.set_color(color)
        self._density_bridge.set_alpha(self._velocity_magnitude.magnitude)
        self._density_bridge.update()

        self._temperature_bridge.set_color(color)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()
        self._gpu_timer.end()

    def _update_event(self, input: BridgeInput, run: bool) -> None:
        """Step the event bridge while it is `run`, from rest when it starts again."""
        if run:
            bridge: VelocityEventBridge = self._event_bridge
            if not self._event_running:
                bridge.reset()
            bridge.set_velocity(input.velocity)
            bridge.update(input.dt)
        self._event_running = run
