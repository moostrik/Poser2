"""The 3D bridge layer: the velocity trail or the kinetic bridge on a 3D velocity (xyz), the density
with its color and the temperature."""

# Standard library imports
from enum import IntEnum, auto

# Local application imports
from modules.gl import GpuTimer, Texture
from modules.flow import (
    VelocityTrailBridge, VelocityTrailBridgeSettings,
    KineticBridge, KineticBridgeSettings,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
)
from modules.settings import Field, Group
from modules.utils import HotReloadMethods
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput
from .port import Port, TextureKind


class BridgeMethod(IntEnum):
    """How the bridge makes the velocity the fluid reads and the amount the density reads."""
    VELOCITY_TRAIL = 0  # a smoothed velocity, density where there is speed
    KINETIC = auto()  # two trails, reading the change of the motion


class Bridge3DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge3DLayer: the method and each bridge's group."""
    method: Field[BridgeMethod] =           Field(BridgeMethod.VELOCITY_TRAIL, newline=True, description="How the bridge makes the fluid's velocity and the density's amount: one trail, or two trails reading the change of the motion")

    trail       = Group(VelocityTrailBridgeSettings)
    kinetic     = Group(KineticBridgeSettings)
    density     = Group(DensityBridgeSettings)
    temperature = Group(TemperatureBridgeSettings)


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer. `settings.method` picks the velocity trail or the kinetic bridge for the
    fluid's velocity and the density's amount; the density bridge colors the amount with the density's
    color, the temperature bridge reads the color where there is speed. The kinetic bridge's fields
    are ports while it runs: its two trails, the acceleration along the motion and its z, the turn
    across it, the held release, the reservoir, the emission, the onset velocity and the curl."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._trail_bridge: VelocityTrailBridge = VelocityTrailBridge(settings.trail)
        self._kinetic_bridge: KineticBridge = KineticBridge(settings.kinetic)
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._density_bridge: DensityBridge = DensityBridge(settings.density)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(settings.temperature)
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def _kinetic(self) -> bool:
        return self.settings.method == BridgeMethod.KINETIC

    @property
    def _velocity(self) -> Texture:
        """The active bridge's smoothed velocity, xyz (RGBA16F); z is 0 without depth."""
        return self._kinetic_bridge.smooth_velocity if self._kinetic else self._trail_bridge.smooth_velocity

    @property
    def output(self) -> BridgeOutput:
        kinetic: KineticBridgeSettings = self.settings.kinetic
        bridge: KineticBridge = self._kinetic_bridge
        diagnostics: dict[str, Port] = {
            'speed': Port(self._velocity_magnitude.magnitude, TextureKind.SCALAR),
            'color': Port(self._color, TextureKind.IMAGE),
        }
        if self._kinetic:
            diagnostics.update({
                'kinetic_fast': Port(bridge.fast, TextureKind.VELOCITY),
                'kinetic_slow': Port(bridge.slow, TextureKind.VELOCITY),
                'kinetic_acceleration': Port(bridge.acceleration, TextureKind.SCALAR),
                'kinetic_depth_acceleration': Port(bridge.depth_acceleration, TextureKind.SCALAR),
                'kinetic_turn': Port(bridge.turn, TextureKind.SCALAR),
                'kinetic_held_release': Port(bridge.held_release, TextureKind.SCALAR),
                'kinetic_reservoir': Port(bridge.reservoir, TextureKind.SCALAR),
                'kinetic_emission': Port(bridge.emission, TextureKind.SCALAR),
                'kinetic_onset_velocity': Port(bridge.onset_velocity, TextureKind.VELOCITY),
                'kinetic_curl': Port(bridge.curl, TextureKind.SCALAR),
            })
        return BridgeOutput(
            self._velocity, self._density_bridge.density, self._temperature_bridge.temperature,
            bridge.onset_velocity if self._kinetic and kinetic.onset.force > 0.0 else None,
            bridge.curl if self._kinetic and kinetic.onset.spin > 0.0 else None,
            diagnostics=diagnostics,
        )

    # ========== Lifecycle ==========

    def deallocate(self) -> None:
        super().deallocate()
        self._gpu_timer.deallocate()

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._trail_bridge.allocate(width, height, output_width, output_height)
        self._kinetic_bridge.allocate(width, height)
        self._velocity_magnitude.allocate(width, height)
        self._temperature_bridge.allocate(width, height)
        self._density_bridge.allocate(output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._trail_bridge.deallocate()
        self._kinetic_bridge.deallocate()
        self._velocity_magnitude.deallocate()
        self._temperature_bridge.deallocate()
        self._density_bridge.deallocate()

    def _reset_stages(self) -> None:
        self._trail_bridge.reset()
        self._kinetic_bridge.reset()
        self._density_bridge.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._gpu_timer.begin()
        kinetic: bool = self._kinetic
        if kinetic:
            bridge: KineticBridge = self._kinetic_bridge
            spin: float = self.settings.kinetic.onset.spin
            bridge.set_velocity(input.velocity)
            bridge.update(input.dt)
            bridge.update_depth_acceleration()
            bridge.update_curl(spin if spin > 0.0 else 1.0)
        else:
            self._trail_bridge.set_velocity(input.velocity)
            self._trail_bridge.update(input.dt)

        self._velocity_magnitude.set_input(self._velocity)
        self._velocity_magnitude.update()

        # The amount the density bridge makes its alpha from: the kinetic bridge's emission, or the trail's speed
        alpha: Texture = self._kinetic_bridge.emission if kinetic else self._velocity_magnitude.magnitude
        self._density_bridge.set_color(color)
        self._density_bridge.set_alpha(alpha)
        self._density_bridge.update()

        self._temperature_bridge.set_color(color)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()
        self._gpu_timer.end()
