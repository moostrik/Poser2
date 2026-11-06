"""The 3D bridge layer: the velocity trail and the velocity impulse bridge on a 3D velocity (xyz), their
mix into what the fluid takes, the density with its color and the temperature."""

# Third-party imports
from OpenGL.GL import GL_R32F, GL_RGBA16F

# Local application imports
from modules.gl import Fbo, GpuTimer, Texture
from modules.flow import (
    VelocityTrailBridge, VelocityTrailBridgeSettings,
    VelocityImpulseBridge, VelocityImpulseBridgeSettings, Impulse, WeightedSum,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    HasPorts, Port, Ports, TextureKind,
)
from modules.settings import BaseSettings, Field, Group, Widget
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class ImpulseMixSettings(BaseSettings):
    """What the fluid's impulse is made of: each impulse's velocity at its share, 0 to 1."""
    start = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the start: a part of the body speeding up")
    bow = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the bow: a limb arriving")
    stop = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the stop: a part of the body slowing down")
    wake = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the wake: a limb leaving")


class AmountMixSettings(BaseSettings):
    """What an amount is made of: the trail's speed and each impulse at its share, 0 to 1."""
    trail = Field(1.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail's speed")
    start = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the start")
    bow = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the bow")
    stop = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the stop")
    wake = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the wake")


IMPULSES: tuple[tuple[str, Impulse], ...] = (('start', Impulse.START), ('bow', Impulse.BOW), ('stop', Impulse.STOP), ('wake', Impulse.WAKE))


class Bridge3DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge3DLayer: each bridge's group, and the mix of what the fluid takes."""
    trail           = Group(VelocityTrailBridgeSettings)
    impulse         = Group(VelocityImpulseBridgeSettings)
    impulse_mix     = Group(ImpulseMixSettings)
    density_mix     = Group(AmountMixSettings)
    temperature_mix = Group(AmountMixSettings)
    density         = Group(DensityBridgeSettings)
    temperature     = Group(TemperatureBridgeSettings)


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer. The velocity trail and, beside it, the velocity impulse bridge read the motion;
    the mix makes what the fluid takes from them, each source at a share, 0 to 1, as the light renderer
    mixes its layers. The sources share one unit, image widths per second, so the mix has no gain: the
    motion's gains balance the axes and the fluid's set the strength.
        velocity      the trail
        impulse       the start, bow, stop and wake velocities (`impulse_mix`); None while all are 0
        density       its amount, the trail's speed and the four impulses (`density_mix`), colored by the
                      density bridge with the density's color
        temperature   its mask, the same sources (`temperature_mix`)
    The defaults are the trail alone. The trail and the impulse stay two velocities, as a fluid may take
    them two ways (FLOW.md's Open).

    The impulse bridge runs while a share of an impulse is above 0, or a monitor shows one of its ports
    (a port's `prepare` says it is shown, and the next tick runs it), from rest when it starts."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._trail_bridge: VelocityTrailBridge = VelocityTrailBridge(settings.trail)
        self._impulse_bridge: VelocityImpulseBridge = VelocityImpulseBridge(settings.impulse)
        self._impulse_shown: bool = False       # a monitor read one of its ports since the last update
        self._impulse_running: bool = False     # it ran the tick before
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._impulse_mix: Fbo = Fbo()          # the fluid's impulse, xyz
        self._density_amount: Fbo = Fbo()       # the density's amount
        self._temperature_amount: Fbo = Fbo()   # the temperature's mask
        self._mix_shader: WeightedSum = WeightedSum()
        self._density_bridge: DensityBridge = DensityBridge(settings.density)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(settings.temperature)
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")

    @property
    def output(self) -> BridgeOutput:
        impulse: Texture | None = self._impulse_mix if self._impulse_mixed() else None
        return BridgeOutput(self._trail_bridge.smooth_velocity, self._density_bridge.density,
                            self._temperature_bridge.temperature, impulse)

    def _bridges(self) -> dict[str, HasPorts]:
        """The trail, the impulse bridge, the speed, the mix, then the density and the temperature."""
        mix: Ports = Ports({'impulse': Port(self._impulse_mix, TextureKind.VELOCITY),
                            'density': Port(self._density_amount, TextureKind.SCALAR),
                            'temperature': Port(self._temperature_amount, TextureKind.SCALAR)})
        return {'trail': self._trail_bridge, 'impulse': self._impulse_ports(), 'speed': self._velocity_magnitude,
                'mix': mix, 'density': self._density_bridge, 'temperature': self._temperature_bridge}

    def _impulse_ports(self) -> Ports:
        """The impulse bridge's ports, each telling it that it is shown when a view reads it."""
        return Ports({name: Port(port.texture, port.kind, port.extent, self._show_impulse)
                      for name, port in self._impulse_bridge.ports.items()})

    def _show_impulse(self) -> None:
        self._impulse_shown = True

    def _impulse_mixed(self) -> bool:
        """Whether the fluid's impulse has a share above 0."""
        mix: ImpulseMixSettings = self.settings.impulse_mix
        return any(getattr(mix, name) > 0.0 for name, _ in IMPULSES)

    def _impulses_used(self) -> bool:
        """Whether any output takes an impulse."""
        amounts: tuple[AmountMixSettings, ...] = (self.settings.density_mix, self.settings.temperature_mix)
        return self._impulse_mixed() or any(getattr(mix, name) > 0.0 for mix in amounts for name, _ in IMPULSES)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        super().allocate(width, height, internal_format)
        self._mix_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._mix_shader.deallocate()
        self._gpu_timer.deallocate()

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._trail_bridge.allocate(width, height, output_width, output_height)
        self._impulse_bridge.allocate(width, height)
        self._velocity_magnitude.allocate(width, height)
        self._impulse_mix.allocate(width, height, GL_RGBA16F)
        self._density_amount.allocate(width, height, GL_R32F)
        self._temperature_amount.allocate(width, height, GL_R32F)
        self._temperature_bridge.allocate(width, height)
        self._density_bridge.allocate(output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._trail_bridge.deallocate()
        self._impulse_bridge.deallocate()
        self._velocity_magnitude.deallocate()
        self._impulse_mix.deallocate()
        self._density_amount.deallocate()
        self._temperature_amount.deallocate()
        self._temperature_bridge.deallocate()
        self._density_bridge.deallocate()

    def _reset_stages(self) -> None:
        self._trail_bridge.reset()
        self._impulse_bridge.reset()
        self._density_bridge.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._gpu_timer.begin()
        self._trail_bridge.set_velocity(input.velocity)
        self._trail_bridge.update(input.dt)
        self._update_impulse(input, self._impulse_shown or self._impulses_used())
        self._impulse_shown = False

        self._velocity_magnitude.set_input(self._trail_bridge.smooth_velocity)
        self._velocity_magnitude.update()

        bridge: VelocityImpulseBridge = self._impulse_bridge
        if self._impulse_mixed():
            mix: ImpulseMixSettings = self.settings.impulse_mix
            self._mix(self._impulse_mix, [(bridge.impulse_velocity(impulse), getattr(mix, name)) for name, impulse in IMPULSES])
        for amount, mix_settings in ((self._density_amount, self.settings.density_mix),
                                     (self._temperature_amount, self.settings.temperature_mix)):
            self._mix(amount, [(self._velocity_magnitude.magnitude, mix_settings.trail),
                               *[(bridge.impulse(impulse), getattr(mix_settings, name)) for name, impulse in IMPULSES]])

        self._density_bridge.set_color(color)
        self._density_bridge.set_alpha(self._density_amount)
        self._density_bridge.update()

        self._temperature_bridge.set_color(color)
        self._temperature_bridge.set_mask(self._temperature_amount)
        self._temperature_bridge.update()
        self._gpu_timer.end()

    def _mix(self, target: Fbo, sources: list[tuple[Texture, float]]) -> None:
        """The sources at their shares, summed into `target`."""
        target.begin()
        self._mix_shader.use(sources)
        target.end()

    def _update_impulse(self, input: BridgeInput, run: bool) -> None:
        """Step the impulse bridge while it is `run`, from rest when it starts again."""
        if run:
            bridge: VelocityImpulseBridge = self._impulse_bridge
            if not self._impulse_running:
                bridge.reset()
            bridge.set_velocity(input.velocity)
            bridge.update(input.dt)
        self._impulse_running = run
