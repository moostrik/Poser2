"""The 3D bridge layer: the velocity trail and the velocity change bridge read a 3D velocity (xyz); their
mix makes what the fluid takes: a velocity, an acceleration, the density with its color and the
temperature."""

# Third-party imports
from OpenGL.GL import GL_R32F, GL_RGBA16F

# Local application imports
from modules.gl import Fbo, GpuTimer, Texture
from modules.flow import (
    VelocityTrailBridge, VelocityTrailBridgeSettings,
    VelocityChangeBridge, VelocityChangeBridgeSettings, Change, WeightedSum,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    HasPorts, Port, Ports, TextureKind,
)
from modules.settings import BaseSettings, Field, Group, Widget
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class MixSettings(BaseSettings):
    """What an output is made of: the trail and each change at its share, 0 to 1, as knobs on the first row."""
    trail = Field(1.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail")
    start = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the start: a part of the body speeding up")
    bow = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the bow: a limb arriving")
    stop = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the stop: a part of the body slowing down")
    wake = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the wake: a limb leaving")


class AccelerationMixSettings(MixSettings):
    """The acceleration: the same sources, none by default."""
    trail = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail")


class DensityMixSettings(DensityBridgeSettings, MixSettings):
    """The density: its amount's mix, the trail's speed and the changes, then its color."""


class TemperatureMixSettings(TemperatureBridgeSettings, MixSettings):
    """The temperature: its mask's mix, the trail's speed and the changes, then its scale."""


class Bridge3DReadingsSettings(BaseSettings):
    """What the bridge reads from the motion: the trail and the changes."""
    trail  = Group(VelocityTrailBridgeSettings)
    change = Group(VelocityChangeBridgeSettings)


class Bridge3DOutputsSettings(BaseSettings):
    """What the fluid takes, each output with its mix."""
    velocity     = Group(MixSettings)
    acceleration = Group(AccelerationMixSettings)
    density      = Group(DensityMixSettings)
    temperature  = Group(TemperatureMixSettings)


CHANGES: tuple[tuple[str, Change], ...] = (('start', Change.START), ('bow', Change.BOW), ('stop', Change.STOP), ('wake', Change.WAKE))


class Bridge3DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge3DLayer: the readings, then the outputs with their mix."""
    readings = Group(Bridge3DReadingsSettings)
    outputs  = Group(Bridge3DOutputsSettings)


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer. The velocity trail and, beside it, the velocity change bridge read the motion;
    each output's group mixes what the fluid takes from them, each source at a share, 0 to 1, as the light
    renderer mixes its layers. The sources share one unit, image widths per second, so the mix has no gain:
    the motion's gains balance the axes and the fluid's set the strength.
        velocity      the trail and the start, bow, stop and wake velocities at their shares
        acceleration  the same sources at shares of its own; None while all are 0
        density       its amount, the trail's speed and the four changes, colored by the density bridge
                      with the density's color
        temperature   its mask, the same sources
    The defaults are the trail alone, and no acceleration. The velocity and the acceleration stay two
    outputs, as a fluid takes them two ways (FLOW.md's Open).

    The change bridge runs while a share of a change is above 0, or a monitor shows one of its ports
    (a port's `prepare` says it is shown, and the next tick runs it), from rest when it starts."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._trail_bridge: VelocityTrailBridge = VelocityTrailBridge(settings.readings.trail)
        self._change_bridge: VelocityChangeBridge = VelocityChangeBridge(settings.readings.change)
        self._change_shown: bool = False        # a monitor read one of its ports since the last update
        self._change_running: bool = False      # it ran the tick before
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._velocity_mix: Fbo = Fbo()         # the fluid's velocity, xyz
        self._acceleration_mix: Fbo = Fbo()     # the fluid's acceleration, xyz
        self._density_amount: Fbo = Fbo()       # the density's amount
        self._temperature_amount: Fbo = Fbo()   # the temperature's mask
        self._mix_shader: WeightedSum = WeightedSum()
        self._density_bridge: DensityBridge = DensityBridge(settings.outputs.density)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(settings.outputs.temperature)
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")

    @property
    def output(self) -> BridgeOutput:
        acceleration: Texture | None = self._acceleration_mix if self._mixed(self.settings.outputs.acceleration) else None
        return BridgeOutput(self._velocity_mix, self._density_bridge.density,
                            self._temperature_bridge.temperature, acceleration)

    def _bridges(self) -> dict[str, HasPorts]:
        """The trail, the change bridge, the speed, the mix, then the density and the temperature."""
        mix: Ports = Ports({'velocity': Port(self._velocity_mix, TextureKind.VELOCITY),
                            'acceleration': Port(self._acceleration_mix, TextureKind.VELOCITY),
                            'density': Port(self._density_amount, TextureKind.SCALAR),
                            'temperature': Port(self._temperature_amount, TextureKind.SCALAR)})
        return {'trail': self._trail_bridge, 'change': self._change_ports(), 'speed': self._velocity_magnitude,
                'mix': mix, 'density': self._density_bridge, 'temperature': self._temperature_bridge}

    def _change_ports(self) -> Ports:
        """The change bridge's ports, each telling it that it is shown when a view reads it."""
        return Ports({name: Port(port.texture, port.kind, port.extent, self._show_change)
                      for name, port in self._change_bridge.ports.items()})

    def _show_change(self) -> None:
        self._change_shown = True

    @staticmethod
    def _mixed(mix: MixSettings) -> bool:
        """Whether an output has a share above 0."""
        return mix.trail > 0.0 or any(getattr(mix, name) > 0.0 for name, _ in CHANGES)

    def _changes_used(self) -> bool:
        """Whether any output takes a change."""
        outputs: Bridge3DOutputsSettings = self.settings.outputs
        mixes: tuple[MixSettings, ...] = (outputs.velocity, outputs.acceleration, outputs.density, outputs.temperature)
        return any(getattr(mix, name) > 0.0 for mix in mixes for name, _ in CHANGES)

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
        self._change_bridge.allocate(width, height)
        self._velocity_magnitude.allocate(width, height)
        self._velocity_mix.allocate(width, height, GL_RGBA16F)
        self._acceleration_mix.allocate(width, height, GL_RGBA16F)
        self._density_amount.allocate(width, height, GL_R32F)
        self._temperature_amount.allocate(width, height, GL_R32F)
        self._temperature_bridge.allocate(width, height)
        self._density_bridge.allocate(output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._trail_bridge.deallocate()
        self._change_bridge.deallocate()
        self._velocity_magnitude.deallocate()
        self._velocity_mix.deallocate()
        self._acceleration_mix.deallocate()
        self._density_amount.deallocate()
        self._temperature_amount.deallocate()
        self._temperature_bridge.deallocate()
        self._density_bridge.deallocate()

    def _reset_stages(self) -> None:
        self._trail_bridge.reset()
        self._change_bridge.reset()
        self._density_bridge.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._gpu_timer.begin()
        self._trail_bridge.set_velocity(input.velocity)
        self._trail_bridge.update(input.dt)
        self._update_change(input, self._change_shown or self._changes_used())
        self._change_shown = False

        self._velocity_magnitude.set_input(self._trail_bridge.smooth_velocity)
        self._velocity_magnitude.update()

        bridge: VelocityChangeBridge = self._change_bridge
        outputs: Bridge3DOutputsSettings = self.settings.outputs
        self._mix(self._velocity_mix, self._velocities(outputs.velocity))
        if self._mixed(outputs.acceleration):
            self._mix(self._acceleration_mix, self._velocities(outputs.acceleration))
        for amount, mix in ((self._density_amount, outputs.density), (self._temperature_amount, outputs.temperature)):
            self._mix(amount, [(self._velocity_magnitude.magnitude, mix.trail),
                               *[(bridge.change(change), getattr(mix, name)) for name, change in CHANGES]])

        self._density_bridge.set_color(color)
        self._density_bridge.set_alpha(self._density_amount)
        self._density_bridge.update()

        self._temperature_bridge.set_color(color)
        self._temperature_bridge.set_mask(self._temperature_amount)
        self._temperature_bridge.update()
        self._gpu_timer.end()

    def _velocities(self, mix: MixSettings) -> list[tuple[Texture, float]]:
        """The trail's velocity and each change's at the output's shares."""
        bridge: VelocityChangeBridge = self._change_bridge
        return [(self._trail_bridge.smooth_velocity, mix.trail),
                *[(bridge.change_velocity(change), getattr(mix, name)) for name, change in CHANGES]]

    def _mix(self, target: Fbo, sources: list[tuple[Texture, float]]) -> None:
        """The sources at their shares, summed into `target`."""
        target.begin()
        self._mix_shader.use(sources)
        target.end()

    def _update_change(self, input: BridgeInput, run: bool) -> None:
        """Step the change bridge while it is `run`, from rest when it starts again."""
        if run:
            bridge: VelocityChangeBridge = self._change_bridge
            if not self._change_running:
                bridge.reset()
            bridge.set_velocity(input.velocity)
            bridge.update(input.dt)
        self._change_running = run
