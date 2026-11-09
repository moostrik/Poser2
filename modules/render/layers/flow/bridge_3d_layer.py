"""The 3D bridge layer: the velocity lowpass and the velocity bandpass read a 3D velocity (xyz) into six
readings, each as a velocity and as an acceleration; two mixes make what the fluid takes from them: the
boundary (a velocity, density, temperature from the readings as velocities) and the body force (an
acceleration from the readings as accelerations, its density and temperature a push's presence from the
readings as velocities, the change of speed). Each density and temperature is 0 to 1 (±1 warm): a presence,
the readings' speed in widths/s, capped at 1, the dye the fluid fills the water toward or adds at a rate."""

# Standard library imports
from collections.abc import Callable

# Third-party imports
from OpenGL.GL import GL_R32F, GL_RGBA16F

# Local application imports
from modules.gl import Fbo, GpuTimer, Texture
from modules.flow import (
    VelocityLowpass, VelocityLowpassSettings,
    VelocityBandpass, VelocityBandpassSettings, Change, WeightedSum,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    HasPorts, Port, Ports, TextureKind,
)
from modules.settings import BaseSettings, Field, Group, Widget
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class MixSettings(BaseSettings):
    """What an output is made of: each reading at its share, 0 to 1, as knobs on the first row; none by default."""
    trail = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail: the motion, held and smoothed")
    change = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the change: the whole change of speed, signed")
    start = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the start: a part of the body speeding up")
    bow = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the bow: a limb arriving")
    stop = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the stop: a part of the body slowing down")
    wake = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the wake: a limb leaving")


class BoundaryMixSettings(MixSettings):
    """A boundary output's mix: the trail by default."""
    trail = Field(1.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail: the motion, held and smoothed")


class BoundaryDensitySettings(DensityBridgeSettings, BoundaryMixSettings):
    """The boundary's density: its amount's mix, the readings' speeds, then its color."""


class BoundaryTemperatureSettings(TemperatureBridgeSettings, BoundaryMixSettings):
    """The boundary's temperature: its mask's mix, the readings' speeds, then its warmth."""


class BodyForceDensitySettings(DensityBridgeSettings, MixSettings):
    """The body force's density: its amount's mix, the readings' speeds (a push's change of speed), then its color."""


class BodyForceTemperatureSettings(TemperatureBridgeSettings, MixSettings):
    """The body force's temperature: its mask's mix, the readings' speeds (a push's change of speed), then its warmth."""


class Bridge3DReadingsSettings(BaseSettings):
    """What the bridge reads from the motion: the lowpass and the bandpass."""
    lowpass  = Group(VelocityLowpassSettings)
    bandpass = Group(VelocityBandpassSettings)


class BoundarySettings(BaseSettings):
    """The boundary: what the water is dragged toward, the readings mixed as velocities."""
    velocity    = Group(BoundaryMixSettings)
    density     = Group(BoundaryDensitySettings)
    temperature = Group(BoundaryTemperatureSettings)


class BodyForceSettings(BaseSettings):
    """The body force: what is added × dt, the readings mixed as accelerations."""
    acceleration = Group(MixSettings)
    density      = Group(BodyForceDensitySettings)
    temperature  = Group(BodyForceTemperatureSettings)


READINGS: tuple[str, ...] = ('trail', 'change', 'start', 'bow', 'stop', 'wake')
BANDPASS_READINGS: tuple[str, ...] = READINGS[1:]
PARTS: dict[str, Change] = {'start': Change.START, 'bow': Change.BOW, 'stop': Change.STOP, 'wake': Change.WAKE}


class Bridge3DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge3DLayer: the readings, then the boundary and the body force with their mixes."""
    readings   = Group(Bridge3DReadingsSettings)
    boundary   = Group(BoundarySettings)
    body_force = Group(BodyForceSettings)


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer. The lowpass reads the trail, the bandpass the change, start, bow, stop and wake, each
    reading as a velocity (widths/s) and as an acceleration (widths/s²). Each output mixes the six at a share,
    0 to 1, as the light renderer mixes its layers; the readings of one unit share it, so the mix has no gain.
        boundary     the readings as velocities: `velocity`; `density` and `temperature` from their sizes
        body_force   the readings as accelerations: `acceleration`; `density` and `temperature` from the
                     readings as velocities, a push's change of speed; each None while its shares are all 0
    The density and the temperature are colored by a density and a temperature bridge per side, with the
    same color. The defaults are the boundary's trail alone.

    The bandpass runs while a share of one of its readings is above 0, or a monitor shows one of its ports
    (a port's `prepare` says it is shown, and the next tick runs it), from rest when it starts; a filter
    writes its accelerations while the body force takes one of its readings or a monitor shows it."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._lowpass: VelocityLowpass = VelocityLowpass(settings.readings.lowpass)
        self._bandpass: VelocityBandpass = VelocityBandpass(settings.readings.bandpass)
        self._lowpass_shown: bool = False       # a monitor read one of its ports since the last update
        self._bandpass_shown: bool = False
        self._bandpass_running: bool = False    # it ran the tick before
        self._boundary_velocity: Fbo = Fbo()            # xyz, widths/s
        self._boundary_density_amount: Fbo = Fbo()
        self._boundary_temperature_amount: Fbo = Fbo()
        self._body_force_acceleration: Fbo = Fbo()      # xyz, widths/s²
        self._body_force_density_amount: Fbo = Fbo()
        self._body_force_temperature_amount: Fbo = Fbo()
        self._mix_shader: WeightedSum = WeightedSum()
        self._boundary_density: DensityBridge = DensityBridge(settings.boundary.density)
        self._boundary_temperature: TemperatureBridge = TemperatureBridge(settings.boundary.temperature)
        self._body_force_density: DensityBridge = DensityBridge(settings.body_force.density)
        self._body_force_temperature: TemperatureBridge = TemperatureBridge(settings.body_force.temperature)
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")
        self._lowpass_timer: GpuTimer = GpuTimer("GPU bridge lowpass")
        self._bandpass_timer: GpuTimer = GpuTimer("GPU bridge bandpass")
        self._mix_timer: GpuTimer = GpuTimer("GPU bridge mix")      # the outputs' mixes, the densities and temperatures

    @property
    def output(self) -> BridgeOutput:
        body_force: BodyForceSettings = self.settings.body_force
        return BridgeOutput(
            self._boundary_velocity, self._boundary_density.density, self._boundary_temperature.temperature,
            self._body_force_acceleration if self._mixed(body_force.acceleration) else None,
            self._body_force_density.density if self._mixed(body_force.density) else None,
            self._body_force_temperature.temperature if self._mixed(body_force.temperature) else None)

    def _bridges(self) -> dict[str, HasPorts]:
        """The readings, then each side's mix and its density and temperature."""
        boundary: Ports = Ports({'velocity': Port(self._boundary_velocity, TextureKind.VELOCITY),
                                 'density_amount': Port(self._boundary_density_amount, TextureKind.SCALAR),
                                 'temperature_amount': Port(self._boundary_temperature_amount, TextureKind.SCALAR)})
        body_force: Ports = Ports({'acceleration': Port(self._body_force_acceleration, TextureKind.ACCELERATION),
                                   'density_amount': Port(self._body_force_density_amount, TextureKind.SCALAR),
                                   'temperature_amount': Port(self._body_force_temperature_amount, TextureKind.SCALAR)})
        return {'lowpass': self._shown_ports(self._lowpass.ports, self._show_lowpass),
                'bandpass': self._shown_ports(self._bandpass.ports, self._show_bandpass),
                'boundary': boundary, 'boundary_density': self._boundary_density, 'boundary_temperature': self._boundary_temperature,
                'body_force': body_force, 'body_force_density': self._body_force_density,
                'body_force_temperature': self._body_force_temperature}

    @staticmethod
    def _shown_ports(ports: dict[str, Port], shown: Callable[[], None]) -> Ports:
        """A filter's ports, each telling it that it is shown when a view reads it."""
        return Ports({name: Port(port.texture, port.kind, port.extent, shown) for name, port in ports.items()})

    def _show_lowpass(self) -> None:
        self._lowpass_shown = True

    def _show_bandpass(self) -> None:
        self._bandpass_shown = True

    @staticmethod
    def _mixed(mix: MixSettings, names: tuple[str, ...] = READINGS) -> bool:
        """Whether an output has a share above 0 among the readings `names`."""
        return any(getattr(mix, name) > 0.0 for name in names)

    def _boundary_mixes(self) -> tuple[MixSettings, ...]:
        boundary: BoundarySettings = self.settings.boundary
        return (boundary.velocity, boundary.density, boundary.temperature)

    def _body_force_mixes(self) -> tuple[MixSettings, ...]:
        body_force: BodyForceSettings = self.settings.body_force
        return (body_force.acceleration, body_force.density, body_force.temperature)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        super().allocate(width, height, internal_format)
        self._mix_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._mix_shader.deallocate()
        for timer in (self._gpu_timer, self._lowpass_timer, self._bandpass_timer, self._mix_timer):
            timer.deallocate()

    def _mix_fbos(self) -> tuple[tuple[Fbo, int], ...]:
        return ((self._boundary_velocity, GL_RGBA16F), (self._boundary_density_amount, GL_R32F),
                (self._boundary_temperature_amount, GL_R32F), (self._body_force_acceleration, GL_RGBA16F),
                (self._body_force_density_amount, GL_R32F), (self._body_force_temperature_amount, GL_R32F))

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._lowpass.allocate(width, height, output_width, output_height)
        self._bandpass.allocate(width, height)
        for fbo, internal_format in self._mix_fbos():
            fbo.allocate(width, height, internal_format)
        for temperature in (self._boundary_temperature, self._body_force_temperature):
            temperature.allocate(width, height)
        for density in (self._boundary_density, self._body_force_density):
            density.allocate(output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._lowpass.deallocate()
        self._bandpass.deallocate()
        for fbo, _ in self._mix_fbos():
            fbo.deallocate()
        for stage in (self._boundary_temperature, self._body_force_temperature, self._boundary_density, self._body_force_density):
            stage.deallocate()

    def _reset_stages(self) -> None:
        self._lowpass.reset()
        self._bandpass.reset()
        self._boundary_density.reset()
        self._body_force_density.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._gpu_timer.begin()
        body_force: tuple[MixSettings, ...] = self._body_force_mixes()
        mixes: tuple[MixSettings, ...] = (*self._boundary_mixes(), *body_force)

        self._lowpass_timer.begin()
        self._lowpass.set_velocity(input.velocity)
        self._lowpass.update(input.dt, self._lowpass_shown or self.settings.body_force.acceleration.trail > 0.0)
        self._lowpass_timer.end()
        self._bandpass_timer.begin()
        self._update_bandpass(input, self._bandpass_shown or any(self._mixed(mix, BANDPASS_READINGS) for mix in mixes),
                              self._bandpass_shown or self._mixed(self.settings.body_force.acceleration, BANDPASS_READINGS))
        self._bandpass_timer.end()
        self._lowpass_shown = self._bandpass_shown = False
        self._mix_timer.begin()

        boundary: BoundarySettings = self.settings.boundary
        self._mix(self._boundary_velocity, self._velocities(), boundary.velocity)
        self._mix(self._boundary_density_amount, self._velocities(), boundary.density, lengths=True)
        self._mix(self._boundary_temperature_amount, self._velocities(), boundary.temperature, lengths=True)
        self._update_density(self._boundary_density, self._boundary_density_amount, color)
        self._update_temperature(self._boundary_temperature, self._boundary_temperature_amount, color)

        side: BodyForceSettings = self.settings.body_force
        if self._mixed(side.acceleration):
            self._mix(self._body_force_acceleration, self._accelerations(), side.acceleration)
        if self._mixed(side.density):
            self._mix(self._body_force_density_amount, self._velocities(), side.density, lengths=True)
            self._update_density(self._body_force_density, self._body_force_density_amount, color)
        if self._mixed(side.temperature):
            self._mix(self._body_force_temperature_amount, self._velocities(), side.temperature, lengths=True)
            self._update_temperature(self._body_force_temperature, self._body_force_temperature_amount, color)
        self._mix_timer.end()
        self._gpu_timer.end()

    def _velocities(self) -> dict[str, Texture]:
        """The six readings as velocities, widths/s."""
        return {'trail': self._lowpass.trail, 'change': self._bandpass.change,
                **{name: self._bandpass.velocity(part) for name, part in PARTS.items()}}

    def _accelerations(self) -> dict[str, Texture]:
        """The six readings as accelerations, widths/s²."""
        return {'trail': self._lowpass.trail_acceleration, 'change': self._bandpass.change_acceleration,
                **{name: self._bandpass.acceleration(part) for name, part in PARTS.items()}}

    def _mix(self, target: Fbo, readings: dict[str, Texture], mix: MixSettings, lengths: bool = False) -> None:
        """The readings at the output's shares, summed into `target`; with `lengths`, their sizes, for an amount."""
        target.begin()
        self._mix_shader.use([(readings[name], getattr(mix, name)) for name in READINGS], lengths)
        target.end()

    @staticmethod
    def _update_density(density: DensityBridge, amount: Texture, color: Texture) -> None:
        """A side's density from its amount, with the color."""
        density.set_color(color)
        density.set_alpha(amount)
        density.update()

    @staticmethod
    def _update_temperature(temperature: TemperatureBridge, amount: Texture, color: Texture) -> None:
        """A side's temperature from its amount, with the color."""
        temperature.set_color(color)
        temperature.set_mask(amount)
        temperature.update()

    def _update_bandpass(self, input: BridgeInput, run: bool, acceleration: bool) -> None:
        """Step the bandpass while it is `run`, from rest when it starts again."""
        if run:
            bandpass: VelocityBandpass = self._bandpass
            if not self._bandpass_running:
                bandpass.reset()
            bandpass.set_velocity(input.velocity)
            bandpass.update(input.dt, acceleration)
        self._bandpass_running = run
