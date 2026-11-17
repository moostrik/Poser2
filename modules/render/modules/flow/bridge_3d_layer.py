"""The 3D bridge layer: the velocity lowpass and the velocity bandpass read a 3D velocity (xyz) into six
readings, each as a velocity and as an acceleration; one mix per output makes what the fluid takes from them:
the boundary's velocity, density and temperature from the readings as velocities, the body force's
acceleration from the readings as accelerations, and the expansion, where the water swells, a presence from
the readings as velocities and the mask. The density and the temperature are 0 to 1 (±1 warm): a presence,
the readings' speed in widths/s, capped at 1, the dye the fluid fills the water toward.

Each trail holds where its motion was, a z position; every output mixes its readings' z positions by the
readings' sizes at its shares (`ZPositionMix`), so the fluid takes each at the depth it was made at, and a
density or a temperature has a z position wherever its amount has one."""

# Standard library imports
from collections.abc import Callable

# Third-party imports
from OpenGL.GL import GL_R32F, GL_RG16F, GL_RGBA16F

# Local application imports
from modules.gl import Fbo, GpuTimer, Texture
from modules.flow import (
    VelocityLowpass, VelocityLowpassSettings,
    VelocityBandpass, VelocityBandpassSettings, Change, WeightedSum, ZPositionMix,
    TemperatureBridge, TemperatureBridgeSettings, CaptureMax,
    DensityBridge, DensityBridgeSettings, VelocityPeak, VelocityPeakSettings,
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


class TrailMixSettings(MixSettings):
    """A mix of the trail by default: a boundary output's."""
    trail = Field(1.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail: the motion, held and smoothed")


class VelocityMixSettings(TrailMixSettings):
    """The velocity, the boundary the water is dragged toward: its mix, and the peak its speed reached."""
    peak = Group(VelocityPeakSettings)


class AmountMixSettings(MixSettings):
    """An amount's mix: the readings' speeds and the mask, a presence while still."""
    mask = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the mask: the player there, still or not")


class TrailAmountMixSettings(AmountMixSettings, TrailMixSettings):
    """An amount's mix of the trail by default: a boundary amount's."""


class DensityMixSettings(DensityBridgeSettings, TrailAmountMixSettings):
    """The density, held at the body as a boundary value: its amount's mix, then its color."""


class TemperatureMixSettings(TemperatureBridgeSettings, TrailAmountMixSettings):
    """The temperature, held at the body as a boundary value: its amount's mix, then its warmth."""


class Bridge3DReadingsSettings(BaseSettings):
    """What the bridge reads from the motion: the lowpass and the bandpass."""
    lowpass  = Group(VelocityLowpassSettings)
    bandpass = Group(VelocityBandpassSettings)


READINGS: tuple[str, ...] = ('trail', 'change', 'start', 'bow', 'stop', 'wake')
BANDPASS_READINGS: tuple[str, ...] = READINGS[1:]
PARTS: dict[str, Change] = {'start': Change.START, 'bow': Change.BOW, 'stop': Change.STOP, 'wake': Change.WAKE}


class ExpansionSettings(AmountMixSettings):
    """The expansion: where the water swells, a presence 0 to 1 from the readings' speeds and the mask."""


class Bridge3DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge3DLayer: the readings, then one group per output, named as the output, in the
    order the bridge makes them: the velocity, the boundary the water is dragged toward (the readings mixed as
    velocities), the acceleration, the body force added × dt (the readings mixed as accelerations), the density
    and the temperature, held at the body as boundary values, and the expansion."""
    readings     = Group(Bridge3DReadingsSettings)
    velocity     = Group(VelocityMixSettings)
    acceleration = Group(MixSettings)
    density      = Group(DensityMixSettings)
    temperature  = Group(TemperatureMixSettings)
    expansion    = Group(ExpansionSettings)


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer. The lowpass reads the trail, the bandpass the change, start, bow, stop and wake, each
    reading as a velocity (widths/s) and as an acceleration (widths/s²). Each output mixes the six at a share,
    0 to 1, as the light renderer mixes its layers; the readings of one unit share it, so the mix has no gain.
        velocity      the readings as velocities: the boundary, dragged toward
        acceleration  the readings as accelerations: the body force, added; None while its shares are all 0
        density       the readings' speeds and the mask, colored by a density bridge
        temperature   the readings' speeds and the mask, warmed by a temperature bridge
        expansion     the readings' speeds and the mask: a presence, capped at 1; None while its
                      shares are all 0, and the newest capture's in a tick of several
    The defaults are the trail alone in the velocity, the density and the temperature.

    The bandpass runs while a share of one of its readings is above 0, or a monitor shows one of its ports
    (a port's `prepare` says it is shown, and the next tick runs it), from rest when it starts; a filter
    writes its accelerations while the acceleration takes one of its readings or a monitor shows it.

    With several captures in one render tick (`BridgeInput.same_tick`) the densities and the temperatures it
    hands out are, per pixel, those of the capture with the larger amount (`CaptureMax`); the rest the newest's."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._lowpass: VelocityLowpass = VelocityLowpass(settings.readings.lowpass)
        self._bandpass: VelocityBandpass = VelocityBandpass(settings.readings.bandpass)
        self._lowpass_shown: bool = False       # a monitor read one of its ports since the last update
        self._bandpass_shown: bool = False
        self._bandpass_running: bool = False    # it ran the tick before
        self._velocity: Fbo = Fbo()                     # xyz, widths/s
        self._acceleration: Fbo = Fbo()                 # xyz, widths/s²
        self._density_amount: Fbo = Fbo()
        self._temperature_amount: Fbo = Fbo()
        self._velocity_z_position: Fbo = Fbo()          # red over green
        self._acceleration_z_position: Fbo = Fbo()
        self._density_z_position: Fbo = Fbo()
        self._temperature_z_position: Fbo = Fbo()
        self._expansion: Fbo = Fbo()                    # a presence, 0 to 1
        self._expansion_z_position: Fbo = Fbo()
        self._peak: VelocityPeak = VelocityPeak(settings.velocity.peak)    # the velocity's p99 speed
        self._mix_shader: WeightedSum = WeightedSum()
        self._z_position_mix_shader: ZPositionMix = ZPositionMix()
        self._density: DensityBridge = DensityBridge(settings.density)
        self._temperature: TemperatureBridge = TemperatureBridge(settings.temperature)
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")
        self._lowpass_timer: GpuTimer = GpuTimer("GPU bridge lowpass")
        self._bandpass_timer: GpuTimer = GpuTimer("GPU bridge bandpass")
        self._mix_timer: GpuTimer = GpuTimer("GPU bridge mix")      # the outputs' mixes, the density and the temperature
        # The density and the temperature over a tick's captures
        self._capture_max: dict[str, CaptureMax] = {name: CaptureMax(by_alpha=name.endswith('density'))
                                                    for name in ('density', 'temperature')}
        self._merging: bool = False     # the outputs are the larger of this tick's captures

    @property
    def output(self) -> BridgeOutput:
        acceleration: bool = self._mixed(self.settings.acceleration)
        merged: dict[str, Texture] = self._merged()
        return BridgeOutput(
            self._velocity, merged['density'], merged['temperature'],
            self._acceleration if acceleration else None,
            self._velocity_z_position, self._density_z_position, self._temperature_z_position,
            self._acceleration_z_position if acceleration else None,
            self._expansion if self._mixed(self.settings.expansion) else None,
            self._expansion_z_position if self._mixed(self.settings.expansion) else None)

    def _outputs(self) -> dict[str, Texture]:
        """The density and the temperature as the last step made them."""
        return {'density': self._density.density, 'temperature': self._temperature.temperature}

    def _merged(self) -> dict[str, Texture]:
        """The density and the temperature the fluid takes: the larger of this tick's captures while it had several."""
        if self._merging:
            return {name: capture_max.texture for name, capture_max in self._capture_max.items()}
        return self._outputs()

    def _bridges(self) -> dict[str, HasPorts]:
        """The readings, then the outputs' mixes, the density and the temperature."""
        return {'lowpass': self._shown_ports(self._lowpass.ports, self._show_lowpass),
                'bandpass': self._shown_ports(self._bandpass.ports, self._show_bandpass),
                'velocity': Ports({'velocity': Port(self._velocity, TextureKind.VELOCITY),
                                   'z_position': Port(self._velocity_z_position, TextureKind.Z_POSITION)}),
                'acceleration': Ports({'acceleration': Port(self._acceleration, TextureKind.ACCELERATION),
                                       'z_position': Port(self._acceleration_z_position, TextureKind.Z_POSITION)}),
                'density': Ports({'amount': Port(self._density_amount, TextureKind.SCALAR),
                                  'z_position': Port(self._density_z_position, TextureKind.Z_POSITION),
                                  **self._density.ports}),
                'temperature': Ports({'amount': Port(self._temperature_amount, TextureKind.SCALAR),
                                      'z_position': Port(self._temperature_z_position, TextureKind.Z_POSITION),
                                      **self._temperature.ports}),
                'expansion': Ports({'expansion': Port(self._expansion, TextureKind.SCALAR),
                                    'z_position': Port(self._expansion_z_position, TextureKind.Z_POSITION)})}

    @staticmethod
    def _shown_ports(ports: dict[str, Port], shown: Callable[[], None]) -> Ports:
        """A filter's ports, each telling it that it is shown when a view reads it."""
        return Ports({name: Port(port.texture, port.kind, port.extent, shown) for name, port in ports.items()})

    def _show_lowpass(self) -> None:
        self._lowpass_shown = True

    def _show_bandpass(self) -> None:
        self._bandpass_shown = True

    @staticmethod
    def _mixed(mix: MixSettings, names: tuple[str, ...] | None = None) -> bool:
        """Whether an output has a share above 0 among `names`: by default its readings, and an amount's mask."""
        if names is None:
            names = (*READINGS, 'mask') if isinstance(mix, AmountMixSettings) else READINGS
        return any(getattr(mix, name) > 0.0 for name in names)

    def _mixes(self) -> tuple[MixSettings, ...]:
        """Every output's mix, in the order the bridge makes them."""
        settings: Bridge3DLayerSettings = self.settings
        return (settings.velocity, settings.acceleration, settings.density,
                settings.temperature, settings.expansion)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        super().allocate(width, height, internal_format)
        self._mix_shader.allocate()
        self._z_position_mix_shader.allocate()
        self._peak.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._peak.deallocate()
        self._mix_shader.deallocate()
        self._z_position_mix_shader.deallocate()
        for timer in (self._gpu_timer, self._lowpass_timer, self._bandpass_timer, self._mix_timer):
            timer.deallocate()
        for capture_max in self._capture_max.values():
            capture_max.deallocate()

    def _mix_fbos(self) -> tuple[tuple[Fbo, int], ...]:
        return ((self._velocity, GL_RGBA16F), (self._acceleration, GL_RGBA16F),
                (self._density_amount, GL_R32F), (self._temperature_amount, GL_R32F),
                (self._expansion, GL_R32F), *((fbo, GL_RG16F) for fbo in self._z_position_fbos()))

    def _z_position_fbos(self) -> tuple[Fbo, ...]:
        return (self._velocity_z_position, self._acceleration_z_position,
                self._density_z_position, self._temperature_z_position, self._expansion_z_position)

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._lowpass.allocate(width, height, output_width, output_height)
        self._bandpass.allocate(width, height)
        for fbo, internal_format in self._mix_fbos():
            fbo.allocate(width, height, internal_format)
        self._density.allocate(output_width, output_height)
        self._temperature.allocate(width, height)

    def _deallocate_stages(self) -> None:
        self._lowpass.deallocate()
        self._bandpass.deallocate()
        for fbo, _ in self._mix_fbos():
            fbo.deallocate()
        for stage in (self._density, self._temperature):
            stage.deallocate()

    def _reset_stages(self) -> None:
        self._merging = False
        self._lowpass.reset()
        self._bandpass.reset()
        self._density.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._gpu_timer.begin()
        if input.same_tick and not self._merging:      # the tick's second capture: keep the first's
            for name, texture in self._outputs().items():
                self._capture_max[name].hold(texture)
        self._merging = input.same_tick
        settings: Bridge3DLayerSettings = self.settings
        acceleration: MixSettings = settings.acceleration

        self._lowpass_timer.begin()
        self._lowpass.set_velocity(input.velocity)
        self._lowpass.set_z_position(input.z_position)
        self._lowpass.update(input.dt, self._lowpass_shown or acceleration.trail > 0.0)
        self._lowpass_timer.end()
        self._bandpass_timer.begin()
        self._update_bandpass(input, self._bandpass_shown or any(self._mixed(mix, BANDPASS_READINGS) for mix in self._mixes()),
                              self._bandpass_shown or self._mixed(acceleration, BANDPASS_READINGS))
        self._bandpass_timer.end()
        self._lowpass_shown = self._bandpass_shown = False
        self._mix_timer.begin()

        self._mix(self._velocity, self._velocities(), settings.velocity)
        self._mix_z_position(self._velocity_z_position, self._velocities(), settings.velocity)
        self._peak.update(self._velocity, input.dt)
        if self._mixed(acceleration):
            self._mix(self._acceleration, self._accelerations(), acceleration)
            self._mix_z_position(self._acceleration_z_position, self._accelerations(), acceleration)
        self._mix_amount(self._density_amount, self._density_z_position, input, settings.density)
        self._update_density(self._density, self._density_amount, color)
        self._mix_amount(self._temperature_amount, self._temperature_z_position, input, settings.temperature)
        self._update_temperature(self._temperature, self._temperature_amount, color)
        if self._mixed(settings.expansion):
            self._mix_amount(self._expansion, self._expansion_z_position, input, settings.expansion, capped=True)
        if self._merging:
            for name, texture in self._outputs().items():
                self._capture_max[name].take(texture)
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

    def _mix(self, target: Fbo, readings: dict[str, Texture], mix: MixSettings) -> None:
        """The readings at the output's shares, summed into `target`."""
        target.begin()
        self._mix_shader.use([(readings[name], getattr(mix, name)) for name in READINGS])
        target.end()

    def _mix_z_position(self, target: Fbo, readings: dict[str, Texture], mix: MixSettings,
                        mask: tuple[Texture, Texture, float] | None = None) -> None:
        """The z position of an output mixing `readings` at its shares, and `mask` (the mask, the motion's z
        position now, its share), into `target`."""
        lowpass, bandpass = self._lowpass, self._bandpass
        target.begin()
        self._z_position_mix_shader.use([(readings[name], getattr(mix, name)) for name in READINGS],
                                        (lowpass.trail, bandpass.fast, bandpass.slow),
                                        (lowpass.z_position, bandpass.fast_z_position, bandpass.slow_z_position), mask)
        target.end()

    def _mix_amount(self, amount: Fbo, z_position: Fbo, input: BridgeInput, mix: AmountMixSettings,
                    capped: bool = False) -> None:
        """An amount: the readings' speeds at its shares and the mask at its share, into `amount`, with `capped`
        held at 1, a presence, and its z position, into `z_position`. Without a mask or the motion's z position
        the mask's share adds nothing."""
        readings: dict[str, Texture] = self._velocities()
        sources: list[tuple[Texture, float]] = [(readings[name], getattr(mix, name)) for name in READINGS]
        mask: tuple[Texture, Texture, float] | None = None
        if mix.mask > 0.0 and input.mask is not None and input.z_position is not None:
            mask = (input.mask, input.z_position, mix.mask)
            sources.append((input.mask, mix.mask))
        amount.begin()
        self._mix_shader.use(sources, lengths=True, capped=capped)
        amount.end()
        self._mix_z_position(z_position, readings, mix, mask)

    @staticmethod
    def _update_density(density: DensityBridge, amount: Texture, color: Texture) -> None:
        """The density from its amount, with the color."""
        density.set_color(color)
        density.set_alpha(amount)
        density.update()

    @staticmethod
    def _update_temperature(temperature: TemperatureBridge, amount: Texture, color: Texture) -> None:
        """The temperature from its amount, with the color."""
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
            bandpass.set_z_position(input.z_position)
            bandpass.update(input.dt, acceleration)
        self._bandpass_running = run
