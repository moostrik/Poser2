"""Bridge3D: the velocity lowpass and the velocity bandpass read a 3D velocity (xyz) into six readings, each as a
velocity and as an acceleration; one mix per output makes what the fluid takes from them: the boundary's velocity,
density and temperature from the readings as velocities, the body force's acceleration from the readings as
accelerations, and the expansion, where the water swells, a presence from the readings as velocities and the mask.
The density is its amount, the readings' speeds in widths/s and the mask, and its color, the `color` stage's as it
comes, its alpha the coverage; the temperature a presence, the readings' speeds and the mask capped at 1, which the fluid's gain makes
the warmth.

Each trail holds where its motion was, a z position; every output mixes its readings' z positions by the readings'
sizes at its shares (`ZPositionMix`), so the fluid takes each at the depth it was made at, and a density or a
temperature has a z position wherever its amount has one. How and why: modules/flow/BRIDGE_3D.md."""

from collections.abc import Callable

from OpenGL.GL import GL_R32F, GL_RG16F, GL_RGBA16F

from modules.gl import Fbo, GpuTimer, Texture
from ..flow_util import FlowUtil
from ..port import HasPorts, Port, Ports, TextureKind, prefixed
from .bridge_settings import AmountMixSettings, Bridge3DSettings, MixSettings
from .frame_max import FrameMax
from .shaders import Change, WeightedSum, ZPositionMix
from .velocity_bandpass import VelocityBandpass
from .velocity_lowpass import VelocityLowpass
from .velocity_peak import VelocityPeak


READINGS: tuple[str, ...] = ('trail', 'change', 'start', 'bow', 'stop', 'wake')
BANDPASS_READINGS: tuple[str, ...] = READINGS[1:]
PARTS: dict[str, Change] = {'start': Change.START, 'bow': Change.BOW, 'stop': Change.STOP, 'wake': Change.WAKE}


class Bridge3D:
    """The 3D bridge. The lowpass reads the trail, the bandpass the change, start, bow, stop and wake, each reading
    as a velocity (widths/s) and as an acceleration (widths/s²). Each output mixes the six at a share, 0 to 1, as
    the light renderer mixes its layers; the readings of one unit share it, so the mix has no gain.
        velocity      the readings as velocities: the boundary, dragged toward
        acceleration  the readings as accelerations: the body force, added; None while its shares are all 0
        density       the readings' speeds and the mask, the amount; its color the image's as it comes
        temperature   the readings' speeds and the mask: a presence, capped at 1
        expansion     the readings' speeds and the mask: a presence, capped at 1; None while its
                      shares are all 0, and the newest frame's in a tick of several
    The defaults are the trail alone in the velocity, the density and the temperature.

    The bandpass runs while a share of one of its readings is above 0, or a monitor shows one of its ports
    (a port's `prepare` says it is shown, and the next update runs it), from rest when it starts; a filter
    writes its accelerations while the acceleration takes one of its readings or a monitor shows it.

    With several frames in one render tick (`update`'s `same_tick`) the densities and the temperatures it
    hands out are, per pixel, those of the frame with the larger amount (`FrameMax`), the density's color with
    its amount; the rest the newest's."""

    def __init__(self, settings: Bridge3DSettings | None = None) -> None:
        self.settings: Bridge3DSettings = settings or Bridge3DSettings()
        settings = self.settings
        self._color: Fbo = Fbo()        # the density's color this step: the image copied, held when a tick has several
        self._lowpass: VelocityLowpass = VelocityLowpass(settings.filters.lowpass)
        self._bandpass: VelocityBandpass = VelocityBandpass(settings.filters.bandpass)
        self._lowpass_shown: bool = False       # a monitor read one of its ports since the last update
        self._bandpass_shown: bool = False
        self._bandpass_running: bool = False    # it ran the update before
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
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")
        self._lowpass_timer: GpuTimer = GpuTimer("GPU bridge lowpass")
        self._bandpass_timer: GpuTimer = GpuTimer("GPU bridge bandpass")
        self._mix_timer: GpuTimer = GpuTimer("GPU bridge mix")      # the outputs' mixes
        # The density, its color and the temperature over a tick's frames
        self._frame_max: dict[str, FrameMax] = {name: FrameMax() for name in ('density', 'color', 'temperature')}
        self._merging: bool = False     # the outputs are the larger of this tick's frames

    # ========== Outputs ==========

    @property
    def velocity(self) -> Fbo:
        """The boundary's velocity, xyz, widths/s."""
        return self._velocity

    @property
    def velocity_z_position(self) -> Fbo:
        return self._velocity_z_position

    @property
    def acceleration(self) -> Fbo | None:
        """The body force's acceleration, xyz, widths/s²; None while its shares are all 0."""
        return self._acceleration if self._mixed(self.settings.acceleration) else None

    @property
    def acceleration_z_position(self) -> Fbo | None:
        return self._acceleration_z_position if self._mixed(self.settings.acceleration) else None

    @property
    def density(self) -> Texture:
        """The density's amount, the readings' speeds (widths/s) and the mask; the larger of this tick's frames
        while it had several."""
        return self._merged()['density']

    @property
    def color(self) -> Texture:
        """The density's color, RGBA, premultiplied by its alpha, the coverage; the one of the frame with the
        larger amount while this tick had several."""
        return self._merged()['color']

    @property
    def density_z_position(self) -> Fbo:
        return self._density_z_position

    @property
    def temperature(self) -> Texture:
        """The temperature, a presence 0 to 1; the larger of this tick's frames while it had several."""
        return self._merged()['temperature']

    @property
    def temperature_z_position(self) -> Fbo:
        return self._temperature_z_position

    @property
    def expansion(self) -> Fbo | None:
        """The expansion, a presence 0 to 1; None while its shares are all 0."""
        return self._expansion if self._mixed(self.settings.expansion) else None

    @property
    def expansion_z_position(self) -> Fbo | None:
        return self._expansion_z_position if self._mixed(self.settings.expansion) else None

    @property
    def ports(self) -> dict[str, Port]:
        """The filters' readings, then the outputs' mixes, each under its settings group's name."""
        ports: dict[str, Port] = {}
        for name, part in self._parts().items():
            ports.update(prefixed(name, part.ports))
        return ports

    def _outputs(self) -> dict[str, Texture]:
        """The density, its color and the temperature as the last step made them."""
        return {'density': self._density_amount, 'color': self._color,
                'temperature': self._temperature_amount}

    def _merged(self) -> dict[str, Texture]:
        """The density, its color and the temperature the fluid takes: the larger of this tick's frames while it
        had several."""
        if self._merging:
            return {name: frame_max.texture for name, frame_max in self._frame_max.items()}
        return self._outputs()

    def _parts(self) -> dict[str, HasPorts]:
        """The readings, then the outputs' mixes."""
        return {'lowpass': self._shown_ports(self._lowpass.ports, self._show_lowpass),
                'bandpass': self._shown_ports(self._bandpass.ports, self._show_bandpass),
                'velocity': Ports({'velocity': Port(self._velocity, TextureKind.VELOCITY),
                                   'z_position': Port(self._velocity_z_position, TextureKind.Z_POSITION)}),
                'acceleration': Ports({'acceleration': Port(self._acceleration, TextureKind.ACCELERATION),
                                       'z_position': Port(self._acceleration_z_position, TextureKind.Z_POSITION)}),
                'density': Ports({'amount': Port(self._density_amount, TextureKind.SCALAR),
                                  'z_position': Port(self._density_z_position, TextureKind.Z_POSITION)}),
                'temperature': Ports({'amount': Port(self._temperature_amount, TextureKind.SCALAR),
                                      'z_position': Port(self._temperature_z_position, TextureKind.Z_POSITION)}),
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
        settings: Bridge3DSettings = self.settings
        return (settings.velocity, settings.acceleration, settings.density,
                settings.temperature, settings.expansion)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, output_width: int, output_height: int) -> None:
        """The filters and the mixes at the velocity's size, `width` by `height`; the density's color's copy at
        `output_width` by `output_height`."""
        self._color.allocate(output_width, output_height, GL_RGBA16F)
        self._lowpass.allocate(width, height, output_width, output_height)
        self._bandpass.allocate(width, height)
        for fbo, internal_format in self._mix_fbos():
            fbo.allocate(width, height, internal_format)
        self._mix_shader.allocate()
        self._z_position_mix_shader.allocate()
        self._peak.allocate()

    def deallocate(self) -> None:
        self._color.deallocate()
        self._lowpass.deallocate()
        self._bandpass.deallocate()
        for fbo, _ in self._mix_fbos():
            fbo.deallocate()
        self._peak.deallocate()
        self._mix_shader.deallocate()
        self._z_position_mix_shader.deallocate()
        for timer in (self._gpu_timer, self._lowpass_timer, self._bandpass_timer, self._mix_timer):
            timer.deallocate()
        for frame_max in self._frame_max.values():
            frame_max.deallocate()

    def _mix_fbos(self) -> tuple[tuple[Fbo, int], ...]:
        return ((self._velocity, GL_RGBA16F), (self._acceleration, GL_RGBA16F),
                (self._density_amount, GL_R32F), (self._temperature_amount, GL_R32F),
                (self._expansion, GL_R32F), *((fbo, GL_RG16F) for fbo in self._z_position_fbos()))

    def _z_position_fbos(self) -> tuple[Fbo, ...]:
        return (self._velocity_z_position, self._acceleration_z_position,
                self._density_z_position, self._temperature_z_position, self._expansion_z_position)

    def reset(self) -> None:
        """Start the filters from rest."""
        self._merging = False
        self._lowpass.reset()
        self._bandpass.reset()

    # ========== Processing ==========

    def update(self, velocity: Texture, dt: float, z_position: Texture | None = None, mask: Texture | None = None,
               image: Texture | None = None, same_tick: bool = False) -> None:
        """One step on the motion's `velocity` and `z_position`, `dt` the time it covers (s): the density's color,
        `image` (the `color` stage's) copied as it comes, or white without one, then the filters and the mixes;
        `mask` is a share of the amounts. `same_tick`: another frame in the render tick of the step before."""
        if same_tick and not self._merging:      # the tick's second frame: keep the first's, its color included
            for name, texture in self._outputs().items():
                self._frame_max[name].hold(texture)
        self._merging = same_tick
        if image is not None and image.allocated:
            FlowUtil.blit(self._color, image)
        else:
            self._color.clear(1.0, 1.0, 1.0, 1.0)

        self._gpu_timer.begin()
        settings: Bridge3DSettings = self.settings
        acceleration: MixSettings = settings.acceleration

        self._lowpass_timer.begin()
        self._lowpass.set_velocity(velocity)
        self._lowpass.set_z_position(z_position)
        self._lowpass.update(dt, self._lowpass_shown or acceleration.trail > 0.0)
        self._lowpass_timer.end()
        self._bandpass_timer.begin()
        self._update_bandpass(velocity, z_position, dt,
                              self._bandpass_shown or any(self._mixed(mix, BANDPASS_READINGS) for mix in self._mixes()),
                              self._bandpass_shown or self._mixed(acceleration, BANDPASS_READINGS))
        self._bandpass_timer.end()
        self._lowpass_shown = self._bandpass_shown = False
        self._mix_timer.begin()

        self._mix(self._velocity, self._velocities(), settings.velocity)
        self._mix_z_position(self._velocity_z_position, self._velocities(), settings.velocity)
        self._peak.update(self._velocity, dt)
        if self._mixed(acceleration):
            self._mix(self._acceleration, self._accelerations(), acceleration)
            self._mix_z_position(self._acceleration_z_position, self._accelerations(), acceleration)
        self._mix_amount(self._density_amount, self._density_z_position, mask, z_position, settings.density)
        self._mix_amount(self._temperature_amount, self._temperature_z_position, mask, z_position, settings.temperature,
                         capped=True)
        if self._mixed(settings.expansion):
            self._mix_amount(self._expansion, self._expansion_z_position, mask, z_position, settings.expansion, capped=True)
        if self._merging:
            outputs: dict[str, Texture] = self._outputs()
            frame_max: dict[str, FrameMax] = self._frame_max
            frame_max['color'].take(outputs['color'], (frame_max['density'].texture, outputs['density']))
            frame_max['density'].take(outputs['density'])
            frame_max['temperature'].take(outputs['temperature'])
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

    def _mix_amount(self, amount: Fbo, z_position: Fbo, mask: Texture | None, motion_z_position: Texture | None,
                    mix: AmountMixSettings, capped: bool = False) -> None:
        """An amount: the readings' speeds at its shares and `mask` at its share, into `amount`, with `capped`
        held at 1, a presence, and its z position, into `z_position`. Without a mask or the motion's z position
        the mask's share adds nothing."""
        readings: dict[str, Texture] = self._velocities()
        sources: list[tuple[Texture, float]] = [(readings[name], getattr(mix, name)) for name in READINGS]
        masked: tuple[Texture, Texture, float] | None = None
        if mix.mask > 0.0 and mask is not None and motion_z_position is not None:
            masked = (mask, motion_z_position, mix.mask)
            sources.append((mask, mix.mask))
        amount.begin()
        self._mix_shader.use(sources, lengths=True, capped=capped)
        amount.end()
        self._mix_z_position(z_position, readings, mix, masked)

    def _update_bandpass(self, velocity: Texture, z_position: Texture | None, dt: float, run: bool,
                         acceleration: bool) -> None:
        """Step the bandpass while it is `run`, from rest when it starts again."""
        if run:
            bandpass: VelocityBandpass = self._bandpass
            if not self._bandpass_running:
                bandpass.reset()
            bandpass.set_velocity(velocity)
            bandpass.set_z_position(z_position)
            bandpass.update(dt, acceleration)
        self._bandpass_running = run
