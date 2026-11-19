"""Bridge: what the 2D and the 3D bridge run around their kernels. The velocity lowpass and the velocity bandpass
read the motion's velocity into six readings, each as a velocity and as an acceleration; one mix per output makes
what the fluid takes from them: the boundary's velocity, density and temperature from the readings as velocities,
the body force's acceleration from the readings as accelerations, and the expansion, where the water swells, a
presence from the readings as velocities and the mask. The density is its amount, the readings' speeds in widths/s
and the mask, and its color, the `color` stage's as it comes, its alpha the coverage; the temperature a presence,
the readings' speeds and the mask capped at 1, which the fluid's gain makes the warmth.

The filters and the mixes are each bridge's own kernels, on its own velocity's format; this holds when each runs,
the outputs, the merge of a tick's frames and the ports. How and why: modules/flow/BRIDGE_3D.md and BRIDGE_2D.md."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol

from OpenGL.GL import GL_R32F, GL_RGBA16F

from modules.gl import Fbo, GpuTimer, Texture
from ..flow_util import FlowUtil
from ..port import HasPorts, Port, Ports, TextureKind, prefixed
from .bridge_settings import AmountMixSettings, BridgeSettings, MixSettings
from .change import Change
from .frame_max import FrameMax
from .velocity_peak import SpeedCount, VelocityPeak

READINGS: tuple[str, ...] = ('trail', 'change', 'start', 'bow', 'stop', 'wake')
BANDPASS_READINGS: tuple[str, ...] = READINGS[1:]
PARTS: dict[str, Change] = {'start': Change.START, 'bow': Change.BOW, 'stop': Change.STOP, 'wake': Change.WAKE}
AMOUNTS: tuple[str, ...] = ('density', 'temperature', 'expansion')     # the outputs that are an amount


class Lowpass(Protocol):
    """What a bridge needs from its lowpass: the trail as a velocity and as an acceleration."""
    @property
    def trail(self) -> Texture: ...
    @property
    def trail_acceleration(self) -> Texture: ...
    @property
    def ports(self) -> dict[str, Port]: ...
    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None: ...
    def deallocate(self) -> None: ...
    def reset(self) -> None: ...
    def set_velocity(self, velocity: Texture) -> None: ...
    def update(self, delta_time: float = 1.0 / 60.0, acceleration: bool = False) -> None: ...


class Bandpass(Protocol):
    """What a bridge needs from its bandpass: the change and the four parts, each as a velocity and as an
    acceleration."""
    @property
    def change(self) -> Texture: ...
    @property
    def change_acceleration(self) -> Texture: ...
    @property
    def ports(self) -> dict[str, Port]: ...
    def velocity(self, change: Change) -> Texture: ...
    def acceleration(self, change: Change) -> Texture: ...
    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None: ...
    def deallocate(self) -> None: ...
    def reset(self) -> None: ...
    def set_velocity(self, velocity: Texture) -> None: ...
    def update(self, delta_time: float = 1.0 / 60.0, acceleration: bool = False) -> None: ...


class Bridge[L: Lowpass, B: Bandpass](ABC):
    """A bridge. The lowpass reads the trail, the bandpass the change, start, bow, stop and wake, each reading as a
    velocity (widths/s) and as an acceleration (widths/s²). Each output mixes the six at a share, 0 to 1, as the light
    renderer mixes its layers; the readings of one unit share it, so the mix has no gain.
        velocity      the readings as velocities: the boundary, dragged toward
        acceleration  the readings as accelerations: the body force, added; None while its shares are all 0
        density       the readings' speeds and the mask, the amount; its color the image's as it comes
        temperature   the readings' speeds and the mask: a presence, capped at 1
        expansion     the readings' speeds and the mask: a presence, capped at 1; None while its
                      shares are all 0, and the newest frame's in a tick of several
    The defaults are the trail alone in the velocity, the density and the temperature.

    The bandpass runs while a share of one of its readings is above 0, or a monitor shows one of its ports (a port's
    `prepare` says it is shown, and the next update runs it), from rest when it starts; a filter writes its
    accelerations while the acceleration takes one of its readings or a monitor shows it.

    With several frames in one render tick (`update`'s `same_tick`) the densities and the temperatures it hands out
    are, per pixel, those of the frame with the larger amount (`FrameMax`), the density's color with its amount; the
    rest the newest's.

    A subclass gives its filters, its velocity's format (`VELOCITY_FORMAT`), its speed count and its mixes (`_mix`,
    `_mix_amount`)."""

    VELOCITY_FORMAT: int = GL_RGBA16F       # the velocity's and the acceleration's

    def __init__(self, settings: BridgeSettings, lowpass: L, bandpass: B, count: SpeedCount) -> None:
        self.settings: BridgeSettings = settings
        self._color: Fbo = Fbo()        # the density's color this step: the image copied, held when a tick has several
        self._lowpass: L = lowpass
        self._bandpass: B = bandpass
        self._lowpass_shown: bool = False       # a monitor read one of its ports since the last update
        self._bandpass_shown: bool = False
        self._bandpass_running: bool = False    # it ran the update before
        # The outputs: the velocity and the acceleration in the velocity's format, the amounts in R32F
        self._outputs: dict[str, Fbo] = {name: Fbo() for name in ('velocity', 'acceleration', *AMOUNTS)}
        self._peak: VelocityPeak = VelocityPeak(settings.velocity.peak, count)    # the velocity's p99 speed
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
        """The boundary's velocity, widths/s."""
        return self._outputs['velocity']

    @property
    def acceleration(self) -> Fbo | None:
        """The body force's acceleration, widths/s²; None while its shares are all 0."""
        return self._outputs['acceleration'] if self._mixed(self.settings.acceleration) else None

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
    def temperature(self) -> Texture:
        """The temperature, a presence 0 to 1; the larger of this tick's frames while it had several."""
        return self._merged()['temperature']

    @property
    def expansion(self) -> Fbo | None:
        """The expansion, a presence 0 to 1; None while its shares are all 0."""
        return self._outputs['expansion'] if self._mixed(self.settings.expansion) else None

    @property
    def ports(self) -> dict[str, Port]:
        """The filters' readings, then the outputs' mixes, each under its settings group's name."""
        ports: dict[str, Port] = {}
        for name, part in self._parts().items():
            ports.update(prefixed(name, part.ports))
        return ports

    def _made(self) -> dict[str, Texture]:
        """The density, its color and the temperature as the last step made them."""
        return {'density': self._outputs['density'], 'color': self._color, 'temperature': self._outputs['temperature']}

    def _merged(self) -> dict[str, Texture]:
        """The density, its color and the temperature the fluid takes: the larger of this tick's frames while it
        had several."""
        if self._merging:
            return {name: frame_max.texture for name, frame_max in self._frame_max.items()}
        return self._made()

    def _parts(self) -> dict[str, HasPorts]:
        """The readings, then the outputs' mixes."""
        return {'lowpass': self._shown_ports(self._lowpass.ports, self._show_lowpass),
                'bandpass': self._shown_ports(self._bandpass.ports, self._show_bandpass),
                **{name: Ports(self._output_ports(name)) for name in self._outputs}}

    def _output_ports(self, output: str) -> dict[str, Port]:
        """An output's ports: its texture, under the output's name, or `amount` for an amount."""
        fbo: Fbo = self._outputs[output]
        if output == 'velocity':
            return {'velocity': Port(fbo, TextureKind.VELOCITY)}
        if output == 'acceleration':
            return {'acceleration': Port(fbo, TextureKind.ACCELERATION)}
        return {'expansion' if output == 'expansion' else 'amount': Port(fbo, TextureKind.SCALAR)}

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
        settings: BridgeSettings = self.settings
        return (settings.velocity, settings.acceleration, settings.density, settings.temperature, settings.expansion)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, output_width: int, output_height: int) -> None:
        """The filters and the mixes at the velocity's size, `width` by `height`; the density's color's copy at
        `output_width` by `output_height`."""
        self._color.allocate(output_width, output_height, GL_RGBA16F)
        self._lowpass.allocate(width, height, output_width, output_height)
        self._bandpass.allocate(width, height)
        for name, fbo in self._outputs.items():
            fbo.allocate(width, height, GL_R32F if name in AMOUNTS else self.VELOCITY_FORMAT)
        self._allocate_mixes(width, height)
        self._peak.allocate()

    def deallocate(self) -> None:
        self._color.deallocate()
        self._lowpass.deallocate()
        self._bandpass.deallocate()
        for fbo in self._outputs.values():
            fbo.deallocate()
        self._deallocate_mixes()
        self._peak.deallocate()
        for timer in (self._gpu_timer, self._lowpass_timer, self._bandpass_timer, self._mix_timer):
            timer.deallocate()
        for frame_max in self._frame_max.values():
            frame_max.deallocate()

    def reset(self) -> None:
        """Start the filters from rest."""
        self._merging = False
        self._lowpass.reset()
        self._bandpass.reset()

    @abstractmethod
    def _allocate_mixes(self, width: int, height: int) -> None:
        """The mixes' shaders, and what the subclass mixes besides the outputs, at the velocity's size."""

    @abstractmethod
    def _deallocate_mixes(self) -> None: ...

    # ========== Processing ==========

    def update(self, velocity: Texture, dt: float, mask: Texture | None = None, image: Texture | None = None,
               same_tick: bool = False) -> None:
        """One step on the motion's `velocity`, `dt` the time it covers (s): the density's color, `image` (the
        `color` stage's) copied as it comes, or white without one, then the filters and the mixes; `mask` is a share
        of the amounts. `same_tick`: another frame in the render tick of the step before."""
        if same_tick and not self._merging:      # the tick's second frame: keep the first's, its color included
            for name, texture in self._made().items():
                self._frame_max[name].hold(texture)
        self._merging = same_tick
        if image is not None and image.allocated:
            FlowUtil.blit(self._color, image)
        else:
            self._color.clear(1.0, 1.0, 1.0, 1.0)

        self._gpu_timer.begin()
        settings: BridgeSettings = self.settings
        acceleration: MixSettings = settings.acceleration

        self._lowpass_timer.begin()
        self._lowpass.set_velocity(velocity)
        self._lowpass.update(dt, self._lowpass_shown or acceleration.trail > 0.0)
        self._lowpass_timer.end()
        self._bandpass_timer.begin()
        self._update_bandpass(velocity, dt,
                              self._bandpass_shown or any(self._mixed(mix, BANDPASS_READINGS) for mix in self._mixes()),
                              self._bandpass_shown or self._mixed(acceleration, BANDPASS_READINGS))
        self._bandpass_timer.end()
        self._lowpass_shown = self._bandpass_shown = False
        self._mix_timer.begin()

        self._mix('velocity', self._velocities(), settings.velocity)
        self._peak.update(self._outputs['velocity'], dt)
        if self._mixed(acceleration):
            self._mix('acceleration', self._accelerations(), acceleration)
        self._mix_amount('density', mask, settings.density)
        self._mix_amount('temperature', mask, settings.temperature, capped=True)
        if self._mixed(settings.expansion):
            self._mix_amount('expansion', mask, settings.expansion, capped=True)
        if self._merging:
            made: dict[str, Texture] = self._made()
            frame_max: dict[str, FrameMax] = self._frame_max
            frame_max['color'].take(made['color'], (frame_max['density'].texture, made['density']))
            frame_max['density'].take(made['density'])
            frame_max['temperature'].take(made['temperature'])
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

    @abstractmethod
    def _mix(self, output: str, readings: dict[str, Texture], mix: MixSettings) -> None:
        """The readings at the output's shares, summed into `output`'s texture."""

    @abstractmethod
    def _mix_amount(self, output: str, mask: Texture | None, mix: AmountMixSettings, capped: bool = False) -> None:
        """An amount: the readings' speeds at its shares and `mask` at its share, into `output`'s texture, with
        `capped` held at 1, a presence."""

    def _update_bandpass(self, velocity: Texture, dt: float, run: bool, acceleration: bool) -> None:
        """Step the bandpass while it is `run`, from rest when it starts again."""
        if run:
            bandpass: B = self._bandpass
            if not self._bandpass_running:
                bandpass.reset()
            bandpass.set_velocity(velocity)
            bandpass.update(dt, acceleration)
        self._bandpass_running = run
