"""Bridge3D: the bridge (`Bridge`) on a 3D velocity (xyz), its velocity lowpass and bandpass and its mixes.

Each trail holds where its motion was, a z position; every output mixes its readings' z positions by the readings'
sizes at its shares (`ZPositionMix`), so the fluid takes each at the depth it was made at, and a density or a
temperature has a z position wherever its amount has one. How and why: modules/flow/BRIDGE_3D.md."""

from OpenGL.GL import GL_RG16F, GL_RGBA16F

from modules.gl import Fbo, Texture
from ..bridge import AmountMixSettings, Bridge, BridgeSettings, MixSettings, READINGS
from ..port import Port, TextureKind
from ..visualization import HistogramCount
from .shaders import WeightedSum, ZPositionMix
from .velocity_bandpass import VelocityBandpass
from .velocity_lowpass import VelocityLowpass


class Bridge3D(Bridge[VelocityLowpass, VelocityBandpass]):
    """The 3D bridge (`Bridge`): its readings and outputs xyz (RGBA16F), each output with its z position."""

    VELOCITY_FORMAT: int = GL_RGBA16F

    def __init__(self, settings: BridgeSettings | None = None) -> None:
        settings = settings or BridgeSettings()
        super().__init__(settings, VelocityLowpass(settings.filters.lowpass), VelocityBandpass(settings.filters.bandpass),
                         HistogramCount())
        self._z_positions: dict[str, Fbo] = {name: Fbo() for name in self._outputs}     # red over green
        self._motion_z_position: Texture | None = None  # the motion's z position this step
        self._mix_shader: WeightedSum = WeightedSum()
        self._z_position_mix_shader: ZPositionMix = ZPositionMix()

    # ========== Outputs ==========

    @property
    def velocity_z_position(self) -> Fbo:
        return self._z_positions['velocity']

    @property
    def acceleration_z_position(self) -> Fbo | None:
        return self._z_positions['acceleration'] if self._mixed(self.settings.acceleration) else None

    @property
    def density_z_position(self) -> Fbo:
        return self._z_positions['density']

    @property
    def temperature_z_position(self) -> Fbo:
        return self._z_positions['temperature']

    @property
    def expansion_z_position(self) -> Fbo | None:
        return self._z_positions['expansion'] if self._mixed(self.settings.expansion) else None

    def _output_ports(self, output: str) -> dict[str, Port]:
        """An output's texture, then its z position."""
        return {**super()._output_ports(output), 'z_position': Port(self._z_positions[output], TextureKind.Z_POSITION)}

    # ========== Lifecycle ==========

    def _allocate_mixes(self, width: int, height: int) -> None:
        for fbo in self._z_positions.values():
            fbo.allocate(width, height, GL_RG16F)
        self._mix_shader.allocate()
        self._z_position_mix_shader.allocate()

    def _deallocate_mixes(self) -> None:
        for fbo in self._z_positions.values():
            fbo.deallocate()
        self._mix_shader.deallocate()
        self._z_position_mix_shader.deallocate()

    # ========== Processing ==========

    def update(self, velocity: Texture, dt: float, z_position: Texture | None = None, mask: Texture | None = None,
               image: Texture | None = None, same_tick: bool = False) -> None:
        """One step on the motion's `velocity` and `z_position` (`Bridge.update`)."""
        self._motion_z_position = z_position
        self._lowpass.set_z_position(z_position)
        self._bandpass.set_z_position(z_position)
        super().update(velocity, dt, mask, image, same_tick)

    def _mix(self, output: str, readings: dict[str, Texture], mix: MixSettings) -> None:
        """The readings at the output's shares, summed, and their z positions."""
        target: Fbo = self._outputs[output]
        target.begin()
        self._mix_shader.use([(readings[name], getattr(mix, name)) for name in READINGS])
        target.end()
        self._mix_z_position(self._z_positions[output], readings, mix)

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

    def _mix_amount(self, output: str, mask: Texture | None, mix: AmountMixSettings, capped: bool = False) -> None:
        """An amount and its z position; without a mask or the motion's z position the mask's share adds
        nothing."""
        readings: dict[str, Texture] = self._velocities()
        sources: list[tuple[Texture, float]] = [(readings[name], getattr(mix, name)) for name in READINGS]
        masked: tuple[Texture, Texture, float] | None = None
        if mix.mask > 0.0 and mask is not None and self._motion_z_position is not None:
            masked = (mask, self._motion_z_position, mix.mask)
            sources.append((mask, mix.mask))
        amount: Fbo = self._outputs[output]
        amount.begin()
        self._mix_shader.use(sources, lengths=True, capped=capped, scalars=int(masked is not None))
        amount.end()
        self._mix_z_position(self._z_positions[output], readings, mix, masked)
