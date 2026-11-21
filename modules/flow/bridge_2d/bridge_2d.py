"""Bridge2D: the bridge (`Bridge`) on a 2D velocity (xy, RG16F), its velocity lowpass and bandpass and its mixes,
each its own 2D kernel; no z positions. The density's color is the `color` stage's, as the 3D's. How and why:
modules/flow/BRIDGE_2D.md."""

from OpenGL.GL import GL_RG16F

from modules.gl import Fbo, Texture
from ..bridge import AmountMixSettings, Bridge, BridgeSettings, MixSettings, READINGS
from .shaders import SpeedCount2D, SourceMix2D
from .velocity_bandpass_2d import VelocityBandpass2D
from .velocity_lowpass_2d import VelocityLowpass2D


class Bridge2D(Bridge[VelocityLowpass2D, VelocityBandpass2D]):
    """The 2D bridge (`Bridge`): its readings and outputs xy (RG16F)."""

    VELOCITY_FORMAT: int = GL_RG16F

    def __init__(self, settings: BridgeSettings | None = None) -> None:
        settings = settings or BridgeSettings()
        super().__init__(settings, VelocityLowpass2D(settings.filters.lowpass), VelocityBandpass2D(settings.filters.bandpass),
                         SpeedCount2D())
        self._mix_shader: SourceMix2D = SourceMix2D()

    # ========== Lifecycle ==========

    def _allocate_mixes(self, width: int, height: int) -> None:
        self._mix_shader.allocate()

    def _deallocate_mixes(self) -> None:
        self._mix_shader.deallocate()

    # ========== Processing ==========

    def _mix(self, output: str, readings: dict[str, Texture], mix: MixSettings) -> None:
        """The readings at the output's shares, summed."""
        target: Fbo = self._outputs[output]
        target.begin()
        self._mix_shader.use([(readings[name], getattr(mix, name)) for name in READINGS])
        target.end()

    def _mix_amount(self, output: str, mask: Texture | None, mix: AmountMixSettings, capped: bool = False) -> None:
        """An amount: the readings' speeds and the mask at their shares; without a mask the mask's share adds
        nothing."""
        readings: dict[str, Texture] = self._velocities()
        sources: list[tuple[Texture, float]] = [(readings[name], getattr(mix, name)) for name in READINGS]
        masked: bool = mix.mask > 0.0 and mask is not None
        if masked:
            assert mask is not None
            sources.append((mask, mix.mask))
        amount: Fbo = self._outputs[output]
        amount.begin()
        self._mix_shader.use(sources, lengths=True, capped=capped, scalars=int(masked))
        amount.end()
