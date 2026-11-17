"""Lucas–Kanade Flow, the x/y solve: how far each pixel moved between two luminance frames, coarse to fine
(Bouguet's pyramidal Lucas–Kanade), in uv per frame: `solved` as solved, `displacement` with `threshold` (px per
frame) taken off its length. How and why: modules/flow/MOTION.md, "x/y: the Lucas–Kanade solve".
"""

from OpenGL.GL import GL_RG32F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field
from ..port import Port, TextureKind
from .luminance_frames import LuminanceFrames
from .patch import Patch
from .shaders import DisplacementThreshold, LucasKanadeTile

MAX_LEVELS: int = 9     # the most pyramid levels a solve has an Fbo for


class LucasKanadeFlowSettings(BaseSettings):
    """The x/y solve of the displacement."""
    levels:         Field[int]   = Field(3, min=1, max=5, description="Pyramid levels of the x/y solve, each doubles the reach (more: faster motion, more spill)")
    window:         Field[Patch] = Field(Patch.SEVEN, min=Patch.THREE, max=Patch.NINE, description="x/y patch size (larger: cleaner, more spill)")
    damping:        Field[float] = Field(0.03, min=0.0, max=0.1, step=0.001, description="Damps x/y where the patch has little texture (higher: less noise, smaller motion)")
    z_smooth:       Field[bool]  = Field(False, newline=True, width=48, description="x/y smoothed over z's smoothing time, so both lag alike (on: less noise, more lag)")
    threshold:      Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, newline=True, description="x/y below this is cut as noise, in px/frame (higher: less noise, less slow motion)")


class LucasKanadeFlow:
    """The displacement (RG32F, uv per frame) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlowSettings) -> None:
        self.settings: LucasKanadeFlowSettings = settings
        self._frames: LuminanceFrames = LuminanceFrames()
        self._solve: SwapFbo = SwapFbo()            # the displacement as solved, at full size: level 0
        self._levels: list[SwapFbo] = [SwapFbo() for _ in range(1, MAX_LEVELS)]    # levels 1 and up, each at its own size
        self._displacement: Fbo = Fbo()             # with the noise floor taken off
        self._tile_shader: LucasKanadeTile = LucasKanadeTile()     # the solve, its window's current frame shared
        self._threshold_shader: DisplacementThreshold = DisplacementThreshold()

    @property
    def ready(self) -> bool:
        """Whether it holds two frames to measure between."""
        return self._frames.ready

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame."""
        return self._frames.luminance

    @property
    def previous_luminance(self) -> Texture:
        """The luminance frame before the newest."""
        return self._frames.previous_luminance

    @property
    def solved(self) -> Texture:
        """The displacement as solved (RG32F, uv per frame), before the threshold."""
        return self._solve

    @property
    def displacement(self) -> Texture:
        """The displacement (RG32F, uv per frame) with `threshold` taken off its length."""
        return self._displacement

    @property
    def ports(self) -> dict[str, Port]:
        """The luminance it measures on, the pyramid levels it solves coarse to fine, the solve and the displacement."""
        used: int = min(self.settings.levels, MAX_LEVELS) - 1      # the levels above the full-size solve
        levels: dict[str, Port] = {f'level_{level}': Port(self._levels[level - 1], TextureKind.VELOCITY)
                                   for level in range(used, 0, -1)}
        return {'luminance': Port(self._frames.luminance, TextureKind.IMAGE), **levels,
                'solve': Port(self._solve, TextureKind.VELOCITY),
                'displacement': Port(self._displacement, TextureKind.VELOCITY)}

    def allocate(self, width: int, height: int) -> None:
        self._frames.allocate(width, height)
        self._solve.allocate(width, height, GL_RG32F)
        for level, fbo in enumerate(self._levels, start=1):
            fbo.allocate(max(width >> level, 1), max(height >> level, 1), GL_RG32F)     # the size of the frames' mip level
        self._displacement.allocate(width, height, GL_RG32F)
        self._tile_shader.allocate()
        self._threshold_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._frames.deallocate()
        self._solve.deallocate()
        for fbo in self._levels:
            fbo.deallocate()
        self._displacement.deallocate()
        self._tile_shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        self._frames.reset()
        self._solve.clear_all()
        for fbo in self._levels:
            fbo.clear_all()
        self._displacement.clear()

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._frames.set_input(texture)

    def update(self) -> None:
        """Solve the displacement between the newest two frames."""
        if not self._frames.ready:
            self._solve.clear_all()
            self._displacement.clear()
            return
        s: LucasKanadeFlowSettings = self.settings
        self._pyramid(self._solve, self._frames.previous_luminance, self._frames.luminance)

        self._displacement.begin()
        self._threshold_shader.use(self._solve, s.threshold)
        self._displacement.end()

    def _pyramid(self, solve: SwapFbo, previous: Texture, current: Texture) -> None:
        """The displacement carrying `previous` onto `current`, coarse to fine, into `solve`: one solve
        per level at that level's size, each from the coarser one's displacement read linear (uv, so
        it needs no scaling). The finest is `solve`."""
        s: LucasKanadeFlowSettings = self.settings
        estimate: Texture | None = None
        for level in reversed(range(min(s.levels, MAX_LEVELS))):
            target: SwapFbo = solve if level == 0 else self._levels[level - 1]
            target.swap()
            self._tile_shader.use(target, previous, current, estimate, level, s.window, s.damping)     # the coarsest from no motion
            glMemoryBarrier(int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT))
            estimate = target

