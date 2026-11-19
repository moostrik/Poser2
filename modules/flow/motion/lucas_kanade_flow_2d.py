"""Lucas–Kanade Flow 2D: the x/y solve (`LucasKanadeFlow`) as one 2D velocity in image widths per second times
`x_gain` and `y_gain`, gated by the mask with `use_mask` (RG16F). How and why: modules/flow/MOTION.md, "The 2D
velocity".
"""

from OpenGL.GL import GL_RG16F

from modules.gl import Fbo, Texture
from modules.settings import BaseSettings, Field, Group, Widget
from .. import FlowUtil, Port, prefixed
from ..finite_guard import FiniteGuard, GuardSlot
from .lucas_kanade_flow import LucasKanadeFlow, LucasKanadeFlowSettings, per_second
from .shaders import ComposeVelocity2D


class LucasKanadeFlow2DSettings(BaseSettings):
    """Configuration for the 2D Lucas–Kanade flow: the mask and the axes, its x/y solve in `xy`."""
    use_mask:       Field[bool]  = Field(True, newline=True, width=48, description="Only the player moves the fluid (off: whole image)")
    x_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")
    y_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")

    xy             = Group(LucasKanadeFlowSettings)


class LucasKanadeFlow2D:
    """The 2D velocity (RG16F, xy) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlow2DSettings) -> None:
        self.settings: LucasKanadeFlow2DSettings = settings
        self._flow: LucasKanadeFlow = LucasKanadeFlow(settings.xy)
        # The velocity's non-finite values, set to 0 where it is written, counted and reported
        self._guard: FiniteGuard = FiniteGuard('motion', ('velocity',))
        self._mask: Texture | None = None
        self._velocity: Fbo = Fbo()
        self._compose_shader: ComposeVelocity2D = ComposeVelocity2D()

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame."""
        return self._flow.luminance

    @property
    def velocity(self) -> Texture:
        """x and y in image widths per second, each times its gain (RG16F)."""
        return self._velocity

    @property
    def ports(self) -> dict[str, Port]:
        """The x/y solve under its settings group. The composed velocity is the motion stage's output."""
        return prefixed('xy', self._flow.ports)

    def allocate(self, width: int, height: int) -> None:
        self._flow.allocate(width, height)
        self._velocity.allocate(width, height, GL_RG16F)
        self._compose_shader.allocate()
        self._guard.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._flow.deallocate()
        self._velocity.deallocate()
        self._compose_shader.deallocate()
        self._guard.deallocate()

    def reset(self) -> None:
        self._flow.reset()
        self._velocity.clear()

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._flow.set_input(texture)

    def set_mask(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) the velocity is multiplied by with `use_mask`, or None for no mask."""
        self._mask = texture

    def update(self, delta_time: float, guarded: bool = False) -> None:
        """Measure between the newest two frames, `delta_time` seconds apart; with `guarded` set the velocity's
        non-finite values to 0, count and log them."""
        if not self._flow.ready:
            FlowUtil.zero(self._velocity)
            return
        s: LucasKanadeFlow2DSettings = self.settings
        self._flow.update()

        # uv per frame as image widths per second: v is uv of the height, so a square motion stays square
        rate: float = per_second(delta_time)
        aspect: float = self._velocity.height / max(self._velocity.width, 1)
        mask: Texture | None = self._mask if s.use_mask and self._mask is not None and self._mask.allocated else None
        guard: GuardSlot | None = self._guard.slot('velocity') if guarded else None
        self._velocity.begin()
        self._compose_shader.use(self._flow.displacement, (rate * s.x_gain, rate * aspect * s.y_gain), mask, guard)
        self._velocity.end()
        self._guard.update(delta_time)
