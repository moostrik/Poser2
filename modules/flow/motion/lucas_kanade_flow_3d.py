"""Lucas–Kanade Flow 3D: the x/y solve (`LucasKanadeFlow`) and the depth velocity (`DepthVelocity`) composed into
one 3D velocity in image widths per second times `x_gain`, `y_gain`, `z_gain`, gated by the mask with `use_mask`
and z by the depth mask; beside it `z_position`, where the body is now in the player's depth range, 0 front to 1
back, times its weight in red and the weight in green (RG16F). How and why: modules/flow/MOTION.md, "The 3D
velocity" and "depth_velocity (z)".
"""

from OpenGL.GL import GL_RG16F, GL_RG32F, GL_RGBA16F, GL_RGBA32F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field, Group, Widget
from .. import FlowUtil, Port, TextureKind, prefixed
from ..finite_guard import FiniteGuard, GuardSlot
from .depth_velocity import DepthVelocity, DepthVelocitySettings
from .solve_capture import SolveCapture
from .lucas_kanade_flow import LucasKanadeFlow, LucasKanadeFlowSettings, per_second
from .shaders import ComposeVelocity, DisplacementFilter, DisplacementThreshold, ZPosition


class LucasKanadeFlow3DSettings(BaseSettings):
    """Configuration for the 3D Lucas–Kanade flow: the mask and the axes, x/y through z's smoothing, its x/y solve in
    `xy` and its depth velocity in `z`."""
    use_mask:       Field[bool]  = Field(True, newline=True, width=48, description="Only the player moves the fluid (off: whole image)")
    x_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")
    y_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")
    z_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, on the x/y scale")
    smooth_like_z:  Field[bool]  = Field(False, newline=True, width=48, description="x/y smoothed over z's smoothing time, so both lag alike (on: less noise, more lag)")

    xy             = Group(LucasKanadeFlowSettings)
    z              = Group(DepthVelocitySettings)


class LucasKanadeFlow3D:
    """The 3D velocity (RGBA16F, xyz) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlow3DSettings) -> None:
        self.settings: LucasKanadeFlow3DSettings = settings
        self._flow: LucasKanadeFlow = LucasKanadeFlow(settings.xy)
        self._depth_velocity: DepthVelocity = DepthVelocity()
        # The outputs' non-finite values, set to 0 where each is written, counted and reported: the velocity, and the
        # two memories a reset clears, the x/y filter's (displacement) and z's (depth_velocity)
        self._guard: FiniteGuard = FiniteGuard('motion', ('velocity', 'displacement', 'depth_velocity'))
        # The solve's inputs kept when it comes out wrong, with POSER_SOLVE_CAPTURE set (a debugging tool)
        self._capture: SolveCapture | None = SolveCapture.from_environment()
        self._mask: Texture | None = None

        self._filtered: SwapFbo = SwapFbo()         # with smooth_like_z: the filter, its displacement and position error
        self._followed: Fbo = Fbo()                 # the filtered displacement with the noise floor taken off
        self._following: bool = False
        self._velocity: Fbo = Fbo()
        self._z_position: Fbo = Fbo()               # where the body is now in the depth range, red over green
        self._z_position_shader: ZPosition = ZPosition()
        self._threshold_shader: DisplacementThreshold = DisplacementThreshold()
        self._filter_shader: DisplacementFilter = DisplacementFilter()
        self._compose_shader: ComposeVelocity = ComposeVelocity()

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame."""
        return self._flow.luminance

    @property
    def velocity(self) -> Texture:
        """x, y and z in image widths per second, each times its gain (RGBA16F)."""
        return self._velocity

    @property
    def z_position(self) -> Texture:
        """Where the body is now in the depth range, 0 front to 1 back, times its weight in red, the weight in
        green (RG16F); 0 without depth."""
        return self._z_position

    @property
    def depth_velocity(self) -> DepthVelocity:
        """Its z, before `z_gain` and the mask."""
        return self._depth_velocity

    @property
    def ports(self) -> dict[str, Port]:
        """Its two parts under their settings groups: the x/y solve with its smoothing, and the z.
        The composed velocity is the motion stage's output."""
        return {**prefixed('xy', {**self._flow.ports,
                                  'filtered': Port(self._filtered, TextureKind.VELOCITY, held=2),     # BA the position error, state
                                  'followed': Port(self._followed, TextureKind.VELOCITY)}),
                **prefixed('z', self._depth_velocity.ports),
                'z_position': Port(self._z_position, TextureKind.Z_POSITION)}

    def allocate(self, width: int, height: int) -> None:
        self._flow.allocate(width, height)
        self._depth_velocity.allocate(width, height)
        self._filtered.allocate(width, height, GL_RGBA32F)
        self._followed.allocate(width, height, GL_RG32F)
        self._velocity.allocate(width, height, GL_RGBA16F)
        self._z_position.allocate(width, height, GL_RG16F)
        self._z_position_shader.allocate()
        self._threshold_shader.allocate()
        self._filter_shader.allocate()
        self._compose_shader.allocate()
        self._guard.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._flow.deallocate()
        self._depth_velocity.deallocate()
        self._filtered.deallocate()
        self._followed.deallocate()
        self._velocity.deallocate()
        self._z_position.deallocate()
        self._z_position_shader.deallocate()
        self._threshold_shader.deallocate()
        self._filter_shader.deallocate()
        self._compose_shader.deallocate()
        self._guard.deallocate()

    def reset(self) -> None:
        self._flow.reset()
        self._depth_velocity.reset()
        self._filtered.clear_all()
        self._followed.clear()
        self._velocity.clear()
        self._z_position.clear()

    def _slot(self, output: str, guarded: bool) -> GuardSlot | None:
        """The output's slot in the motion's guard while `guarded`, else None."""
        return self._guard.slot(output) if guarded else None

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._flow.set_input(texture)

    def set_mask(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) the velocity is multiplied by with `use_mask`, or None for no mask."""
        self._mask = texture

    def update(self, depth: Texture | None, delta_time: float, view_span: float, depth_step: tuple[float, float],
               guarded: bool = False, depth_range: tuple[float, float] = (0.0, 0.0)) -> None:
        """Measure between the newest two frames.

        Args:
            depth: Meters in red, the camera's confidence in green (it sizes z's gate); None without depth (z is 0)
            delta_time: Seconds since the previous frame
            view_span: Metres one view width spans per metre of depth, for z
            depth_step: uv between two depth samples in the flow's view, for z
            guarded: Set the outputs' non-finite values to 0, count and log them (`guard`)
            depth_range: The player's depth range, near and far (m), for `z_position`
        """
        if not self._flow.ready:
            FlowUtil.zero(self._velocity)
            return
        s: LucasKanadeFlow3DSettings = self.settings
        xy_s: LucasKanadeFlowSettings = s.xy
        z_s: DepthVelocitySettings = s.z

        captured = self._capture.before(self._flow.previous_luminance, self._flow.luminance, xy_s.levels, depth) if self._capture else None
        self._flow.update()
        solved: Texture = self._flow.solved
        if self._capture is not None:
            self._capture.after(captured, solved, {'levels': xy_s.levels, 'window': int(xy_s.window), 'damping': xy_s.damping})

        if depth is not None:
            self._depth_velocity.update_depth(depth, delta_time, view_span, depth_step, z_s.window, z_s.smooth, z_s.halving_speed,
                                              z_s.rest_speed, z_s.threshold, z_s.edge, z_s.dead_zone, solved, z_s.max_speed,
                                              self._slot('depth_velocity', guarded), z_s.depth_noise / 1000.0)

        xy: Texture = self._flow.displacement
        if s.smooth_like_z:
            if not self._following:
                self._filtered.clear_all()                  # from no motion, not from when it was last on
                self._following = True
            self._filtered.swap()
            self._filtered.begin()
            self._filter_shader.use(solved, self._filtered.back, solved, z_s.smooth, z_s.rest_speed, z_s.halving_speed,
                                    max(delta_time, 1e-6), self._slot('displacement', guarded))
            self._filtered.end()
            self._followed.begin()
            self._threshold_shader.use(self._filtered, xy_s.threshold)
            self._followed.end()
            xy = self._followed
        else:
            self._following = False

        # uv per frame as image widths per second: v is uv of the height, so a square motion stays square;
        # z is in view widths per frame already
        rate: float = per_second(delta_time)
        aspect: float = self._velocity.height / max(self._velocity.width, 1)
        mask: Texture | None = self._mask if s.use_mask and self._mask is not None and self._mask.allocated else None
        self._velocity.begin()
        self._compose_shader.use(xy, self._depth_velocity.depth_velocity if depth is not None else None,
                                 (rate * s.x_gain, rate * aspect * s.y_gain, rate * s.z_gain), mask,
                                 self._slot('velocity', guarded))
        self._velocity.end()

        if depth is not None:
            self._z_position.begin()
            self._z_position_shader.use(self._depth_velocity.difference, depth_range[0], depth_range[1], mask)
            self._z_position.end()
        else:
            self._z_position.clear()
        self._guard.update(delta_time)

