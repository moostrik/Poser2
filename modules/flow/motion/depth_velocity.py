"""Depth Velocity: the player's motion toward the camera, z, from the depth image, in view widths per frame:
per pixel a disparity and its rate, carried along the x/y displacement and corrected by each measurement
(an alpha–beta filter) that passes its gate, over a time that shortens as x/y moves (the 1€ filter's), with a
step filter before it and a dead zone after it. How and why:
modules/flow/MOTION.md, "z: the depth velocity".
"""
from OpenGL.GL import GL_NEAREST, GL_R16F, GL_RG32F, GL_RGBA32F

from modules.gl import Fbo, MultiTargetFbo, SwapMultiTargetFbo, SwapTarget, Texture
from modules.settings import BaseSettings, Field
from .. import FlowBase, FlowUtil, Port, TextureKind
from .patch import Patch
from ..finite_guard import GuardSlot
from .shaders import DepthStepFilter, DepthThreshold, DepthVelocityShader


class DepthVelocitySettings(BaseSettings):
    """The z part: the depth velocity, a depth track carried along the displacement. Its panel in the order the code
    runs it, a line each: the surface, the smoothing, the cut."""
    edge:           Field[float] = Field(0.1, min=0.05, max=0.3, step=0.01, description="z: a depth step past this, in m, is another surface (lower: cleaner edges)")
    window:         Field[Patch] = Field(Patch.SEVEN, min=Patch.ONE, max=Patch.NINE, description="z: depth samples averaged (larger: less noise, small parts blur)")
    depth_noise:    Field[float] = Field(10.0, min=1.0, max=50.0, step=0.5, description="z: the camera's depth noise at 1 m, in mm, at full confidence (higher: fewer restarts)")
    max_speed:      Field[float] = Field(6.0, min=0.0, max=20.0, step=0.5, description="z: faster than this, in m/s, is another surface, the track starts over (0: no check)")
    smooth:         Field[float] = Field(100.0, min=0.0, max=500.0, step=1.0, newline=True, description="z: smoothing time at rest, in ms; 0 the change of two frames (higher: less noise at rest)")
    rest_speed:     Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, description="speed below which the motion counts as at rest, in px/frame")
    halving_speed:  Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, description="speed past rest_speed at which the smoothing time is half, in px/frame (0: smooth throughout)")
    dead_zone:      Field[float] = Field(0.0, min=0.0, max=0.05, step=0.001, newline=True, description="z: depth drift ignored before z counts as motion, in m (higher: less drift, slower start)")
    threshold:      Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, description="z below this is cut as noise, in px/frame (higher: less noise, less slow z)")


class DepthVelocity(FlowBase):
    """Signed motion toward the camera in view widths per frame (R16F), positive toward it.

    Pipeline, on each new flow frame:
        1. Take the depth and its confidence → the raw Fbo (RG32F, meters and 0 to 1, at the flow's size, nearest),
           and without its edges → the input SwapFbo
        2. Correct the track carried from the last one → the track, a SwapMultiTargetFbo drawn at once: the estimate
           (RGBA32F: z, disparity, rate, anchor) and its misses in a row (R16F)
        3. Take the noise floor off z → output_fbo (R16F)
    """

    def __init__(self) -> None:
        super().__init__()

        self._input_internal_format = GL_RG32F
        self._output_internal_format = GL_R16F
        self._input_kind = TextureKind.DEPTH
        self._output_kind = TextureKind.SCALAR

        self._step_filter_shader: DepthStepFilter = DepthStepFilter()
        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._threshold_shader: DepthThreshold = DepthThreshold()

        self._raw: Fbo = Fbo()                   # the depth and its confidence as they came, before the edges are taken off
        # The track, swapped as one: the estimate (target 0) and its misses in a row (target 1), drawn at once
        self._track: SwapMultiTargetFbo = SwapMultiTargetFbo(MultiTargetFbo(Texture(), Texture(GL_NEAREST)),
                                                             MultiTargetFbo(Texture(), Texture(GL_NEAREST)))
        self._estimate: SwapTarget = self._track.target(0)
        self._misses: SwapTarget = self._track.target(1)

    @property
    def depth_velocity(self) -> Texture:
        """Motion toward the camera in view widths per frame (R16F)."""
        return self._output

    @property
    def difference(self) -> Texture:
        """The estimate (RGBA32F): z before the threshold in red (view widths per frame), its disparity
        (1/m) in green, the disparity's rate (1/m per second) in blue and the dead zone's anchor (m) in alpha."""
        return self._estimate

    @property
    def misses(self) -> Texture:
        """The track's misses in a row (R16F): 0 where it follows its measurement, 1 where it coasts."""
        return self._misses

    @property
    def ports(self) -> dict[str, Port]:
        """`input` is the depth with its edges taken off, `raw` as it came."""
        return {'raw': Port(self._raw, TextureKind.DEPTH), **super().ports,
                'estimate': Port(self._estimate, TextureKind.SCALAR)}

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        super().allocate(width, height, output_width, output_height)
        self._raw.allocate(width, height, GL_RG32F)
        self._track.allocate(width, height, (GL_RGBA32F, GL_R16F))
        self._clear_track()
        self._step_filter_shader.allocate()
        self._velocity_shader.allocate()
        self._threshold_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._raw.deallocate()
        self._track.deallocate()
        self._step_filter_shader.deallocate()
        self._velocity_shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        super().reset()
        self._clear_track()

    def _clear_track(self) -> None:
        """No track anywhere: the estimate and its misses 0."""
        self._estimate.clear_all()
        self._misses.clear_all()

    def update_depth(self, depth: Texture, delta_time: float, view_span: float, depth_step: tuple[float, float], window: int,
                     smooth: float, halving_speed: float, rest_speed: float, threshold: float, edge: float, dead_zone: float,
                     displacement: Texture | None = None, max_speed: float = 0.0, guard: GuardSlot | None = None,
                     depth_noise: float = 0.01) -> None:
        """Advance on a new flow frame: correct the track by the new depth, then take the noise floor off.

        Args:
            depth: Meters in red, 0 where there is none; the camera's confidence (0 to 1) in green, 0 where the view filled it
            delta_time: Seconds since the last depth
            view_span: Meters one view width spans per meter of depth; 0 or less reads no motion
            depth_step: uv between two depth samples in the flow's view; 0 takes one flow texel
            window: Radius in depth samples
            smooth: Time over which the track averages at rest (ms); 0 is the change of two frames
            halving_speed: speed (the larger of x/y's and the track's own) past `rest_speed` at which that time is half (px per frame); 0 is `smooth` throughout
            rest_speed: speed below which the motion is at rest (px per frame)
            threshold: Depth motion below this is noise, taken off the rest (px per frame)
            edge: Depth step to another surface within a frame (m): a 3×3 of depth samples spanning it is
                dropped (the step filter), and the window averages only the samples within it
            dead_zone: Depth the track wanders within its anchor without z (m); 0 is none
            displacement: How far each pixel moved since then (uv); when given, the track is carried
                along the motion (Lucas–Kanade flow)
            max_speed: The fastest a body part moves in depth (m/s): the gate's motion, with its noise; a measurement
                outside the gate is a miss, two in a row start the track over, and a speed past it does at once; 0 is no gate
            guard: The depth velocity's slot in the motion's FiniteGuard: a non-finite track 0 and counted; None no guard
            depth_noise: The camera's depth noise at 1 m depth and full confidence (m), the gate's noise
        """
        if not self._allocated:
            return
        width, height = self._input_fbo.width, self._input_fbo.height
        step: tuple[float, float] = depth_step if depth_step[0] > 0.0 and depth_step[1] > 0.0 else (1.0 / width, 1.0 / height)

        FlowUtil.blit(self._raw, depth)
        self._input_fbo.swap()
        self._input_fbo.begin()
        self._step_filter_shader.use(self._raw, step, edge)
        self._input_fbo.end()

        self._track.swap()
        self._track.current.begin()          # the estimate and its misses
        self._velocity_shader.use(self._input_fbo, self._estimate.back, smooth, halving_speed, rest_speed, max(delta_time, 1e-6),
                                  view_span, step, window, edge, dead_zone, self._misses.back,
                                  depth_noise, displacement, max_speed, guard)
        self._track.current.end()

        self._output_fbo.begin()
        self._threshold_shader.use(self._estimate, threshold / max(width, 1))     # px as view widths
        self._output_fbo.end()

