"""Flow 3D Layer - the 3D optical flow (x/y and z from the depth) and the flow bridge's velocity/density/temperature."""

# Standard library imports
import logging
import math
import time     # TEMP diagnostic
from enum import IntEnum, auto
from typing import Protocol, runtime_checkable

# Third-party imports
from OpenGL.GL import GL_RGBA16F, GL_VIEWPORT, glGetIntegerv

# Local application imports
from modules.gl import Fbo, GpuTimer, Texture, Style, Blit, Text
from ..layer_base import LayerBase
from ...shaders import MaskMultiply
from modules.board import HasFrames

from modules.settings import Field, BaseSettings, Group
from modules.flow import (
    LucasKanadeFlow, LucasKanadeFlowSettings,
    VelocityTrailBridge, VelocityTrailBridgeSettings,
    KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    Visualizer, VisualizationFieldSettings
)

from modules.utils import HotReloadMethods

logger = logging.getLogger(__name__)

MAX_FRAME_INTERVAL: float = 0.5     # s: a longer gap between two captures is a pause or a loop, not a frame's
FRAME_PERIOD_BAND: tuple[float, float] = (0.5, 2.0)    # a capture gap is held to this share of the frame interval
FRAME_PERIOD_WEIGHT: float = 0.05   # weight of a new gap on the frame interval's Ema
FLASH: float = 5.0                  # TEMP diagnostic: a frame's mean |x/y| past this times the running median is logged


def frame_interval(timestamp: float | None, last: float | None, period: float | None, ticks: float,
                   time_step: float) -> tuple[float, float | None]:
    """The interval (s) a flow frame divides its motion by, and the frame period one frame on: the
    gap between the capture times `last` and `timestamp`, held to FRAME_PERIOD_BAND of `period` (an
    Ema of the gaps); a gap that is not a frame's (none, not finite, 0 or less, a pause or a loop)
    takes `period`, else `ticks` (the time counted in ticks), else `time_step`. Never 0, negative or
    not finite."""
    if period is not None and not (math.isfinite(period) and period > 0.0):
        period = None
    gap: float = timestamp - last if timestamp is not None and last is not None else math.nan
    if not (math.isfinite(gap) and 0.0 < gap < MAX_FRAME_INTERVAL):
        if period is not None:
            return period, period
        return (ticks if math.isfinite(ticks) and ticks > 0.0 else time_step), None
    if period is None:
        return gap, gap
    low, high = FRAME_PERIOD_BAND
    gap = min(max(gap, low * period), high * period)
    return gap, period + FRAME_PERIOD_WEIGHT * (gap - period)


@runtime_checkable
class HasCaptureTime(Protocol):
    """A flow source that keeps the capture time of the image it holds."""
    @property
    def timestamp(self) -> float | None: ...


class FlowBridgeMethod(IntEnum):
    """How the flow bridge makes the velocity the fluid reads and the amount the density reads."""
    VELOCITY_TRAIL = 0  # a smoothed velocity, density where there is speed
    KINETIC = auto()  # two trails, reading the change of the motion


class Flow3DDrawMode(IntEnum):
    """Draw modes for Flow3DLayer. A 3D velocity (xyz) draws as white x/y arrows with a ring of radius |z|
    on the arrow's scale, red toward the camera and green away; as scalars, x/y/z as red/green/blue around grey.

    Inputs:
        LUMINANCE - The newest luminance frame of the optical flow

    Depth (red toward the camera, green away, in px per frame: 1 is 76% color at scalar_scale 1):
        DEPTH_DIFFERENCE - The flow's tracked z, before the threshold
        DEPTH_VELOCITY - The flow's z after the threshold, before the z switch and the mask

    Outputs:
        VELOCITY - The flow's 3D velocity, times the mask with its use_mask
        SMOOTH_VELOCITY_INPUT - What the bridge reads: the velocity (shaped on the velocity trail bridge)
        SMOOTH_VELOCITY_OUTPUT - The bridge's smoothed 3D velocity, what the fluid gets
        SMOOTH_VELOCITY_DEPTH - Its z: red toward the camera, green away
        DENSITY_BRIDGE_INPUT_ALPHA - The amount the density's alpha is made from: speed (velocity trail) or emission (kinetic)
        DENSITY_BRIDGE_OUTPUT - Colored density (RGBA16F)
        TEMPERATURE_BRIDGE_OUTPUT - Temperature field (R16F)

    Kinetic bridge:
        KINETIC_FAST / KINETIC_SLOW - The two trails (RGBA16F, xyz)
        KINETIC_ACCELERATION - Along the motion in two colors: + onset warm, - release cold
        KINETIC_TURN - Across the motion in two colors
        KINETIC_HELD_RELEASE - The peak-held release
        KINETIC_RESERVOIR - Ink stored per pixel
        KINETIC_VELOCITY - The velocity the fluid gets, after the release's drop_velocity (RGBA16F, xyz)
        KINETIC_EMISSION - The ink the density bridge gets
        KINETIC_ONSET - The second velocity input (RGBA16F, xyz)
        KINETIC_CURL - The fast trail's 2D curl in two colors
        KINETIC_STRIP - Speed, onset, release and emission over the last seconds
        KINETIC_DEPTH_ACCELERATION - The z of fast - slow: red toward the camera, green away
    """
    LUMINANCE = 0
    VELOCITY = auto()
    SMOOTH_VELOCITY_INPUT = auto()
    SMOOTH_VELOCITY_OUTPUT = auto()
    SMOOTH_VELOCITY_MAGNITUDE = auto()
    DENSITY_BRIDGE_INPUT_COLOR = auto()
    DENSITY_BRIDGE_INPUT_ALPHA = auto()
    DENSITY_BRIDGE_OUTPUT = auto()
    TEMPERATURE_BRIDGE_INPUT_COLOR = auto()
    TEMPERATURE_BRIDGE_INPUT_MASK = auto()
    TEMPERATURE_BRIDGE_OUTPUT = auto()
    KINETIC_FAST = auto()
    KINETIC_SLOW = auto()
    KINETIC_ACCELERATION = auto()
    KINETIC_TURN = auto()
    KINETIC_HELD_RELEASE = auto()
    KINETIC_RESERVOIR = auto()
    KINETIC_VELOCITY = auto()
    KINETIC_EMISSION = auto()
    KINETIC_ONSET = auto()
    KINETIC_CURL = auto()
    KINETIC_STRIP = auto()
    SMOOTH_VELOCITY_DEPTH = auto()
    KINETIC_DEPTH_ACCELERATION = auto()
    DEPTH_DIFFERENCE = auto()
    DEPTH_VELOCITY = auto()


DEPTH_DRAW_MODES: set[Flow3DDrawMode] = {     # red and green
    Flow3DDrawMode.SMOOTH_VELOCITY_DEPTH, Flow3DDrawMode.KINETIC_DEPTH_ACCELERATION, Flow3DDrawMode.DEPTH_DIFFERENCE, Flow3DDrawMode.DEPTH_VELOCITY,
}
PIXEL_DRAW_MODES: set[Flow3DDrawMode] = {Flow3DDrawMode.DEPTH_DIFFERENCE, Flow3DDrawMode.DEPTH_VELOCITY}     # view widths per frame, shown in px per frame
VECTOR_DRAW_MODES: set[Flow3DDrawMode] = {       # 3D velocities, drawn as velocities whatever the texture's format
    Flow3DDrawMode.VELOCITY, Flow3DDrawMode.SMOOTH_VELOCITY_INPUT, Flow3DDrawMode.SMOOTH_VELOCITY_OUTPUT,
    Flow3DDrawMode.KINETIC_FAST, Flow3DDrawMode.KINETIC_SLOW, Flow3DDrawMode.KINETIC_VELOCITY, Flow3DDrawMode.KINETIC_ONSET,
}


class FlowBridgeSettings(BaseSettings):
    """How the flow bridge turns the optical flow into the fluid's velocity, density and temperature."""
    method: Field[FlowBridgeMethod] =   Field(FlowBridgeMethod.VELOCITY_TRAIL, description="How the bridge makes the fluid's velocity and the density's amount: one trail, or two trails reading the change of the motion")

    velocity_trail = Group(VelocityTrailBridgeSettings)
    kinetic     = Group(KineticBridgeSettings)
    density     = Group(DensityBridgeSettings)
    temperature = Group(TemperatureBridgeSettings)


class Flow3DLayerSettings(BaseSettings):
    """Configuration for Flow3DLayer (optical flow + flow bridge)."""
    draw_mode: Field[Flow3DDrawMode] =      Field(Flow3DDrawMode.DENSITY_BRIDGE_INPUT_COLOR)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")
    stamped_interval: Field[bool] =         Field(True, description="TEMP: frame interval from capture stamps (off: render ticks)")

    optical_flow  = Group(LucasKanadeFlowSettings)
    flow_bridge   = Group(FlowBridgeSettings)
    visualization = Group(VisualizationFieldSettings)


class Flow3DLayer(LayerBase):
    """Flow processing layer with the optical flow and the flow bridge.

    Measures the motion and prepares velocity/density/temperature for fluid simulation:
    1. LucasKanadeFlow: the 3D velocity (xyz), z from the depth
    2. VelocityTrailBridge or KineticBridge (settings.flow_bridge.method): the smoothed velocity for the fluid
    3. DensityBridge: Combines RGB density with an amount as its alpha
    4. TemperatureBridge: Generates temperature field

    Output textures for Fluid3DLayer:
        - smooth_velocity: Smoothed velocity, xyz (RGBA16F)
        - density: Colored density (RGBA16F)
        - temperature: Temperature field (R16F)

    Usage:
        flow = Flow3DLayer(cam_id, board, flow_source, flow_input, density_input, settings)
        flow.update()
        # Access outputs for fluid simulation:
        fluid.add_velocity(flow.smooth_velocity)
        fluid.add_density(flow.density)
        fluid.add_temperature(flow.temperature)
    """

    def __init__(self, cam_id: int, board: HasFrames, flow_source: LayerBase, flow_input: Texture, density_input: Texture, settings: Flow3DLayerSettings | None = None) -> None:
        """Initialize flow layer.

        Args:
            cam_id: Camera ID
            board: Board for pose data
            flow_source: The source layer behind the flow input; the flow measures when it has a new frame
            flow_input: Texture for optical flow input
            density_input: Texture for density and temperature input
            settings: Layer configuration
        """
        self._cam_id: int = cam_id
        self._board: HasFrames = board
        self._flow_source: LayerBase = flow_source
        self._flow_input: Texture = flow_input
        self._density_input: Texture = density_input
        self.settings: Flow3DLayerSettings = settings or Flow3DLayerSettings()

        bridge: FlowBridgeSettings = self.settings.flow_bridge
        self._flow: LucasKanadeFlow = LucasKanadeFlow(self.settings.optical_flow)
        self._velocity: Texture = self._flow.velocity
        self._depth_input: Texture | None = None
        self._mask_input: Texture | None = None
        self._gate_shader: MaskMultiply = MaskMultiply()
        self._gated_color: Fbo = Fbo()      # the density input times the mask
        self._velocity_trail_bridge: VelocityTrailBridge = VelocityTrailBridge(bridge.velocity_trail)
        self._kinetic_bridge: KineticBridge = KineticBridge(bridge.kinetic)
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._density_bridge: DensityBridge = DensityBridge(bridge.density)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(bridge.temperature)

        self._visualizer: Visualizer = Visualizer(self.settings.visualization)
        self._kinetic_strip: KineticStrip = KineticStrip()
        self._strip_text: Text = Text()

        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")

        self._time_step: float = 1.0 / 60.0     # dt of one tick
        self._view_span: float = 0.0            # metres a view width spans per metre of depth; 0 reads no z
        self._depth_step: tuple[float, float] = (0.0, 0.0)     # uv between two depth samples; 0 takes one flow texel
        self._warned_view_span: bool = False
        self._flow_interval: float = 0.0        # time since the last optical flow frame (s), counted in ticks; on a new one, its capture interval
        self._capture_timestamp: float | None = None    # the capture time of the last image the flow measured
        self._frame_period: float | None = None         # the frame interval, an Ema of the capture gaps (s)

        self._flow_size: tuple[int, int] = (0, 0)       # the size the flow runs at; follows settings.width/height
        self._output_size: tuple[int, int] = (0, 0)     # the layer's allocation size

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Visualization output texture."""
        return self._visualizer.texture

    @property
    def smooth_velocity(self) -> Texture:
        """Smoothed velocity, xyz (RGBA16F), for fluid simulation; z is 0 without depth."""
        if self.settings.flow_bridge.method == FlowBridgeMethod.KINETIC:
            return self._kinetic_bridge.smooth_velocity
        return self._velocity_trail_bridge.smooth_velocity

    @property
    def depth_enabled(self) -> bool:
        """Whether the optical flow measures z: a source has depth. The flow's `z` switch only zeroes it in the velocity."""
        return self._depth_input is not None and self._depth_input.allocated

    @property
    def onset_velocity(self) -> Texture | None:
        """Second velocity input, xyz (RGBA16F), for fluid simulation; None when the bridge has none."""
        bridge: FlowBridgeSettings = self.settings.flow_bridge
        if bridge.method == FlowBridgeMethod.KINETIC and bridge.kinetic.onset.force > 0.0:
            return self._kinetic_bridge.onset_velocity
        return None

    @property
    def spin(self) -> Texture | None:
        """2D curl (R16F) for the fluid's vorticity force; None when the bridge has none."""
        bridge: FlowBridgeSettings = self.settings.flow_bridge
        if bridge.method == FlowBridgeMethod.KINETIC and bridge.kinetic.onset.spin > 0.0:
            return self._kinetic_bridge.curl
        return None

    @property
    def density(self) -> Texture:
        """Colored density output (RGBA16F) for fluid simulation."""
        return self._density_bridge.density

    @property
    def magnitude(self) -> Texture:
        """The smoothed velocity's 3D speed (R32F) for fluid simulation."""
        return self._velocity_magnitude.magnitude

    @property
    def temperature(self) -> Texture:
        """Temperature output (R16F) for fluid simulation."""
        return self._temperature_bridge.temperature

    # ========== Input Setters ==========

    def set_flow_input(self, texture: Texture) -> None:
        self._flow_input = texture

    def set_flow_source(self, layer: LayerBase) -> None:
        self._flow_source = layer

    def set_density_input(self, texture: Texture) -> None:
        self._density_input = texture

    def set_mask_input(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) on the flow's view, or None for no gate. The flow multiplies
        its 3D velocity by it with its use_mask, and the density color is gated by it, after they are
        measured, so they run on the whole image: a jittery mask edge then only scales what passes, it
        never reads as motion, and needs no blending."""
        self._mask_input = texture
        self._flow.set_mask(texture)

    def set_depth_input(self, texture: Texture) -> None:
        """Depth for the velocity toward the camera: metres in red, 0 where there is none, on the flow's
        view; `set_mask_input` gates the result to the player."""
        self._depth_input = texture

    def set_time_step(self, dt: float) -> None:
        """The dt of one tick, polled by the kinetic bridge; the fluid's step, so a fixed step compares runs."""
        self._time_step = dt

    def set_view_span(self, span: float) -> None:
        """The metres one width of the flow's view spans per metre of depth, polled: it puts z in view
        widths like x/y. Set it per tick, as the view can follow the pose."""
        self._view_span = span

    def set_depth_step(self, step: tuple[float, float]) -> None:
        """The uv between two of the depth's own samples in the flow's view, polled: z's window steps by
        it, so it covers real samples, not copies. Set it per tick, as the view can follow the pose."""
        self._depth_step = step

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate all processing stages.

        Args:
            width: Output width (density, visualization)
            height: Output height
            internal_format: Ignored (formats determined by each stage)
        """
        self._output_size = (width, height)
        self._allocate_flow()
        self._gated_color.allocate(width, height, GL_RGBA16F)
        self._gate_shader.allocate()
        self._density_bridge.allocate(width, height)

        self._visualizer.allocate(width, height)
        self._kinetic_strip.allocate()
        self._strip_text.allocate(font_size=18)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._deallocate_flow()
        self._gated_color.deallocate()
        self._gate_shader.deallocate()
        self._density_bridge.deallocate()
        self._visualizer.deallocate()
        self._kinetic_strip.deallocate()
        self._strip_text.deallocate()
        self._gpu_timer.deallocate()

    def _allocate_flow(self) -> None:
        """Allocate the stages that run at the flow's size (settings.width/height)."""
        flow_width, flow_height = self._flow_size = (self.settings.width, self.settings.height)
        output_width, output_height = self._output_size

        self._flow.allocate(flow_width, flow_height)
        self._velocity_trail_bridge.allocate(flow_width, flow_height, output_width, output_height)
        self._kinetic_bridge.allocate(flow_width, flow_height)
        self._velocity_magnitude.allocate(flow_width, flow_height)
        self._temperature_bridge.allocate(flow_width, flow_height)

    def _deallocate_flow(self) -> None:
        self._flow.deallocate()
        self._velocity_trail_bridge.deallocate()
        self._kinetic_bridge.deallocate()
        self._velocity_magnitude.deallocate()
        self._temperature_bridge.deallocate()
        self._flow_size = (0, 0)

    def _follow_flow_size(self) -> None:
        """Reallocate the flow's stages when settings.width/height changed, and start them from rest."""
        if self._flow_size == (0, 0) or self._flow_size == (self.settings.width, self.settings.height):
            return
        self._deallocate_flow()
        self._allocate_flow()
        self.reset()

    def reset(self) -> None:
        """Reset all processing stages."""
        self._flow.reset()
        self._velocity_trail_bridge.reset()
        self._kinetic_bridge.reset()
        self._density_bridge.reset()
        self._kinetic_strip.reset()
        self._flow_interval = 0.0
        self._capture_timestamp = None

    def _frame_interval(self) -> float:
        """Seconds between the image the flow measures now and the one before, from the capture times
        the flow source keeps with its images (`HasCaptureTime`): counted in ticks, a frame that lands
        a tick early or late reads a third shorter or longer, and its speed jumps by as much. A gap is
        held to FRAME_PERIOD_BAND of the frame interval, an Ema of the gaps, so one odd stamp cannot
        flash; where there is no gap of a frame (the first frame, after a reset, a recording that
        loops, a source without capture times), the frame interval, or the tick count before there is one."""
        timestamp: float | None = self._flow_source.timestamp if isinstance(self._flow_source, HasCaptureTime) else None
        last: float | None = self._capture_timestamp
        if timestamp is not None and math.isfinite(timestamp):
            self._capture_timestamp = timestamp
        interval, self._frame_period = frame_interval(timestamp, last, self._frame_period, self._flow_interval, self._time_step)
        self._diagnose(timestamp - last if timestamp is not None and last is not None else None, interval, timestamp)   # TEMP diagnostic
        return interval

    def _probe_flash(self, stamped: float) -> None:
        """TEMP diagnostic: each flow frame, the mean |x| and |y| of the velocity over an 8×8 blit; a
        frame past FLASH times the running median logs its timing and the frame's before it."""
        import numpy as np
        from OpenGL.GL import GL_FLOAT, GL_RGBA, GL_RGBA32F, glReadPixels
        from modules.flow import FlowUtil
        if not hasattr(self, '_flash_fbo'):
            self._flash_fbo: Fbo = Fbo()
            self._flash_fbo.allocate(8, 8, GL_RGBA32F)
            self._flash_means: list[float] = []
            self._flash_last: dict | None = None
        FlowUtil.blit(self._flash_fbo, self._velocity)
        self._flash_fbo.begin()
        pixels = np.asarray(glReadPixels(0, 0, 8, 8, GL_RGBA, GL_FLOAT), np.float32).reshape(8, 8, 4)
        self._flash_fbo.end()
        mean_x, mean_y, mean_z = (float(np.mean(np.abs(pixels[..., c]))) for c in range(3))
        capture = getattr(self._board, 'get_capture', lambda cam_id: None)(self._cam_id)
        info: dict = dict(stamp=self._capture_timestamp, interval_used=self._flow_interval, stamped=stamped,
                          ticks_before=self._diagnostic['ticks'][-1] / 1000.0 if hasattr(self, '_diagnostic') and self._diagnostic['ticks'] else None,
                          time_step=self._time_step, board_stamp=capture.timestamp if capture is not None else None,
                          levels=self.settings.optical_flow.lucas_kanade.levels, x=mean_x, y=mean_y, z=mean_z,
                          period=self._frame_period, finite=bool(np.all(np.isfinite(pixels))))
        means: list[float] = self._flash_means
        median: float = sorted(means)[len(means) // 2] if len(means) >= 10 else 0.0
        if median > 0.0 and max(mean_x, mean_y) > FLASH * median:
            logger.info("FLOW FLASH cam %d | now %s | before %s", self._cam_id,
                        {k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()},
                        {k: (round(v, 5) if isinstance(v, float) else v) for k, v in (self._flash_last or {}).items()})
        means.append(max(mean_x, mean_y))
        if len(means) > 200:
            means.pop(0)
        self._flash_last = info

    def _diagnose(self, gap: float | None, interval: float, timestamp: float | None) -> None:
        """TEMP diagnostic: once a second, the flow intervals over that second and the raw stamp gaps, to the log."""
        if not hasattr(self, '_diagnostic'):
            self._diagnostic: dict = {'start': time.perf_counter(), 'intervals': [], 'gaps': [], 'clamped': 0, 'none': 0, 'ticks': [], 'stamps': set()}
        d: dict = self._diagnostic
        d['intervals'].append(interval * 1000.0)
        d['ticks'].append(self._flow_interval * 1000.0)
        if gap is None:
            d['none'] += 1
        else:
            d['gaps'].append(gap * 1000.0)
            if abs(gap - interval) > 1e-9:
                d['clamped'] += 1
        if timestamp is not None:
            d['stamps'].add(timestamp)
        now: float = time.perf_counter()
        if now - d['start'] < 1.0:
            return

        def stats(values: list[float]) -> str:
            if not values:
                return 'none'
            v = sorted(values)
            return f'n {len(v)} min {v[0]:.1f} median {v[len(v) // 2]:.1f} max {v[-1]:.1f}'

        logger.info("FLOW TIMING cam %d | interval %s | raw gap %s | ticks %s | clamped %d, no gap %d | distinct stamps %d of %d flow updates",
                    self._cam_id, stats(d['intervals']), stats(d['gaps']), stats(d['ticks']), d['clamped'], d['none'],
                    len(d['stamps']), len(d['intervals']))
        self._diagnostic = {'start': now, 'intervals': [], 'gaps': [], 'clamped': 0, 'none': 0, 'ticks': [], 'stamps': set()}

    # ========== Processing ==========

    def update(self) -> None:
        """Update optical flow and bridges."""
        self._follow_flow_size()

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._gpu_timer.begin()

        new_flow: bool = self._flow_source.dirty
        self._flow_interval += self._time_step

        mask: Texture | None = self._mask_input if self._mask_input is not None and self._mask_input.allocated else None
        kinetic: bool = self.settings.flow_bridge.method == FlowBridgeMethod.KINETIC
        flow: LucasKanadeFlow = self._flow
        if new_flow:
            stamped: float = self._frame_interval()
            if self.settings.stamped_interval:     # TEMP switch: off keeps the tick count, as before
                self._flow_interval = stamped
            # Stage 1: the flow's 3D velocity, on the whole image; with its use_mask, times the mask
            flow.set_input(self._flow_input)
            if self.depth_enabled and self._view_span <= 0.0 and not self._warned_view_span:
                self._warned_view_span = True
                logger.warning("Flow3DLayer %d: depth is on but no view span is set (set_view_span), so z reads no motion", self._cam_id)
            flow.update(self._depth_input if self.depth_enabled else None, self._flow_interval, self._view_span, self._depth_step)
            self._velocity = flow.velocity
            self._probe_flash(stamped)   # TEMP diagnostic

        # Stage 2: The flow bridge
        if kinetic:
            if new_flow:
                # Lucas–Kanade's velocity is a distance, so the trails can travel with it
                self._kinetic_bridge.set_velocity(self._velocity)
                self._kinetic_bridge.update(self._flow_interval)
                if self.settings.draw_mode == Flow3DDrawMode.KINETIC_DEPTH_ACCELERATION:
                    self._kinetic_bridge.update_depth_acceleration()
                if self.settings.flow_bridge.kinetic.onset.spin > 0.0 or self.settings.draw_mode == Flow3DDrawMode.KINETIC_CURL:
                    self._kinetic_bridge.update_curl()
                if self.settings.draw_mode == Flow3DDrawMode.KINETIC_STRIP:
                    bridge: KineticBridge = self._kinetic_bridge
                    self._kinetic_strip.update(bridge.fast, bridge.acceleration, bridge.emission, self.settings.visualization.strip_scale)
        else:
            self._velocity_trail_bridge.set_velocity(self._velocity)
            self._velocity_trail_bridge.update(self._time_step)

        if new_flow:
            self._flow_interval = 0.0

        self._velocity_magnitude.set_input(self.smooth_velocity)
        self._velocity_magnitude.update()

        # The amount the density bridge makes its alpha from: the kinetic bridge's emission, or the trail's speed
        alpha: Texture = self._kinetic_bridge.emission if kinetic else self._velocity_magnitude.magnitude

        # The density color, gated like the velocity: where the bridges spread velocity past the mask,
        # no background color comes along
        density_color: Texture = self._density_input
        if mask is not None and self._density_input.allocated:
            self._gate(self._density_input, mask, self._gated_color)
            density_color = self._gated_color

        self._density_bridge.set_color(density_color)
        self._density_bridge.set_alpha(alpha)
        self._density_bridge.update()

        self._temperature_bridge.set_color(density_color)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()

        self._gpu_timer.end()

        Style.set_blend_mode(Style.BlendMode.ALPHA)

        signed: tuple[Texture, int] | None = self._get_signed_texture()
        if signed is not None:
            unit: float = float(self._flow_size[0]) if self.settings.draw_mode in PIXEL_DRAW_MODES else 1.0
            self._visualizer.process_signed(signed[0], self.settings.draw_mode in DEPTH_DRAW_MODES, signed[1], unit)
        else:
            self._visualizer.process(self._get_draw_texture(), self.settings.draw_mode in VECTOR_DRAW_MODES)
        Style.pop_style()

    def _gate(self, source: Texture, mask: Texture, target: Fbo) -> None:
        """`source` times the mask's red into `target`."""
        target.begin()
        self._gate_shader.use(source, mask)
        target.end()

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        Style.push_style()
        Style.set_blend_mode(self.settings.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        if self.settings.draw_mode == Flow3DDrawMode.KINETIC_STRIP and self._strip_text.allocated:
            self._draw_strip_labels()
        Style.pop_style()

    def _draw_strip_labels(self) -> None:
        """The name of each lane of the kinetic strip, at its top left."""
        vp = glGetIntegerv(GL_VIEWPORT)
        w, h = int(vp[2]), int(vp[3])
        if w <= 0 or h <= 0:
            return
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for lane, name in enumerate(KINETIC_STRIP_LANES):
            self._strip_text.draw_box_text(8, lane * h // len(KINETIC_STRIP_LANES) + 8, name, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), w, h)

    # ========== Texture Selection ==========

    def _get_signed_texture(self) -> tuple[Texture, int] | None:
        """The scalar field to draw in two colors and its channel, None when draw_mode is another kind of texture."""
        textures: dict[Flow3DDrawMode, tuple[Texture, int]] = {
            Flow3DDrawMode.KINETIC_ACCELERATION: (self._kinetic_bridge.acceleration, 0),
            Flow3DDrawMode.KINETIC_TURN: (self._kinetic_bridge.turn, 0),
            Flow3DDrawMode.KINETIC_HELD_RELEASE: (self._kinetic_bridge.held_release, 0),
            Flow3DDrawMode.KINETIC_RESERVOIR: (self._kinetic_bridge.reservoir, 0),
            Flow3DDrawMode.KINETIC_EMISSION: (self._kinetic_bridge.emission, 0),
            Flow3DDrawMode.KINETIC_CURL: (self._kinetic_bridge.curl, 0),
            Flow3DDrawMode.SMOOTH_VELOCITY_DEPTH: (self.smooth_velocity, 2),       # its z
            Flow3DDrawMode.DEPTH_DIFFERENCE: (self._flow.depth_velocity.difference, 0),
            Flow3DDrawMode.DEPTH_VELOCITY: (self._flow.depth_velocity.depth_velocity, 0),
            Flow3DDrawMode.KINETIC_DEPTH_ACCELERATION: (self._kinetic_bridge.depth_acceleration, 0),
            Flow3DDrawMode.DENSITY_BRIDGE_INPUT_ALPHA: (self._density_bridge.alpha_input, 0),
        }
        return textures.get(self.settings.draw_mode)

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        textures = {
            Flow3DDrawMode.LUMINANCE: self._flow.luminance,
            Flow3DDrawMode.VELOCITY: self._velocity,
            Flow3DDrawMode.SMOOTH_VELOCITY_INPUT: self._kinetic_bridge.velocity_input if self.settings.flow_bridge.method == FlowBridgeMethod.KINETIC else self._velocity_trail_bridge.velocity_input,
            Flow3DDrawMode.SMOOTH_VELOCITY_OUTPUT: self.smooth_velocity,
            Flow3DDrawMode.SMOOTH_VELOCITY_MAGNITUDE: self._velocity_magnitude.magnitude,
            Flow3DDrawMode.DENSITY_BRIDGE_INPUT_COLOR: self._density_bridge.color_input,
            Flow3DDrawMode.DENSITY_BRIDGE_OUTPUT: self._density_bridge.density,
            Flow3DDrawMode.TEMPERATURE_BRIDGE_INPUT_COLOR: self._temperature_bridge.color_input,
            Flow3DDrawMode.TEMPERATURE_BRIDGE_INPUT_MASK: self._temperature_bridge.mask_input,
            Flow3DDrawMode.TEMPERATURE_BRIDGE_OUTPUT: self._temperature_bridge.temperature,
            Flow3DDrawMode.KINETIC_FAST: self._kinetic_bridge.fast,
            Flow3DDrawMode.KINETIC_SLOW: self._kinetic_bridge.slow,
            Flow3DDrawMode.KINETIC_VELOCITY: self._kinetic_bridge.smooth_velocity,
            Flow3DDrawMode.KINETIC_ONSET: self._kinetic_bridge.onset_velocity,
            Flow3DDrawMode.KINETIC_STRIP: self._kinetic_strip.texture,
        }
        return textures.get(self.settings.draw_mode, self._density_input)
