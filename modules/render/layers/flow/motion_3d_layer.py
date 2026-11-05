"""Flow 3D Layer - the 3D optical flow (x/y and z from the depth) and the flow bridge's velocity/density/temperature."""

# Standard library imports
import logging
import math
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
from .motion_layer import frame_interval

logger = logging.getLogger(__name__)


@runtime_checkable
class HasCaptureTime(Protocol):
    """A flow source that keeps the capture time of the image it holds."""
    @property
    def timestamp(self) -> float | None: ...


class FlowBridgeMethod(IntEnum):
    """How the flow bridge makes the velocity the fluid reads and the amount the density reads."""
    VELOCITY_TRAIL = 0  # a smoothed velocity, density where there is speed
    KINETIC = auto()  # two trails, reading the change of the motion


class Motion3DDrawMode(IntEnum):
    """Draw modes for Motion3DLayer. A 3D velocity (xyz) draws as white x/y arrows with a ring of radius |z|
    on the arrow's scale, red toward the camera and green away; as scalars, x/y/z as red/green/blue around grey.

    In pipeline order, each named after the panel group and the property it shows. z has no mode of its own:
    it is the ring on every velocity's arrows.

    Inputs:
        INPUT_2D - The image the optical flow reads (inputs.flow)
        INPUT_DEPTH - The depth image, metres
        INPUT_MASK - The player's mask (inputs.mask)

    Optical flow:
        OPTICAL_FLOW_VELOCITY - The flow's 3D velocity, times its gains and the mask

    Outputs, what the fluid gets:
        VELOCITY - The active bridge's smoothed 3D velocity
        DENSITY - The colored density (RGBA16F)
        TEMPERATURE - The temperature field, in two colors

    Kinetic bridge (only while it runs):
        KINETIC_FAST / KINETIC_SLOW - The two trails (RGBA16F, xyz)
        KINETIC_ACCELERATION - fast - slow along the motion in two colors: + onset warm, - release cold
        KINETIC_DEPTH_ACCELERATION - The z of fast - slow: red toward the camera, green away
        KINETIC_TURN - fast - slow across the motion in two colors
        KINETIC_HELD_RELEASE - The peak-held release
        KINETIC_RESERVOIR - Density stored per pixel
        KINETIC_EMISSION - The density amount it emits
        KINETIC_ONSET_VELOCITY - The second velocity input (RGBA16F, xyz)
        KINETIC_CURL - The fast trail's 2D curl in two colors
        KINETIC_STRIP - Speed, onset, release and emission over the last seconds
    """
    INPUT_2D = 0
    INPUT_DEPTH = auto()
    INPUT_MASK = auto()
    OPTICAL_FLOW_VELOCITY = auto()
    VELOCITY = auto()
    DENSITY = auto()
    TEMPERATURE = auto()
    KINETIC_FAST = auto()
    KINETIC_SLOW = auto()
    KINETIC_ACCELERATION = auto()
    KINETIC_DEPTH_ACCELERATION = auto()
    KINETIC_TURN = auto()
    KINETIC_HELD_RELEASE = auto()
    KINETIC_RESERVOIR = auto()
    KINETIC_EMISSION = auto()
    KINETIC_ONSET_VELOCITY = auto()
    KINETIC_CURL = auto()
    KINETIC_STRIP = auto()


DEPTH_DRAW_MODES: set[Motion3DDrawMode] = {Motion3DDrawMode.KINETIC_DEPTH_ACCELERATION}     # red toward the camera, green away
VECTOR_DRAW_MODES: set[Motion3DDrawMode] = {       # 3D velocities, drawn as velocities whatever the texture's format
    Motion3DDrawMode.OPTICAL_FLOW_VELOCITY, Motion3DDrawMode.VELOCITY,
    Motion3DDrawMode.KINETIC_FAST, Motion3DDrawMode.KINETIC_SLOW, Motion3DDrawMode.KINETIC_ONSET_VELOCITY,
}


class FlowBridgeSettings(BaseSettings):
    """How the flow bridge turns the optical flow into the fluid's velocity, density and temperature."""
    method: Field[FlowBridgeMethod] =   Field(FlowBridgeMethod.VELOCITY_TRAIL, description="How the bridge makes the fluid's velocity and the density's amount: one trail, or two trails reading the change of the motion")

    velocity_trail = Group(VelocityTrailBridgeSettings)
    kinetic     = Group(KineticBridgeSettings)
    density     = Group(DensityBridgeSettings)
    temperature = Group(TemperatureBridgeSettings)


class Motion3DLayerSettings(BaseSettings):
    """Configuration for Motion3DLayer (optical flow + flow bridge)."""
    draw_mode: Field[Motion3DDrawMode] =      Field(Motion3DDrawMode.INPUT_2D)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")

    optical_flow  = Group(LucasKanadeFlowSettings)
    flow_bridge   = Group(FlowBridgeSettings)
    visualization = Group(VisualizationFieldSettings)


class Motion3DLayer(LayerBase):
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
        flow = Motion3DLayer(cam_id, board, flow_source, flow_input, density_input, settings)
        flow.update()
        # Access outputs for fluid simulation:
        fluid.add_velocity(flow.smooth_velocity)
        fluid.add_density(flow.density)
        fluid.add_temperature(flow.temperature)
    """

    def __init__(self, cam_id: int, board: HasFrames, flow_source: LayerBase, flow_input: Texture, density_input: Texture, settings: Motion3DLayerSettings | None = None) -> None:
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
        self.settings: Motion3DLayerSettings = settings or Motion3DLayerSettings()

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
        return interval

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
            self._flow_interval = self._frame_interval()
            # Stage 1: the flow's 3D velocity, on the whole image; with its use_mask, times the mask
            flow.set_input(self._flow_input)
            if self.depth_enabled and self._view_span <= 0.0 and not self._warned_view_span:
                self._warned_view_span = True
                logger.warning("Motion3DLayer %d: depth is on but no view span is set (set_view_span), so z reads no motion", self._cam_id)
            flow.update(self._depth_input if self.depth_enabled else None, self._flow_interval, self._view_span, self._depth_step)
            self._velocity = flow.velocity

        # Stage 2: The flow bridge
        if kinetic:
            if new_flow:
                # Lucas–Kanade's velocity is a distance, so the trails can travel with it
                self._kinetic_bridge.set_velocity(self._velocity)
                self._kinetic_bridge.update(self._flow_interval)
                if self.settings.draw_mode == Motion3DDrawMode.KINETIC_DEPTH_ACCELERATION:
                    self._kinetic_bridge.update_depth_acceleration()
                if self.settings.flow_bridge.kinetic.onset.spin > 0.0 or self.settings.draw_mode == Motion3DDrawMode.KINETIC_CURL:
                    self._kinetic_bridge.update_curl()
                if self.settings.draw_mode == Motion3DDrawMode.KINETIC_STRIP:
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
            self._visualizer.process_signed(signed[0], self.settings.draw_mode in DEPTH_DRAW_MODES, signed[1])
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
        if self.settings.draw_mode == Motion3DDrawMode.KINETIC_STRIP and self._strip_text.allocated:
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
        textures: dict[Motion3DDrawMode, tuple[Texture | None, int]] = {
            Motion3DDrawMode.INPUT_DEPTH: (self._depth_input, 0),
            Motion3DDrawMode.INPUT_MASK: (self._mask_input, 0),
            Motion3DDrawMode.TEMPERATURE: (self._temperature_bridge.temperature, 0),
            Motion3DDrawMode.KINETIC_ACCELERATION: (self._kinetic_bridge.acceleration, 0),
            Motion3DDrawMode.KINETIC_DEPTH_ACCELERATION: (self._kinetic_bridge.depth_acceleration, 0),
            Motion3DDrawMode.KINETIC_TURN: (self._kinetic_bridge.turn, 0),
            Motion3DDrawMode.KINETIC_HELD_RELEASE: (self._kinetic_bridge.held_release, 0),
            Motion3DDrawMode.KINETIC_RESERVOIR: (self._kinetic_bridge.reservoir, 0),
            Motion3DDrawMode.KINETIC_EMISSION: (self._kinetic_bridge.emission, 0),
            Motion3DDrawMode.KINETIC_CURL: (self._kinetic_bridge.curl, 0),
        }
        texture, channel = textures.get(self.settings.draw_mode, (None, 0))
        return (texture, channel) if texture is not None else None

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        textures = {
            Motion3DDrawMode.INPUT_2D: self._flow_input,
            Motion3DDrawMode.OPTICAL_FLOW_VELOCITY: self._velocity,
            Motion3DDrawMode.VELOCITY: self.smooth_velocity,
            Motion3DDrawMode.DENSITY: self._density_bridge.density,
            Motion3DDrawMode.KINETIC_FAST: self._kinetic_bridge.fast,
            Motion3DDrawMode.KINETIC_SLOW: self._kinetic_bridge.slow,
            Motion3DDrawMode.KINETIC_ONSET_VELOCITY: self._kinetic_bridge.onset_velocity,
            Motion3DDrawMode.KINETIC_STRIP: self._kinetic_strip.texture,
        }
        return textures.get(self.settings.draw_mode, self._flow_input)
