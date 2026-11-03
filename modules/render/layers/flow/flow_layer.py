"""Flow Layer - optical flow computation and velocity/density/temperature bridges."""

# Standard library imports
import logging
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import GL_RGBA16F, GL_VIEWPORT, glGetIntegerv

# Local application imports
from modules.gl import Fbo, GpuTimer, Texture, Style, Blit, Text
from ..layer_base import LayerBase
from ...shaders import MaskMultiply
from modules.board import HasFrames

from modules.settings import Field, BaseSettings, Group
from modules.flow import (
    GradientFlow, GradientFlowSettings, LucasKanadeFlow, LucasKanadeFlowSettings,
    TrailBridge, TrailBridgeSettings,
    KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    Visualizer, VisualizationFieldSettings
)

from modules.utils import HotReloadMethods

logger = logging.getLogger(__name__)


class OpticalFlowMethod(IntEnum):
    """How the optical flow measures motion."""
    GRADIENT =     0    # the motion along the brightness gradient, as a strength (GradientFlow); z at the pixel
    LUCAS_KANADE = 1    # how far each pixel moved (LucasKanadeFlow); z along that motion


class FlowBridgeMethod(IntEnum):
    """How the flow bridge makes the velocity the fluid reads and the amount the density reads."""
    TRAIL = 0       # a smoothed velocity, ink where there is speed
    KINETIC = auto()  # two trails, reading the change of the motion


class FlowDrawMode(IntEnum):
    """Draw modes for FlowLayer. A 3D velocity (xyz) draws as white x/y arrows with a ring of radius |z|
    on the arrow's scale, red toward the camera and green away; as scalars, x/y/z as red/green/blue around grey.

    Inputs:
        LUMINANCE - The newest luminance frame of the flow the optical flow's method picks

    Depth (red toward the camera, green away, in px per frame: 1 is full color at scalar_scale 1):
        DEPTH_DIFFERENCE - The flow's tracked z, before the threshold
        DEPTH_VELOCITY - The flow's z after the threshold, before strength_z and the mask

    Outputs:
        VELOCITY - The flow's 3D velocity, times the mask with its use_mask
        SMOOTH_VELOCITY_INPUT - What the bridge reads: the velocity (shaped on the trail bridge)
        SMOOTH_VELOCITY_OUTPUT - The bridge's smoothed 3D velocity, what the fluid gets
        SMOOTH_VELOCITY_DEPTH - Its z: red toward the camera, green away
        DENSITY_BRIDGE_INPUT_ALPHA - The amount the density's alpha is made from: speed (trail) or emission (kinetic)
        DENSITY_BRIDGE_OUTPUT - Colored density (RGBA16F)
        TEMPERATURE_BRIDGE_OUTPUT - Temperature field (R16F)

    Kinetic bridge:
        KINETIC_FAST / KINETIC_SLOW - The two trails (RGBA16F, xyz)
        KINETIC_ACCELERATION - Along the motion in two colors: + onset warm, - release cold
        KINETIC_TURN - Across the motion in two colors
        KINETIC_HELD_RELEASE - The peak-held release
        KINETIC_RESERVOIR - Ink stored per pixel
        KINETIC_VELOCITY - The velocity the fluid gets, after drop_velocity (RGBA16F, xyz)
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


DEPTH_DRAW_MODES: set[FlowDrawMode] = {     # red and green
    FlowDrawMode.SMOOTH_VELOCITY_DEPTH, FlowDrawMode.KINETIC_DEPTH_ACCELERATION, FlowDrawMode.DEPTH_DIFFERENCE, FlowDrawMode.DEPTH_VELOCITY,
}
PIXEL_DRAW_MODES: set[FlowDrawMode] = {FlowDrawMode.DEPTH_DIFFERENCE, FlowDrawMode.DEPTH_VELOCITY}     # view widths per frame, shown in px per frame
VECTOR_DRAW_MODES: set[FlowDrawMode] = {       # 3D velocities, drawn as velocities whatever the texture's format
    FlowDrawMode.VELOCITY, FlowDrawMode.SMOOTH_VELOCITY_INPUT, FlowDrawMode.SMOOTH_VELOCITY_OUTPUT,
    FlowDrawMode.KINETIC_FAST, FlowDrawMode.KINETIC_SLOW, FlowDrawMode.KINETIC_VELOCITY, FlowDrawMode.KINETIC_ONSET,
}


class OpticalFlowSettings(BaseSettings):
    """How the optical flow measures the 3D velocity: its method, and each method's settings, z from the depth included."""
    method: Field[OpticalFlowMethod] =  Field(OpticalFlowMethod.GRADIENT, description="How motion is measured: along the brightness gradient, or how far each pixel moved")

    lucas_kanade   = Group(LucasKanadeFlowSettings)
    gradient       = Group(GradientFlowSettings)


class FlowBridgeSettings(BaseSettings):
    """How the flow bridge turns the optical flow into the fluid's velocity, density and temperature."""
    method: Field[FlowBridgeMethod] =   Field(FlowBridgeMethod.TRAIL, description="How the bridge makes the fluid's velocity and the density's amount: one trail, or two trails reading the change of the motion")

    trail       = Group(TrailBridgeSettings)
    kinetic     = Group(KineticBridgeSettings)
    density     = Group(DensityBridgeSettings)
    temperature = Group(TemperatureBridgeSettings)


class FlowLayerSettings(BaseSettings):
    """Configuration for FlowLayer (optical flow + flow bridge)."""
    draw_mode: Field[FlowDrawMode] =        Field(FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")

    optical_flow  = Group(OpticalFlowSettings)
    flow_bridge   = Group(FlowBridgeSettings)
    visualization = Group(VisualizationFieldSettings)


class FlowLayer(LayerBase):
    """Flow processing layer with the optical flow and the flow bridge.

    Measures the motion and prepares velocity/density/temperature for fluid simulation:
    1. GradientFlow or LucasKanadeFlow (settings.optical_flow.method): the 3D velocity (xyz), z from the depth
    2. TrailBridge or KineticBridge (settings.flow_bridge.method): the smoothed velocity for the fluid
    3. DensityBridge: Combines RGB density with an amount as its alpha
    4. TemperatureBridge: Generates temperature field

    Output textures for FluidLayer:
        - smooth_velocity: Smoothed velocity, xyz (RGBA16F)
        - density: Colored density (RGBA16F)
        - temperature: Temperature field (R16F)

    Usage:
        flow = FlowLayer(cam_id, board, flow_source, flow_input, density_input, settings)
        flow.update()
        # Access outputs for fluid simulation:
        fluid.add_velocity(flow.smooth_velocity)
        fluid.add_density(flow.density)
        fluid.add_temperature(flow.temperature)
    """

    def __init__(self, cam_id: int, board: HasFrames, flow_source: LayerBase, flow_input: Texture, density_input: Texture, settings: FlowLayerSettings | None = None) -> None:
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
        self.settings: FlowLayerSettings = settings or FlowLayerSettings()

        optical_flow: OpticalFlowSettings = self.settings.optical_flow
        bridge: FlowBridgeSettings = self.settings.flow_bridge
        self._gradient_flow: GradientFlow = GradientFlow(optical_flow.gradient)
        self._lucas_kanade_flow: LucasKanadeFlow = LucasKanadeFlow(optical_flow.lucas_kanade)
        self._optical_flow_method: OpticalFlowMethod = optical_flow.method     # the flow that measured last; a switch restarts the other
        self._velocity: Texture = self._gradient_flow.velocity     # the velocity of the flow the method picks
        self._depth_input: Texture | None = None
        self._mask_input: Texture | None = None
        self._gate_shader: MaskMultiply = MaskMultiply()
        self._gated_color: Fbo = Fbo()      # the density input times the mask
        self._trail_bridge: TrailBridge = TrailBridge(bridge.trail)
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
        self._flow_interval: float = 0.0        # time since the last optical flow frame (s), counted in ticks

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
        return self._trail_bridge.smooth_velocity

    @property
    def uses_depth(self) -> bool:
        """Whether the flow the optical flow's method picks measures z: its strength_z is above 0."""
        return self._flow.settings.strength_z > 0.0

    @property
    def depth_enabled(self) -> bool:
        """Whether z reaches the bridge: the flow measures it and a source has depth."""
        return self.uses_depth and self._depth_input is not None and self._depth_input.allocated

    @property
    def onset_velocity(self) -> Texture | None:
        """Second velocity input, xyz (RGBA16F), for fluid simulation; None when the bridge has none."""
        bridge: FlowBridgeSettings = self.settings.flow_bridge
        if bridge.method == FlowBridgeMethod.KINETIC and bridge.kinetic.onset_force > 0.0:
            return self._kinetic_bridge.onset_velocity
        return None

    @property
    def spin(self) -> Texture | None:
        """2D curl (R16F) for the fluid's vorticity force; None when the bridge has none."""
        bridge: FlowBridgeSettings = self.settings.flow_bridge
        if bridge.method == FlowBridgeMethod.KINETIC and bridge.kinetic.spin > 0.0:
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

    @property
    def _flow(self) -> GradientFlow | LucasKanadeFlow:
        """The flow the optical flow's method picks."""
        if self.settings.optical_flow.method == OpticalFlowMethod.LUCAS_KANADE:
            return self._lucas_kanade_flow
        return self._gradient_flow

    # ========== Input Setters ==========

    def set_flow_input(self, texture: Texture) -> None:
        self._flow_input = texture

    def set_flow_source(self, layer: LayerBase) -> None:
        self._flow_source = layer

    def set_density_input(self, texture: Texture) -> None:
        self._density_input = texture

    def set_mask_input(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) on the flow's view, or None for no gate. Each flow multiplies
        its 3D velocity by it with its use_mask, and the density color is gated by it, after they are
        measured, so they run on the whole image: a jittery mask edge then only scales what passes, it
        never reads as motion, and needs no blending."""
        self._mask_input = texture
        self._gradient_flow.set_mask(texture)
        self._lucas_kanade_flow.set_mask(texture)

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

        self._gradient_flow.allocate(flow_width, flow_height)
        self._lucas_kanade_flow.allocate(flow_width, flow_height)
        self._trail_bridge.allocate(flow_width, flow_height, output_width, output_height)
        self._kinetic_bridge.allocate(flow_width, flow_height)
        self._velocity_magnitude.allocate(flow_width, flow_height)
        self._temperature_bridge.allocate(flow_width, flow_height)

    def _deallocate_flow(self) -> None:
        self._gradient_flow.deallocate()
        self._lucas_kanade_flow.deallocate()
        self._trail_bridge.deallocate()
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
        self._gradient_flow.reset()
        self._lucas_kanade_flow.reset()
        self._trail_bridge.reset()
        self._kinetic_bridge.reset()
        self._density_bridge.reset()
        self._kinetic_strip.reset()
        self._flow_interval = 0.0

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
        optical_flow: OpticalFlowSettings = self.settings.optical_flow
        kinetic: bool = self.settings.flow_bridge.method == FlowBridgeMethod.KINETIC
        flow: GradientFlow | LucasKanadeFlow = self._flow
        if optical_flow.method != self._optical_flow_method:
            # The other flow's frames and depth are from before it stopped measuring
            self._optical_flow_method = optical_flow.method
            flow.reset()
        if new_flow:
            # Stage 1: the flow's 3D velocity, on the whole image; with its use_mask, times the mask
            flow.set_input(self._flow_input)
            if self.depth_enabled and self._view_span <= 0.0 and not self._warned_view_span:
                self._warned_view_span = True
                logger.warning("FlowLayer %d: depth is on but no view span is set (set_view_span), so z reads no motion", self._cam_id)
            flow.update(self._depth_input if self.depth_enabled else None, self._flow_interval, self._view_span, self._depth_step)
            self._velocity = flow.velocity

        # Stage 2: The flow bridge
        if kinetic:
            if new_flow:
                # Lucas–Kanade's velocity is a distance, so the trails can travel with it
                distance_scale: float | None = optical_flow.lucas_kanade.strength_x if optical_flow.method == OpticalFlowMethod.LUCAS_KANADE else None
                self._kinetic_bridge.set_velocity(self._velocity, distance_scale)
                self._kinetic_bridge.update(self._flow_interval)
                if self.settings.draw_mode == FlowDrawMode.KINETIC_DEPTH_ACCELERATION:
                    self._kinetic_bridge.update_depth_acceleration()
                if self.settings.flow_bridge.kinetic.spin > 0.0 or self.settings.draw_mode == FlowDrawMode.KINETIC_CURL:
                    self._kinetic_bridge.update_curl()
                if self.settings.draw_mode == FlowDrawMode.KINETIC_STRIP:
                    bridge: KineticBridge = self._kinetic_bridge
                    self._kinetic_strip.update(bridge.fast, bridge.acceleration, bridge.emission, self.settings.visualization.strip_scale)
        else:
            self._trail_bridge.set_velocity(self._velocity)
            self._trail_bridge.update()

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
        if self.settings.draw_mode == FlowDrawMode.KINETIC_STRIP and self._strip_text.allocated:
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
        textures: dict[FlowDrawMode, tuple[Texture, int]] = {
            FlowDrawMode.KINETIC_ACCELERATION: (self._kinetic_bridge.acceleration, 0),
            FlowDrawMode.KINETIC_TURN: (self._kinetic_bridge.turn, 0),
            FlowDrawMode.KINETIC_HELD_RELEASE: (self._kinetic_bridge.held_release, 0),
            FlowDrawMode.KINETIC_RESERVOIR: (self._kinetic_bridge.reservoir, 0),
            FlowDrawMode.KINETIC_EMISSION: (self._kinetic_bridge.emission, 0),
            FlowDrawMode.KINETIC_CURL: (self._kinetic_bridge.curl, 0),
            FlowDrawMode.SMOOTH_VELOCITY_DEPTH: (self.smooth_velocity, 2),       # its z
            FlowDrawMode.DEPTH_DIFFERENCE: (self._flow.depth_velocity.difference, 0),
            FlowDrawMode.DEPTH_VELOCITY: (self._flow.depth_velocity.depth_velocity, 0),
            FlowDrawMode.KINETIC_DEPTH_ACCELERATION: (self._kinetic_bridge.depth_acceleration, 0),
            FlowDrawMode.DENSITY_BRIDGE_INPUT_ALPHA: (self._density_bridge.alpha_input, 0),
        }
        return textures.get(self.settings.draw_mode)

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        textures = {
            FlowDrawMode.LUMINANCE: self._flow.luminance,
            FlowDrawMode.VELOCITY: self._velocity,
            FlowDrawMode.SMOOTH_VELOCITY_INPUT: self._kinetic_bridge.velocity_input if self.settings.flow_bridge.method == FlowBridgeMethod.KINETIC else self._trail_bridge.velocity_input,
            FlowDrawMode.SMOOTH_VELOCITY_OUTPUT: self.smooth_velocity,
            FlowDrawMode.SMOOTH_VELOCITY_MAGNITUDE: self._velocity_magnitude.magnitude,
            FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR: self._density_bridge.color_input,
            FlowDrawMode.DENSITY_BRIDGE_OUTPUT: self._density_bridge.density,
            FlowDrawMode.TEMPERATURE_BRIDGE_INPUT_COLOR: self._temperature_bridge.color_input,
            FlowDrawMode.TEMPERATURE_BRIDGE_INPUT_MASK: self._temperature_bridge.mask_input,
            FlowDrawMode.TEMPERATURE_BRIDGE_OUTPUT: self._temperature_bridge.temperature,
            FlowDrawMode.KINETIC_FAST: self._kinetic_bridge.fast,
            FlowDrawMode.KINETIC_SLOW: self._kinetic_bridge.slow,
            FlowDrawMode.KINETIC_VELOCITY: self._kinetic_bridge.smooth_velocity,
            FlowDrawMode.KINETIC_ONSET: self._kinetic_bridge.onset_velocity,
            FlowDrawMode.KINETIC_STRIP: self._kinetic_strip.texture,
        }
        return textures.get(self.settings.draw_mode, self._density_input)
