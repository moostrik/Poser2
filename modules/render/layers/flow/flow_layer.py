"""Flow Layer - optical flow computation and velocity/density/temperature bridges."""

# Standard library imports
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
    VelocitySmoothTrail, SmoothTrailSettings,
    DepthVelocity, DepthVelocitySettings,
    KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    Visualizer, VisualizationFieldSettings
)

from modules.utils import HotReloadMethods


class Motion(IntEnum):
    """How the flow measures motion."""
    GRADIENT_FLOW =     0   # the motion along the brightness gradient, as a strength (GradientFlow); z at the pixel
    LUCAS_KANADE_FLOW = 1   # how far each pixel moved (LucasKanadeFlow); z along that motion


class FlowBridge(IntEnum):
    """Which bridge produces the velocity, density and temperature the fluid reads."""
    TRAIL = 0       # a smoothed velocity, ink where there is speed
    KINETIC = auto()  # two trails, reading the change of the motion


class FlowDrawMode(IntEnum):
    """Draw modes for FlowLayer. A 3D velocity (xyz) draws as lines with a dot that grows with z, red
    toward the camera and green away; as scalars, x/y/z as red/green/blue around grey.

    Inputs:
        LUMINANCE - The newest luminance frame of the flow `motion` picks

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


DEPTH_DRAW_MODES: set[FlowDrawMode] = {FlowDrawMode.SMOOTH_VELOCITY_DEPTH, FlowDrawMode.KINETIC_DEPTH_ACCELERATION}     # red and green
VECTOR_DRAW_MODES: set[FlowDrawMode] = {       # 3D velocities, drawn as velocities whatever the texture's format
    FlowDrawMode.VELOCITY, FlowDrawMode.SMOOTH_VELOCITY_INPUT, FlowDrawMode.SMOOTH_VELOCITY_OUTPUT,
    FlowDrawMode.KINETIC_FAST, FlowDrawMode.KINETIC_SLOW, FlowDrawMode.KINETIC_VELOCITY, FlowDrawMode.KINETIC_ONSET,
}


class FlowLayerSettings(BaseSettings):
    """Configuration for FlowLayer (optical flow + bridges)."""
    stage: Field[int] =                     Field(0, description="Pipeline stage for pose data")
    fps: Field[float] =                     Field(60.0, min=1.0, max=240.0)
    draw_mode: Field[FlowDrawMode] =        Field(FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")
    bridge: Field[FlowBridge] =             Field(FlowBridge.TRAIL, description="The bridge the fluid reads")
    motion: Field[Motion] =                 Field(Motion.GRADIENT_FLOW, description="How motion is measured: along the brightness gradient, or how far each pixel moved")

    visualization      = Group(VisualizationFieldSettings)
    gradient_flow      = Group(GradientFlowSettings)
    lucas_kanade_flow  = Group(LucasKanadeFlowSettings)
    depth_velocity     = Group(DepthVelocitySettings)
    velocity_trail     = Group(SmoothTrailSettings)
    kinetic_bridge     = Group(KineticBridgeSettings)
    density_bridge     = Group(DensityBridgeSettings)
    temperature_bridge = Group(TemperatureBridgeSettings)


class FlowLayer(LayerBase):
    """Flow processing layer with optical flow and bridges.

    Measures the motion and prepares velocity/density/temperature for fluid simulation:
    1. GradientFlow or LucasKanadeFlow (settings.motion): the 3D velocity (xyz), z from the depth
    2. VelocitySmoothTrail or KineticBridge (settings.bridge): the smoothed velocity for the fluid
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

        self._depth_velocity: DepthVelocity = DepthVelocity(self.settings.depth_velocity)     # z for both flows
        self._gradient_flow: GradientFlow = GradientFlow(self.settings.gradient_flow, self._depth_velocity)
        self._lucas_kanade_flow: LucasKanadeFlow = LucasKanadeFlow(self.settings.lucas_kanade_flow, self._depth_velocity)
        self._motion: Motion = self.settings.motion     # the flow that measured last; a switch restarts the other
        self._velocity: Texture = self._gradient_flow.velocity     # the velocity of the flow `motion` picks
        self._depth_input: Texture | None = None
        self._mask_input: Texture | None = None
        self._gate_shader: MaskMultiply = MaskMultiply()
        self._gated_color: Fbo = Fbo()      # the density input times the mask
        self._velocity_trail: VelocitySmoothTrail = VelocitySmoothTrail(self.settings.velocity_trail)
        self._kinetic_bridge: KineticBridge = KineticBridge(self.settings.kinetic_bridge)
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._density_bridge: DensityBridge = DensityBridge(self.settings.density_bridge)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(self.settings.temperature_bridge)

        self._visualizer: Visualizer = Visualizer(self.settings.visualization)
        self._kinetic_strip: KineticStrip = KineticStrip()
        self._strip_text: Text = Text()

        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")

        self._time_step: float = 1.0 / 60.0     # dt of one tick
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
        if self.settings.bridge == FlowBridge.KINETIC:
            return self._kinetic_bridge.smooth_velocity
        return self._velocity_trail.smooth_velocity

    @property
    def uses_depth(self) -> bool:
        """Whether the flow `motion` picks measures z: its strength_z is above 0."""
        return self._flow.settings.strength_z > 0.0

    @property
    def depth_enabled(self) -> bool:
        """Whether z reaches the bridge: the flow measures it and a source has depth."""
        return self.uses_depth and self._depth_input is not None and self._depth_input.allocated

    @property
    def onset_velocity(self) -> Texture | None:
        """Second velocity input, xyz (RGBA16F), for fluid simulation; None when the bridge has none."""
        if self.settings.bridge == FlowBridge.KINETIC and self.settings.kinetic_bridge.onset_force > 0.0:
            return self._kinetic_bridge.onset_velocity
        return None

    @property
    def spin(self) -> Texture | None:
        """2D curl (R16F) for the fluid's vorticity force; None when the bridge has none."""
        if self.settings.bridge == FlowBridge.KINETIC and self.settings.kinetic_bridge.spin > 0.0:
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
        """The flow `motion` picks."""
        if self.settings.motion == Motion.LUCAS_KANADE_FLOW:
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
        """Depth for the velocity toward the camera: metres in red, and in alpha where depth counts (the
        player's mask, or everywhere when `set_mask_input` gates the result instead)."""
        self._depth_input = texture

    def set_time_step(self, dt: float) -> None:
        """The dt of one tick, polled by the kinetic bridge; the fluid's step, so a fixed step compares runs."""
        self._time_step = dt

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

        self._depth_velocity.allocate(flow_width, flow_height)
        self._gradient_flow.allocate(flow_width, flow_height)
        self._lucas_kanade_flow.allocate(flow_width, flow_height)
        self._velocity_trail.allocate(flow_width, flow_height, output_width, output_height)
        self._kinetic_bridge.allocate(flow_width, flow_height)
        self._velocity_magnitude.allocate(flow_width, flow_height)
        self._temperature_bridge.allocate(flow_width, flow_height)

    def _deallocate_flow(self) -> None:
        self._depth_velocity.deallocate()
        self._gradient_flow.deallocate()
        self._lucas_kanade_flow.deallocate()
        self._velocity_trail.deallocate()
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
        self._depth_velocity.reset()
        self._velocity_trail.reset()
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
        flow: GradientFlow | LucasKanadeFlow = self._flow
        if self.settings.motion != self._motion:
            # The other flow's frames and the smoothed depth are from before it stopped measuring
            self._motion = self.settings.motion
            flow.reset()
            self._depth_velocity.reset()
        if new_flow:
            # Stage 1: the flow's 3D velocity, on the whole image; with its use_mask, times the mask
            flow.set_input(self._flow_input)
            flow.update(self._depth_input if self.depth_enabled else None, self._flow_interval)
            self._velocity = flow.velocity

        # Stage 2: Bridges
        if self.settings.bridge == FlowBridge.KINETIC:
            if new_flow:
                # Lucas–Kanade's velocity is a distance, so the trails can travel with it
                distance_scale: float | None = self.settings.lucas_kanade_flow.strength_x if self.settings.motion == Motion.LUCAS_KANADE_FLOW else None
                self._kinetic_bridge.set_velocity(self._velocity, distance_scale)
                self._kinetic_bridge.update(self._flow_interval)
                if self.settings.draw_mode == FlowDrawMode.KINETIC_DEPTH_ACCELERATION:
                    self._kinetic_bridge.update_depth_acceleration()
                if self.settings.kinetic_bridge.spin > 0.0 or self.settings.draw_mode == FlowDrawMode.KINETIC_CURL:
                    self._kinetic_bridge.update_curl()
                if self.settings.draw_mode == FlowDrawMode.KINETIC_STRIP:
                    bridge: KineticBridge = self._kinetic_bridge
                    self._kinetic_strip.update(bridge.fast, bridge.acceleration, bridge.emission, self.settings.visualization.strip_scale)
        else:
            self._velocity_trail.set_velocity(self._velocity)
            self._velocity_trail.update()

        if new_flow:
            self._flow_interval = 0.0

        self._velocity_magnitude.set_input(self.smooth_velocity)
        self._velocity_magnitude.update()

        # The amount the density bridge makes its alpha from: the kinetic bridge's emission, or the trail's speed
        alpha: Texture = self._kinetic_bridge.emission if self.settings.bridge == FlowBridge.KINETIC else self._velocity_magnitude.magnitude

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
            FlowDrawMode.KINETIC_DEPTH_ACCELERATION: (self._kinetic_bridge.depth_acceleration, 0),
            FlowDrawMode.DENSITY_BRIDGE_INPUT_ALPHA: (self._density_bridge.alpha_input, 0),
        }
        return textures.get(self.settings.draw_mode)

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        textures = {
            FlowDrawMode.LUMINANCE: self._flow.luminance,
            FlowDrawMode.VELOCITY: self._velocity,
            FlowDrawMode.SMOOTH_VELOCITY_INPUT: self._kinetic_bridge.velocity_input if self.settings.bridge == FlowBridge.KINETIC else self._velocity_trail.velocity_input,
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
