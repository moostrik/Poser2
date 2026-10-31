"""Flow Layer - optical flow computation and velocity/density/temperature bridges."""

# Standard library imports
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import GL_VIEWPORT, glGetIntegerv

# Local application imports
from modules.gl import GpuTimer, Texture, Style, Blit, Text
from ..layer_base import LayerBase
from modules.board import HasFrames
from modules.pose import Frame, AngleMotion

from modules.settings import Field, BaseSettings, Group
from modules.flow import (
    OpticalFlow, OpticalFlowSettings,
    VelocitySmoothTrail, SmoothTrailSettings,
    KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES,
    Magnitude, VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
    Visualizer, VisualizationFieldSettings
)

from modules.utils import HotReloadMethods


class FlowBridge(IntEnum):
    """Which bridge produces the velocity, density and temperature the fluid reads."""
    TRAIL = 0       # a smoothed velocity, ink where there is speed
    KINETIC = auto()  # two trails, reading the change of the motion


class FlowDrawMode(IntEnum):
    """Draw modes for FlowLayer.

    Inputs:
        OPTICAL_INPUT - Luminance frames fed to optical flow

    Outputs:
        OPTICAL_OUTPUT - Raw optical flow velocity (RG32F)
        SMOOTH_VELOCITY_OUTPUT - Smoothed velocity (RG32F)
        DENSITY_BRIDGE_OUTPUT - Colored density (RGBA32F)
        TEMPERATURE_BRIDGE_OUTPUT - Temperature field (R32F)

    Kinetic bridge:
        KINETIC_FAST / KINETIC_SLOW - The two trails (RG16F)
        KINETIC_ACCELERATION - Along the motion in two colors: + onset warm, - release cold
        KINETIC_TURN - Across the motion in two colors
        KINETIC_HELD_RELEASE - The peak-held release
        KINETIC_RESERVOIR - Ink stored per pixel
        KINETIC_VELOCITY - The velocity the fluid gets, after drop_velocity (RG16F)
        KINETIC_EMISSION - The ink the density bridge gets
        KINETIC_ONSET - The second velocity input (RG16F)
        KINETIC_CURL - The fast trail's 2D curl in two colors
        KINETIC_STRIP - Speed, onset, release and emission over the last seconds
    """
    OPTICAL_INPUT = 0
    OPTICAL_OUTPUT = auto()
    SMOOTH_VELOCITY_INPUT = auto()
    SMOOTH_VELOCITY_OUTPUT = auto()
    SMOOTH_VELOCITY_MAGNITUDE = auto()
    DENSITY_BRIDGE_INPUT_COLOR = auto()
    DENSITY_BRIDGE_INPUT_VELOCITY = auto()
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


class FlowLayerSettings(BaseSettings):
    """Configuration for FlowLayer (optical flow + bridges)."""
    stage: Field[int] =                     Field(0, description="Pipeline stage for pose data")
    fps: Field[float] =                     Field(60.0, min=1.0, max=240.0)
    draw_mode: Field[FlowDrawMode] =        Field(FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")
    bridge: Field[FlowBridge] =             Field(FlowBridge.TRAIL, description="The bridge the fluid reads")

    visualization      = Group(VisualizationFieldSettings)
    optical_flow       = Group(OpticalFlowSettings)
    velocity_trail     = Group(SmoothTrailSettings)
    kinetic_bridge     = Group(KineticBridgeSettings)
    density_bridge     = Group(DensityBridgeSettings)
    temperature_bridge = Group(TemperatureBridgeSettings)


class FlowLayer(LayerBase):
    """Flow processing layer with optical flow and bridges.

    Computes optical flow and prepares velocity/density/temperature for fluid simulation:
    1. OpticalFlow: Computes motion from source frames
    2. VelocitySmoothTrail or KineticBridge (settings.bridge): The velocity for the fluid
    3. DensityBridge: Combines RGB density with velocity info
    4. TemperatureBridge: Generates temperature field

    Output textures for FluidLayer:
        - velocity: Smoothed velocity (RG16F)
        - density: Colored density (RGBA16F)
        - temperature: Temperature field (R16F)

    Usage:
        flow = FlowLayer(cam_id, board, mask, motion, colors, settings)
        flow.update()
        # Access outputs for fluid simulation:
        fluid.add_velocity(flow.velocity)
        fluid.add_density(flow.density)
        fluid.add_temperature(flow.temperature)
    """

    def __init__(self, cam_id: int, board: HasFrames, flow_source: LayerBase, flow_input: Texture, density_input: Texture, settings: FlowLayerSettings | None = None) -> None:
        """Initialize flow layer.

        Args:
            cam_id: Camera ID
            board: Board for pose data
            flow_source: The source layer behind the flow input; optical flow runs when it has a new frame
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

        self._optical_flow: OpticalFlow = OpticalFlow(self.settings.optical_flow)
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

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Visualization output texture."""
        return self._visualizer.texture

    @property
    def velocity(self) -> Texture:
        """Velocity output (RG16F) for fluid simulation."""
        if self.settings.bridge == FlowBridge.KINETIC:
            return self._kinetic_bridge.velocity
        return self._velocity_trail.velocity

    @property
    def onset(self) -> Texture | None:
        """Second velocity input (RG16F) for fluid simulation; None when the bridge has none."""
        if self.settings.bridge == FlowBridge.KINETIC and self.settings.kinetic_bridge.onset_force > 0.0:
            return self._kinetic_bridge.onset
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
        """Velocity magnitude output (RR16F) for fluid simulation."""
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

    def set_time_step(self, dt: float) -> None:
        """The dt of one tick, polled by the kinetic bridge; the fluid's step, so a fixed step compares runs."""
        self._time_step = dt

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate all processing stages.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (formats determined by each stage)
        """
        sim_width = self.settings.width
        sim_height = self.settings.height

        self._optical_flow.allocate(sim_width, sim_height)
        self._velocity_trail.allocate(sim_width, sim_height, width, height)
        self._kinetic_bridge.allocate(sim_width, sim_height)
        self._velocity_magnitude.allocate(sim_width, sim_height)
        self._density_bridge.allocate(width, height)
        self._temperature_bridge.allocate(sim_width, sim_height)

        self._visualizer.allocate(width, height)
        self._kinetic_strip.allocate()
        self._strip_text.allocate(font_size=18)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._optical_flow.deallocate()
        self._velocity_trail.deallocate()
        self._kinetic_bridge.deallocate()
        self._velocity_magnitude.deallocate()
        self._density_bridge.deallocate()
        self._temperature_bridge.deallocate()
        self._visualizer.deallocate()
        self._kinetic_strip.deallocate()
        self._strip_text.deallocate()
        self._gpu_timer.deallocate()

    def reset(self) -> None:
        """Reset all processing stages."""
        self._optical_flow.reset()
        self._velocity_trail.reset()
        self._kinetic_bridge.reset()
        self._density_bridge.reset()
        self._kinetic_strip.reset()
        self._flow_interval = 0.0

    # ========== Processing ==========

    def update(self) -> None:
        """Update optical flow and bridges."""
        # Get motion data from pose
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._cam_id)
        motion = pose[AngleMotion].value if pose is not None and AngleMotion in pose else 0.0

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._gpu_timer.begin()

        new_flow: bool = self._flow_source.dirty
        self._flow_interval += self._time_step

        if new_flow:
            # pass
            # Stage 1: Optical flow
            self._optical_flow.set_input(self._flow_input)
            self._optical_flow.update()

        # Stage 2: Bridges
        velocity: Texture
        ink: Texture    # what the density bridge reads as alpha, through its length
        if self.settings.bridge == FlowBridge.KINETIC:
            if new_flow:
                self._kinetic_bridge.set_velocity(self._optical_flow.velocity)
                self._kinetic_bridge.update(self._flow_interval)
                if self.settings.kinetic_bridge.spin > 0.0 or self.settings.draw_mode == FlowDrawMode.KINETIC_CURL:
                    self._kinetic_bridge.update_curl()
                if self.settings.draw_mode == FlowDrawMode.KINETIC_STRIP:
                    bridge: KineticBridge = self._kinetic_bridge
                    self._kinetic_strip.update(bridge.fast, bridge.acceleration, bridge.emission, self.settings.visualization.strip_scale)
            velocity = self._kinetic_bridge.velocity
            ink = self._kinetic_bridge.emission
        else:
            self._velocity_trail.set_velocity(self._optical_flow.velocity)
            self._velocity_trail.update()
            velocity = self._velocity_trail.velocity
            ink = velocity

        if new_flow:
            self._flow_interval = 0.0

        self._velocity_magnitude.set_input(velocity)
        self._velocity_magnitude.update()

        # self._density_bridge.set_color(self._flow_input)
        self._density_bridge.set_color(self._density_input)
        # self._density_bridge.update(motion)
        self._density_bridge.set_velocity(ink)
        self._density_bridge.update()

        self._temperature_bridge.set_color(self._density_input)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()

        self._gpu_timer.end()

        Style.set_blend_mode(Style.BlendMode.ALPHA)

        signed: Texture | None = self._get_signed_texture()
        if signed is not None:
            self._visualizer.process_signed(signed)
        else:
            self._visualizer.process(self._get_draw_texture())
        Style.pop_style()

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

    def _get_signed_texture(self) -> Texture | None:
        """The scalar field to draw in two colors, None when draw_mode is another kind of texture."""
        textures = {
            FlowDrawMode.KINETIC_ACCELERATION: self._kinetic_bridge.acceleration,
            FlowDrawMode.KINETIC_TURN: self._kinetic_bridge.turn,
            FlowDrawMode.KINETIC_HELD_RELEASE: self._kinetic_bridge.held_release,
            FlowDrawMode.KINETIC_RESERVOIR: self._kinetic_bridge.reservoir,
            FlowDrawMode.KINETIC_EMISSION: self._kinetic_bridge.emission,
            FlowDrawMode.KINETIC_CURL: self._kinetic_bridge.curl,
        }
        if self.settings.bridge == FlowBridge.KINETIC:
            # On this bridge the density bridge's second input is the emission, not a velocity
            textures[FlowDrawMode.DENSITY_BRIDGE_INPUT_VELOCITY] = self._density_bridge.velocity_input
        return textures.get(self.settings.draw_mode)

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        textures = {
            FlowDrawMode.OPTICAL_INPUT: self._optical_flow.color_input,
            FlowDrawMode.OPTICAL_OUTPUT: self._optical_flow.velocity,
            FlowDrawMode.SMOOTH_VELOCITY_INPUT: self._kinetic_bridge.velocity_input if self.settings.bridge == FlowBridge.KINETIC else self._velocity_trail.velocity_input,
            FlowDrawMode.SMOOTH_VELOCITY_OUTPUT: self.velocity,
            FlowDrawMode.SMOOTH_VELOCITY_MAGNITUDE: self._velocity_magnitude.magnitude,
            FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR: self._density_bridge.color_input,
            FlowDrawMode.DENSITY_BRIDGE_INPUT_VELOCITY: self._density_bridge.velocity_input,
            FlowDrawMode.DENSITY_BRIDGE_OUTPUT: self._density_bridge.density,
            FlowDrawMode.TEMPERATURE_BRIDGE_INPUT_COLOR: self._temperature_bridge.color_input,
            FlowDrawMode.TEMPERATURE_BRIDGE_INPUT_MASK: self._temperature_bridge.mask_input,
            FlowDrawMode.TEMPERATURE_BRIDGE_OUTPUT: self._temperature_bridge.temperature,
            FlowDrawMode.KINETIC_FAST: self._kinetic_bridge.fast,
            FlowDrawMode.KINETIC_SLOW: self._kinetic_bridge.slow,
            FlowDrawMode.KINETIC_VELOCITY: self._kinetic_bridge.velocity,
            FlowDrawMode.KINETIC_ONSET: self._kinetic_bridge.onset,
            FlowDrawMode.KINETIC_STRIP: self._kinetic_strip.texture,
        }
        return textures.get(self.settings.draw_mode, self._density_input)
