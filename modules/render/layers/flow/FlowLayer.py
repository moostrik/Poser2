"""Flow Layer - optical flow computation and velocity/density/temperature bridges."""

# Standard library imports
from enum import IntEnum, auto
from dataclasses import dataclass, field

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Rect, Blit
from modules.DataHub import DataHub, DataHubType, Stage
from modules.pose.Frame import Frame, MotionGate, Similarity

from modules.flow import (
    FlowBase,
    OpticalFlow, OpticalFlowConfig,
    VelocitySmoothTrail, SmoothTrailConfig,
    Magnitude, VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeConfig,
    DensityBridge, DensityBridgeConfig,
    Visualizer, VisualisationFieldConfig
)

from modules.utils.HotReloadMethods import HotReloadMethods


class FlowDrawMode(IntEnum):
    """Draw modes for FlowLayer.

    Inputs:
        OPTICAL_INPUT - Luminance frames fed to optical flow

    Outputs:
        OPTICAL_OUTPUT - Raw optical flow velocity (RG32F)
        SMOOTH_VELOCITY_OUTPUT - Smoothed velocity (RG32F)
        DENSITY_BRIDGE_OUTPUT - Colored density (RGBA32F)
        TEMP_BRIDGE_OUTPUT - Temperature field (R32F)
    """
    OPTICAL_INPUT = 0
    OPTICAL_OUTPUT = auto()
    SMOOTH_VELOCITY_INPUT = auto()
    SMOOTH_VELOCITY_OUTPUT = auto()
    SMOOTH_VELOCITY_MAGNITUDE = auto()
    DENSITY_BRIDGE_INPUT_COLOR = auto()
    DENSITY_BRIDGE_INPUT_VELOCITY = auto()
    DENSITY_BRIDGE_OUTPUT = auto()
    TEMP_BRIDGE_INPUT_COLOR = auto()
    TEMP_BRIDGE_INPUT_MASK = auto()
    TEMP_BRIDGE_OUTPUT = auto()


@dataclass
class FlowLayerConfig:
    """Configuration for FlowLayer (optical flow + bridges)."""
    fps: float = 60.0
    draw_mode: FlowDrawMode = FlowDrawMode.SMOOTH_VELOCITY_OUTPUT
    blend_mode: Style.BlendMode = Style.BlendMode.ADDITIVE
    simulation_scale: float = 0.125

    visualisation: VisualisationFieldConfig = field(default_factory=VisualisationFieldConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    velocity_trail: SmoothTrailConfig = field(default_factory=SmoothTrailConfig)
    density_bridge: DensityBridgeConfig = field(default_factory=DensityBridgeConfig)
    temperature_bridge: TemperatureBridgeConfig = field(default_factory=TemperatureBridgeConfig)


# Keep FlowConfig as alias for backward compatibility
FlowConfig = FlowLayerConfig

class FlowLayer(LayerBase):
    """Flow processing layer with optical flow and bridges.

    Computes optical flow and prepares velocity/density/temperature for fluid simulation:
    1. OpticalFlow: Computes motion from source frames
    2. VelocitySmoothTrail: Smooths velocity temporally
    3. DensityBridge: Combines RGB density with velocity info
    4. TemperatureBridge: Generates temperature field

    Output textures for FluidLayer:
        - velocity: Smoothed velocity (RG16F)
        - density: Colored density (RGBA16F)
        - temperature: Temperature field (R16F)

    Usage:
        flow = FlowLayer(cam_id, data_hub, mask, motion, colors, config)
        flow.update()
        # Access outputs for fluid simulation:
        fluid.add_velocity(flow.velocity)
        fluid.add_density(flow.density)
        fluid.add_temperature(flow.temperature)
    """

    def __init__(self, cam_id: int, data_hub: DataHub, mask: Texture, motion: Texture, colors: list[tuple[float, float, float, float]], config: FlowLayerConfig | None = None) -> None:
        """Initialize flow layer.

        Args:
            cam_id: Camera ID
            data_hub: Data hub for pose data
            mask: Mask texture for optical flow input
            motion: Motion texture for density bridge color input
            colors: Track colors (unused, kept for compatibility)
            config: Layer configuration
        """
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._mask: Texture = mask
        self._motion: Texture = motion
        self._colors: list[tuple[float, float, float, float]] = colors
        self.config: FlowLayerConfig = config or FlowLayerConfig()

        self._delta_time: float = 1 / self.config.fps

        self._optical_flow: OpticalFlow = OpticalFlow(self.config.optical_flow)
        self._velocity_trail: VelocitySmoothTrail = VelocitySmoothTrail(self.config.velocity_trail)
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._density_bridge: DensityBridge = DensityBridge(self.config.density_bridge)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(self.config.temperature_bridge)

        self._visualizer: Visualizer = Visualizer(self.config.visualisation)

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Visualization output texture."""
        return self._visualizer.texture

    @property
    def velocity(self) -> Texture:
        """Smoothed velocity output (RG16F) for fluid simulation."""
        return self._velocity_trail.velocity

    @property
    def density(self) -> Texture:
        """Colored density output (RGBA16F) for fluid simulation."""
        return self._density_bridge.density

    @property
    def temperature(self) -> Texture:
        """Temperature output (R16F) for fluid simulation."""
        return self._temperature_bridge.temperature

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate all processing stages.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (formats determined by each stage)
        """
        sim_width = int(width * self.config.simulation_scale)
        sim_height = int(height * self.config.simulation_scale)

        self._optical_flow.allocate(sim_width, sim_height)
        self._velocity_trail.allocate(sim_width, sim_height, width, height)
        self._velocity_magnitude.allocate(sim_width, sim_height)
        self._density_bridge.allocate(width, height)
        self._temperature_bridge.allocate(sim_width, sim_height)

        self._visualizer.allocate(width, height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._optical_flow.deallocate()
        self._velocity_trail.deallocate()
        self._velocity_magnitude.deallocate()
        self._density_bridge.deallocate()
        self._temperature_bridge.deallocate()
        self._visualizer.deallocate()

    def reset(self) -> None:
        """Reset all processing stages."""
        self._optical_flow.reset()
        self._velocity_trail.reset()
        self._density_bridge.reset()

    # ========== Processing ==========

    def update(self) -> None:
        """Update optical flow and bridges."""
        # Configuration overrides (for hot-reload testing)
        self.config.visualisation.element_width = 1.0
        self.config.visualisation.spacing = 20
        self.config.visualisation.element_length = 40.0
        self.config.visualisation.scale = 1.0
        self.config.visualisation.toggle_scalar = False

        self.config.optical_flow.offset = 3
        self.config.optical_flow.threshold = 0.0
        self.config.optical_flow.strength_x = 3.3
        self.config.optical_flow.strength_y = 3.3
        self.config.optical_flow.boost = 0.0

        self.config.velocity_trail.scale = 1.0
        self.config.velocity_trail.trail_weight = 0.1
        self.config.velocity_trail.blur_steps = 2
        self.config.velocity_trail.blur_radius = 3.0

        self.config.density_bridge.saturation = 1.2
        self.config.density_bridge.brightness = 0.5

        # Get motion data from pose
        pose: Frame | None = self._data_hub.get_pose(Stage.LERP, self._cam_id)
        motion = pose.angle_motion.value if pose is not None else 0.0

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Stage 1: Optical flow
        self._optical_flow.set_input(self._mask)
        self._optical_flow.update()

        # Stage 2: Bridges
        self._velocity_trail.set_velocity(self._optical_flow.velocity)
        self._velocity_trail.update()

        self._velocity_magnitude.set_input(self._velocity_trail.velocity)
        self._velocity_magnitude.update()

        self._density_bridge.set_color(self._motion)
        self._density_bridge.set_velocity(self._mask)
        self._density_bridge.update(motion / 60.0)

        self._temperature_bridge.set_color(self._mask)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()

        # Update visualization
        self._visualizer.update(self._get_draw_texture())

        self.config.blend_mode = Style.BlendMode.DISABLED

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        Style.push_style()
        Style.set_blend_mode(self.config.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        Style.pop_style()

    # ========== Texture Selection ==========

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        if self.config.draw_mode == FlowDrawMode.OPTICAL_INPUT:
            return self._optical_flow.color_input
        elif self.config.draw_mode == FlowDrawMode.OPTICAL_OUTPUT:
            return self._optical_flow.velocity
        elif self.config.draw_mode == FlowDrawMode.SMOOTH_VELOCITY_INPUT:
            return self._velocity_trail.velocity_input
        elif self.config.draw_mode == FlowDrawMode.SMOOTH_VELOCITY_OUTPUT:
            return self._velocity_trail.velocity
        elif self.config.draw_mode == FlowDrawMode.SMOOTH_VELOCITY_MAGNITUDE:
            return self._velocity_magnitude.magnitude
        elif self.config.draw_mode == FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR:
            return self._density_bridge.color_input
        elif self.config.draw_mode == FlowDrawMode.DENSITY_BRIDGE_INPUT_VELOCITY:
            return self._density_bridge.velocity_input
        elif self.config.draw_mode == FlowDrawMode.DENSITY_BRIDGE_OUTPUT:
            return self._density_bridge.density
        elif self.config.draw_mode == FlowDrawMode.TEMP_BRIDGE_INPUT_COLOR:
            return self._temperature_bridge.color_input
        elif self.config.draw_mode == FlowDrawMode.TEMP_BRIDGE_INPUT_MASK:
            return self._temperature_bridge.mask_input
        elif self.config.draw_mode == FlowDrawMode.TEMP_BRIDGE_OUTPUT:
            return self._temperature_bridge.temperature
        else:
            return self._mask