"""Unified Flow Layer - optical flow, velocity bridge, and density bridge."""

# Standard library imports
from enum import IntEnum, auto
from dataclasses import dataclass, field


# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Rect

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
    """Draw modes for unified FlowLayer.

    Inputs:
        OPTICAL_INPUT - Luminance frames fed to optical flow

    Outputs:
        OPTICAL_RAW - Raw optical flow (RG32F, visualized)
        VELOCITY - Smoothed velocity (RG32F, visualized)
        DENSITY - Colored density (RGBA32F, direct)
        DENSITY_VISIBLE - Display density (RGBA8, direct)
    """
    # Inputs
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

DRAW_MODE_BLEND_MODES: dict[FlowDrawMode, Style.BlendMode] = {
    FlowDrawMode.OPTICAL_INPUT:                 Style.BlendMode.DISABLED,
    FlowDrawMode.OPTICAL_OUTPUT:                Style.BlendMode.ADDITIVE,
    FlowDrawMode.SMOOTH_VELOCITY_INPUT:         Style.BlendMode.DISABLED,
    FlowDrawMode.SMOOTH_VELOCITY_OUTPUT:        Style.BlendMode.ADDITIVE,
    FlowDrawMode.SMOOTH_VELOCITY_MAGNITUDE:     Style.BlendMode.ADDITIVE,
    FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR:    Style.BlendMode.DISABLED,
    FlowDrawMode.DENSITY_BRIDGE_INPUT_VELOCITY: Style.BlendMode.ADDITIVE,
    FlowDrawMode.DENSITY_BRIDGE_OUTPUT:         Style.BlendMode.DISABLED,
    FlowDrawMode.TEMP_BRIDGE_INPUT_COLOR:       Style.BlendMode.DISABLED,
    FlowDrawMode.TEMP_BRIDGE_INPUT_MASK:        Style.BlendMode.ADDITIVE,
    FlowDrawMode.TEMP_BRIDGE_OUTPUT:            Style.BlendMode.DISABLED,
}


@dataclass
class FlowConfig:
    """Configuration for unified FlowLayer."""
    fps: float = 60.0
    draw_mode: FlowDrawMode = FlowDrawMode.SMOOTH_VELOCITY_OUTPUT
    field_mode: bool = False  # False=scalar/direction, True=arrow field
    simulation_scale: float = 0.25

    visualisation: VisualisationFieldConfig = field(default_factory=VisualisationFieldConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    velocity_trail: SmoothTrailConfig = field(default_factory=SmoothTrailConfig)
    density_bridge: DensityBridgeConfig = field(default_factory=DensityBridgeConfig)
    TemperatureBridge: TemperatureBridgeConfig = field(default_factory=TemperatureBridgeConfig)
    velocity_bridge_scale: float = 1.0 # second parameter that only applies to velocity input of the fluid simulation (scale of velocity_trail influences all the bridges)

class FlowLayer(LayerBase):
    """Unified flow processing layer.

    Combines optical flow computation with velocity and density bridges:
    1. OpticalFlow: Computes motion from source frames
    2. VelocityBridge: Smooths velocity for fluid simulation
    3. DensityBridge: Combines RGB density with velocity magnitude

    Usage:
        flow = FlowLayer(camera_source, fps=60)
        flow.draw_mode = FlowDrawMode.VELOCITY  # or DENSITY, OPTICAL_RAW, etc.
        flow.velocity_field_mode = True  # Arrow field vs scalar direction map

        # Each frame
        flow.update()
        velocity = flow.velocity  # RG32F for simulation
        density = flow.density    # RGBA32F colored density
    """

    def __init__(self, source: LayerBase, config: FlowConfig | None = None) -> None:
        """Initialize unified flow layer.

        Args:
            source: Source layer providing RGB frames (e.g., FlowSourceLayer)
            fps: Target framerate for timestep calculation
        """
        self.config: FlowConfig = config or FlowConfig()

        self._source: LayerBase = source
        self._delta_time: float = 1 / self.config.fps

        self._flows: list[FlowBase] = []

        self._optical_flow: OpticalFlow = OpticalFlow(self.config.optical_flow)

        self._velocity_trail: VelocitySmoothTrail = VelocitySmoothTrail(self.config.velocity_trail)
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._density_bridge: DensityBridge = DensityBridge(self.config.density_bridge)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(self.config.TemperatureBridge)

        # Visualization
        self._visualizer: Visualizer = Visualizer(self.config.visualisation)

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Smoothed velocity output (RG32F)."""
        return self._velocity_trail.velocity

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
        self._velocity_trail.allocate(sim_width, sim_height)
        self._velocity_magnitude.allocate(sim_width, sim_height)
        self._density_bridge.allocate(width, height)
        self._temperature_bridge.allocate(sim_width, sim_height)
        self._visualizer.allocate(width, height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._optical_flow.deallocate()
        self._velocity_trail.deallocate()
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


        self.config.visualisation.element_width = 1.0 # in pixels
        self.config.visualisation.spacing = 20 # in pixels
        self.config.visualisation.element_length = 40.0 # in pixels
        self.config.visualisation.scale = 1.0
        self.config.visualisation.toggle_scalar = False

        self.config.optical_flow.offset = 8
        self.config.optical_flow.threshold = 0.01
        self.config.optical_flow.strength_x = 3.3
        self.config.optical_flow.strength_y = 3.3
        self.config.optical_flow.boost = 0.0

        self.config.velocity_trail.scale = 1.0
        self.config.velocity_trail.trail_weight = 0.66
        self.config.velocity_trail.blur_steps = 2
        self.config.velocity_trail.blur_radius = 3.0
        self.config.density_bridge.saturation = 1.2
        self.config.density_bridge.brightness = 1.0


        self.config.draw_mode = FlowDrawMode.OPTICAL_INPUT
        # self.config.draw_mode = FlowDrawMode.OPTICAL_OUTPUT
        # self.config.draw_mode = FlowDrawMode.SMOOTH_VELOCITY_INPUT
        self.config.draw_mode = FlowDrawMode.SMOOTH_VELOCITY_OUTPUT
        # self.config.draw_mode = FlowDrawMode.SMOOTH_VELOCITY_MAGNITUDE
        # self.config.draw_mode = FlowDrawMode.DENSITY_BRIDGE_INPUT_COLOR
        # self.config.draw_mode = FlowDrawMode.DENSITY_BRIDGE_INPUT_VELOCITY
        # self.config.draw_mode = FlowDrawMode.DENSITY_BRIDGE_OUTPUT
        # self.config.draw_mode = FlowDrawMode.TEMP_BRIDGE_INPUT_COLOR
        # self.config.draw_mode = FlowDrawMode.TEMP_BRIDGE_INPUT_MASK
        # self.config.draw_mode = FlowDrawMode.TEMP_BRIDGE_OUTPUT


        """Update full processing pipeline."""
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Check if source is active
        active: bool = getattr(self._source, "available", True)
        if not active:
            self.reset()
            Style.pop_style()
            return

        # Stage 1: Compute optical flow
        dirty: bool = getattr(self._source, "dirty", True)
        if dirty:
            prev_tex: Texture | None = getattr(self._source, "prev_texture", None)
            if prev_tex is not None:
                self._optical_flow.set_color(prev_tex)

            curr_tex: Texture = self._source.texture
            self._optical_flow.set_color(curr_tex)
            self._optical_flow.update()

        img = self._optical_flow.velocity.read_to_numpy()
        # if img is not None:
        #     print("Optical Flow Velocity Stats: min =", img.min(), "max =", img.max())


        # Stage 2: Bridge
        self._velocity_trail.set_velocity(self._optical_flow.velocity)
        self._velocity_trail.update()

        self._velocity_magnitude.set_input(self._velocity_trail.velocity)
        self._velocity_magnitude.update()
        self._density_bridge.set_color(self._source.texture)
        self._density_bridge.set_velocity(self._velocity_trail.velocity)
        self._density_bridge.update()
        self._temperature_bridge.set_color(self._source.texture)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self, rect: Rect) -> None:

        """Draw flow visualization based on draw_mode."""
        Style.push_style()
        Style.set_blend_mode(DRAW_MODE_BLEND_MODES.get(self.config.draw_mode, Style.BlendMode.DISABLED))
        self._visualizer.draw(self._get_draw_texture(), rect)

        Style.pop_style()

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
            return self._source.texture