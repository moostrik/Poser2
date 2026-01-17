"""Unified Flow Layer - optical flow, velocity bridge, and density bridge."""

# Standard library imports
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.flow import (
    OpticalFlow, OpticalFlowConfig,
    VelocityBridge, VelocityBridgeConfig,
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
    VELOCITY_BRIDGE_INPUT = auto()
    VELOCITY_BRIDGE_OUTPUT = auto()
    DENSITY_BRIDGE_INPUT_DENSITY = auto()
    DENSITY_BRIDGE_INPUT_VELOCITY = auto()
    DENSITY_BRIDGE_OUTPUT = auto()
    DENSITY_BRIDGE_OUTPUT_VELOCITY = auto()

DRAW_MODE_BLEND_MODES: dict[FlowDrawMode, Style.BlendMode] = {
    FlowDrawMode.OPTICAL_INPUT:                 Style.BlendMode.DISABLED,
    FlowDrawMode.OPTICAL_OUTPUT:                Style.BlendMode.ADDITIVE,
    FlowDrawMode.VELOCITY_BRIDGE_INPUT:         Style.BlendMode.DISABLED,
    FlowDrawMode.VELOCITY_BRIDGE_OUTPUT:        Style.BlendMode.ADDITIVE,
    FlowDrawMode.DENSITY_BRIDGE_INPUT_DENSITY:  Style.BlendMode.DISABLED,
    FlowDrawMode.DENSITY_BRIDGE_INPUT_VELOCITY: Style.BlendMode.ADDITIVE,
    FlowDrawMode.DENSITY_BRIDGE_OUTPUT:         Style.BlendMode.DISABLED,
    FlowDrawMode.DENSITY_BRIDGE_OUTPUT_VELOCITY:Style.BlendMode.ADDITIVE,
}


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

    def __init__(self, source: LayerBase, fps: float = 60) -> None:
        """Initialize unified flow layer.

        Args:
            source: Source layer providing RGB frames (e.g., FlowSourceLayer)
            fps: Target framerate for timestep calculation
        """
        self._source: LayerBase = source
        self._delta_time: float = 1 / fps

        # Stage 1: Optical Flow (always computed)
        self._optical_flow: OpticalFlow = OpticalFlow()

        # Stage 2: Velocity Bridge (smooths optical flow)
        self._velocity_bridge: VelocityBridge = VelocityBridge()

        # Stage 3: Density Bridge (adds color)
        self._density_bridge: DensityBridge = DensityBridge()

        # Visualization
        self._visualizer: Visualizer = Visualizer()

        # Draw settings
        self.draw_mode: FlowDrawMode = FlowDrawMode.VELOCITY_BRIDGE_INPUT
        self.velocity_field_mode: bool = False  # False=scalar/direction, True=arrow field

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Configuration Access ==========

    @property
    def optical_flow_config(self) -> OpticalFlowConfig:
        """Optical flow configuration."""
        return self._optical_flow.config  # type: ignore

    @property
    def velocity_bridge_config(self) -> VelocityBridgeConfig:
        """Velocity bridge configuration."""
        return self._velocity_bridge.config  # type: ignore

    @property
    def density_bridge_config(self) -> DensityBridgeConfig:
        """Density bridge configuration."""
        return self._density_bridge.config  # type: ignore

    @property
    def visualisation_config(self) -> VisualisationFieldConfig:
        """Visualization configuration."""
        return self._visualizer.config  # type: ignore

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Smoothed velocity output (RG32F)."""
        return self._velocity_bridge.velocity

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate all processing stages.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (formats determined by each stage)
        """
        self._optical_flow.allocate(width, height)
        self._velocity_bridge.allocate(width, height)
        self._density_bridge.allocate(width, height)
        self._visualizer.allocate(width, height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._optical_flow.deallocate()
        self._velocity_bridge.deallocate()
        self._density_bridge.deallocate()
        self._visualizer.deallocate()

    def reset(self) -> None:
        """Reset all processing stages."""
        self._optical_flow.reset()
        self._velocity_bridge.reset()
        self._density_bridge.reset()

    # ========== Processing ==========

    def update(self) -> None:


        self.visualisation_config.element_length = 40.0
        self.visualisation_config.toggle_scalar = True
        self.velocity_bridge_config.trail_weight = 0.6
        self.velocity_bridge_config.blur_steps = 2
        self.velocity_bridge_config.blur_radius = 3.0
        self.velocity_bridge_config.time_scale = 160.0
        self.density_bridge_config.velocity_trail_weight = self.velocity_bridge_config.trail_weight
        self.density_bridge_config.velocity_blur_steps = self.velocity_bridge_config.blur_steps
        self.density_bridge_config.velocity_blur_radius = self.velocity_bridge_config.blur_radius
        self.density_bridge_config.time_scale = 1.0


        # self.draw_mode = FlowDrawMode.VELOCITY_BRIDGE_INPUT
        # self.draw_mode = FlowDrawMode.VELOCITY_BRIDGE_OUTPUT
        # self.draw_mode = FlowDrawMode.DENSITY_BRIDGE_OUTPUT_VELOCITY
        self.draw_mode = FlowDrawMode.OPTICAL_OUTPUT


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
                self._optical_flow._set(prev_tex)

            curr_tex: Texture = self._source.texture
            self._optical_flow.set_density(curr_tex)
            self._optical_flow.update()

        # Stage 2: Smooth velocity through bridge
        self._velocity_bridge.set_velocity(self._optical_flow._output)
        self._velocity_bridge.update(self._delta_time)

        # Stage 3: Apply density bridge (uses smoothed velocity + source RGB)
        self._density_bridge.set_velocity(self._optical_flow._output)
        self._density_bridge._set(self._source.texture)
        self._density_bridge.update(self._delta_time)

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self, rect: Rect) -> None:

        """Draw flow visualization based on draw_mode."""
        Style.push_style()
        Style.set_blend_mode(DRAW_MODE_BLEND_MODES.get(self.draw_mode, Style.BlendMode.DISABLED))
        self._visualizer.draw(self._get_draw_texture(), rect)

        Style.pop_style()

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        if self.draw_mode == FlowDrawMode.OPTICAL_INPUT:
            return self._optical_flow._input
        elif self.draw_mode == FlowDrawMode.OPTICAL_OUTPUT:
            return self._optical_flow._output
        elif self.draw_mode == FlowDrawMode.VELOCITY_BRIDGE_INPUT:
            return self._velocity_bridge._input
        elif self.draw_mode == FlowDrawMode.VELOCITY_BRIDGE_OUTPUT:
            return self._velocity_bridge._output
        elif self.draw_mode == FlowDrawMode.DENSITY_BRIDGE_INPUT_DENSITY:
            return self._density_bridge._input
        elif self.draw_mode == FlowDrawMode.DENSITY_BRIDGE_INPUT_VELOCITY:
            return self._density_bridge.velocity_input
        elif self.draw_mode == FlowDrawMode.DENSITY_BRIDGE_OUTPUT:
            return self._density_bridge.density
        elif self.draw_mode == FlowDrawMode.DENSITY_BRIDGE_OUTPUT_VELOCITY:
            return self._density_bridge.velocity
        else:
            return self._optical_flow._output