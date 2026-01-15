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

    # Outputs
    OPTICAL_RAW = auto()
    VELOCITY = auto()
    DENSITY = auto()
    DENSITY_VISIBLE = auto()

from modules.utils.HotReloadMethods import HotReloadMethods


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
        self.draw_mode: FlowDrawMode = FlowDrawMode.VELOCITY
        self.velocity_field_mode: bool = False  # False=scalar/direction, True=arrow field

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Configuration Access ==========

    @property
    def optical_config(self) -> OpticalFlowConfig:
        """Optical flow configuration."""
        return self._optical_flow.config  # type: ignore

    @property
    def velocity_config(self) -> VelocityBridgeConfig:
        """Velocity bridge configuration."""
        return self._velocity_bridge.config  # type: ignore

    @property
    def density_config(self) -> DensityBridgeConfig:
        """Density bridge configuration."""
        return self._density_bridge.config  # type: ignore

    @property
    def vis_config(self) -> VisualisationFieldConfig:
        """Visualization configuration."""
        return self._visualizer.config  # type: ignore

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Primary output based on draw_mode."""
        if self.draw_mode == FlowDrawMode.DENSITY:
            return self._density_bridge.density
        elif self.draw_mode == FlowDrawMode.DENSITY_VISIBLE:
            return self._density_bridge.density_visible
        elif self.draw_mode == FlowDrawMode.OPTICAL_RAW:
            return self._optical_flow.output
        else:
            return self._velocity_bridge.velocity

    @property
    def velocity(self) -> Texture:
        """Smoothed velocity output (RG32F)."""
        return self._velocity_bridge.velocity

    @property
    def density(self) -> Texture:
        """Density output with velocity-driven alpha (RGBA32F)."""
        return self._density_bridge.density

    @property
    def density_visible(self) -> Texture:
        """Display-ready density (RGBA8, framerate-scaled)."""
        return self._density_bridge.density_visible

    @property
    def optical_flow_raw(self) -> Texture:
        """Raw optical flow output (RG32F, before bridging)."""
        return self._optical_flow.output

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


        self.vis_config.element_length = 40.0
        self.vis_config.toggle_scalar = False
        self.velocity_config.trail_weight = 0.9
        self.velocity_config.blur_steps = 2
        self.velocity_config.blur_radius = 3.0

        self.draw_mode = FlowDrawMode.DENSITY_VISIBLE


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
                self._optical_flow.set(prev_tex)

            curr_tex: Texture = self._source.texture
            self._optical_flow.set(curr_tex)
            self._optical_flow.update()

        # Stage 2: Smooth velocity through bridge
        self._velocity_bridge.set_velocity(self._optical_flow.output)
        self._velocity_bridge.update(self._delta_time)

        # Stage 3: Apply density bridge (uses smoothed velocity + source RGB)
        self._density_bridge.set_velocity(self._optical_flow.output)
        self._density_bridge.set_density(self._source.texture)
        self._density_bridge.update(self._delta_time)

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self, rect: Rect) -> None:

        """Draw flow visualization based on draw_mode."""
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        if self.draw_mode == FlowDrawMode.OPTICAL_INPUT:
            # Show optical flow input (luminance)
            self._optical_flow.draw_input(rect)

        elif self.draw_mode == FlowDrawMode.OPTICAL_RAW:
            # Raw optical flow (RG32F) - visualize it
            Style.set_blend_mode(Style.BlendMode.ADDITIVE)
            self._visualizer.velocity_field.config.toggle_scalar = self.velocity_field_mode
            self._visualizer.draw(self._optical_flow.output, rect)

        elif self.draw_mode == FlowDrawMode.VELOCITY:
            # Smoothed velocity (RG32F) - visualize it
            Style.set_blend_mode(Style.BlendMode.ADDITIVE)
            self._visualizer.velocity_field.config.toggle_scalar = self.velocity_field_mode
            self._visualizer.draw(self._velocity_bridge.velocity, rect)

        elif self.draw_mode == FlowDrawMode.DENSITY:
            # Density (RGBA32F) - draw directly
            self._density_bridge.density.draw(rect.x, rect.y, rect.width, rect.height)

        elif self.draw_mode == FlowDrawMode.DENSITY_VISIBLE:
            # Display density (RGBA8) - draw directly
            self._density_bridge.density_visible.draw(rect.x, rect.y, rect.width, rect.height)

        Style.pop_style()
