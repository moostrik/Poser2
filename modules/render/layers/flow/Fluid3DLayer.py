"""Fluid3D Layer - 3D Navier-Stokes fluid simulation with cross-camera flow inputs.

Volumetric extension of FluidLayer. Receives 2D velocity, density, and
temperature from all FlowLayers and runs a full 3D Navier-Stokes simulation
with gaussian depth injection.  The composited 2D density output is already
RGB-coloured — no post-composite colorisation step is needed.
"""

# Standard library imports
from __future__ import annotations
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import *  # type: ignore
import numpy as np

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Blit
from modules.data_hub import DataHub, Stage
from modules.pose.frame import Frame
from modules.pose.features import Similarity, MotionGate, AngleMotion

from modules.settings import Field, Settings, Group
from modules.flow import Visualizer, VisualisationFieldSettings, FluidFlow, FluidFlow3D, FluidFlow3DArray, FluidFlowSettings

from modules.utils.HotReloadMethods import HotReloadMethods

from .FlowLayer import FlowLayer
from modules.render.color_settings import ColorSettings

class FluidSimEngine(IntEnum):
    """Simulation engine selection.

    FLUID_2D:       2D Navier-Stokes (FluidFlow) — fastest, no depth
    FLUID_3D:       3D volumetric with GL_TEXTURE_3D density (FluidFlow3D)
    FLUID_3D_ARRAY: 3D volumetric with GL_TEXTURE_2D_ARRAY density (FluidFlow3DArray) — fastest 3D
    """
    FLUID_2D = 0
    FLUID_3D = auto()
    FLUID_3D_ARRAY = auto()


# Map enum -> class for engine creation
_ENGINE_CLASSES: dict[FluidSimEngine, type] = {
    FluidSimEngine.FLUID_2D: FluidFlow,
    FluidSimEngine.FLUID_3D: FluidFlow3D,
    FluidSimEngine.FLUID_3D_ARRAY: FluidFlow3DArray,
}


class Fluid3DDrawMode(IntEnum):
    """Draw modes for Fluid3DLayer.

    Only fields with a 2D composited output are exposed:
        DENSITY      - RGB-coloured composited density (colour baked in during injection)
        DENSITY_RAW  - Alias for DENSITY (kept for API compatibility)
    """
    DENSITY = 0
    DENSITY_RAW = auto()


class Fluid3DLayerSettings(Settings):
    """Configuration for Fluid3DLayer (3D fluid simulation)."""
    num_players: Field[int] =               Field(3, min=1, max=8, access=Field.INIT)
    sim_engine: Field[FluidSimEngine] =     Field(FluidSimEngine.FLUID_3D_ARRAY)
    draw_mode: Field[Fluid3DDrawMode] =     Field(Fluid3DDrawMode.DENSITY)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)

    fluid_flow    = Group(FluidFlowSettings)
    visualisation = Group(VisualisationFieldSettings)


class Fluid3DLayer(LayerBase):
    """3D fluid simulation layer with cross-camera flow inputs.

    Receives velocity, density, and temperature from all FlowLayers and runs
    a volumetric Navier-Stokes fluid simulation:
    1. Aggregate inputs from connected FlowLayers
    2. Inject velocity, density (decomposed to V+colour), temperature into 3D volume
    3. Run 3D advection, pressure projection, viscosity, buoyancy, vorticity
    4. Composite 3D density+colour -> 2D output (colour is already baked in)

    The density volume stores scalar brightness (R16F, output resolution).
    The colour volume stores RGB colour (RGBA16F, sim resolution).
    Both are composited by Composite3D into a final RGBA16F 2D output.

    Cross-camera support:
        Each Fluid3DLayer receives a dict of all FlowLayers, enabling
        cross-camera influence (e.g., motion from camera 0 affecting
        fluid in camera 1's output).

    Usage:
        # In RenderManager:
        flow_layers = {i: FlowLayer(...) for i in range(num_cams)}
        fluid3d = Fluid3DLayer(cam_id, data_hub, flow_layers, colors, config)

        # Each frame:
        fluid3d.update()
        density_texture = fluid3d.density
    """

    def __init__(self, cam_id: int, data_hub: DataHub, flow_layers: dict[int, FlowLayer], settings: Fluid3DLayerSettings, color_settings: ColorSettings) -> None:
        """Initialize 3D fluid layer.

        Args:
            cam_id: Camera ID for this layer's output
            data_hub: Data hub for pose data
            flow_layers: Dict of all FlowLayers (cam_id -> FlowLayer)
            settings: Layer configuration
            color_settings: Per-camera colors for density colorization
        """
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._flow_layers: dict[int, FlowLayer] = flow_layers
        self.config: Fluid3DLayerSettings = settings or Fluid3DLayerSettings()
        self._color_settings: ColorSettings = color_settings

        # Simulation engine (swappable at runtime)
        self._fluid_flow: FluidFlow | FluidFlow3D | FluidFlow3DArray = self._create_engine()
        self._alloc_width: int = 0
        self._alloc_height: int = 0
        self._swap_engine_pending: bool = False

        self._visualizer: Visualizer = Visualizer(self.config.visualisation)

        # Defer engine swap to GL thread (callbacks fire from websocket thread)
        self.config.bind(Fluid3DLayerSettings.sim_engine, lambda _: self._request_swap_engine())

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def _create_engine(self) -> FluidFlow | FluidFlow3D | FluidFlow3DArray:
        """Instantiate the selected simulation engine with the shared config."""
        cls = _ENGINE_CLASSES[self.config.sim_engine]
        return cls(self.config.fluid_flow)

    def _request_swap_engine(self) -> None:
        """Request engine swap (deferred to next update on GL thread)."""
        self._swap_engine_pending = True

    def _swap_engine(self) -> None:
        """Swap simulation engine at runtime, preserving allocation state.

        Must run on the GL thread (called from update).
        """
        was_allocated = self._alloc_width > 0
        if was_allocated:
            self._fluid_flow.deallocate()
        self._fluid_flow = self._create_engine()
        if was_allocated:
            self._fluid_flow.allocate(self._alloc_width, self._alloc_height)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Visualization output texture."""
        return self._visualizer.texture

    @property
    def density(self) -> Texture:
        """Composited 2D density (RGBA16F, 3D->2D composite output)."""
        return self._fluid_flow.density

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate 3D fluid simulation buffers.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (formats determined by FluidFlow3D)
        """
        self._alloc_width = width
        self._alloc_height = height
        self._fluid_flow.allocate(width, height)
        self._visualizer.allocate(self._fluid_flow.sim_width, self._fluid_flow.sim_height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._fluid_flow.deallocate()
        self._visualizer.deallocate()

    def reset(self) -> None:
        """Reset fluid simulation state."""
        self._fluid_flow.reset()

    # ========== Processing ==========

    def update(self) -> None:
        """Update 3D fluid simulation with inputs from all flow layers."""
        # Handle deferred engine swap (must run on GL thread)
        if self._swap_engine_pending:
            self._swap_engine_pending = False
            self._swap_engine()

        # Get motion data from pose
        pose: Frame | None = self._data_hub.get_pose(Stage.LERP, self._cam_id)
        similarities: np.ndarray = (
            pose[Similarity].values if pose is not None and Similarity in pose
            else np.full((self.config.num_players,), 0.0)
        )
        motion_gates: np.ndarray = (
            pose[MotionGate].values if pose is not None and MotionGate in pose
            else np.full((self.config.num_players,), 0.0)
        )
        motion: float = pose[AngleMotion].value if pose is not None and AngleMotion in pose else 0.0
        m_s = similarities

        # self.config.fluid_flow.density.fade_time = 18.0 - (pow(motion, 2.0) * 14.0)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Aggregate inputs from all flow layers (cross-camera)
        vel_strength: float
        den_strength: float
        for cam_id, flow_layer in self._flow_layers.items():
            if cam_id == self._cam_id:
                vel_strength = motion
                den_strength = motion
            else:
                vel_strength = m_s[cam_id]
                den_strength = m_s[cam_id]

            # Add velocity from each flow layer
            self._fluid_flow.add_velocity(flow_layer.velocity, vel_strength)

            # Add Z-velocity from motion magnitude (3D engines only)
            if isinstance(self._fluid_flow, (FluidFlow3D, FluidFlow3DArray)):
                self._fluid_flow.add_velocity_z(flow_layer.magnitude, vel_strength * 0.1)

            # Add density to per-camera channel (R=cam0, G=cam1, B=cam2, A=cam3)
            channel: int = cam_id % 4
            # self._fluid_flow.add_density_channel(flow_layer.magnitude, channel, den_strength)
            self._fluid_flow.add_density(flow_layer.density, den_strength)

            # Add temperature from each flow layer
            self._fluid_flow.add_temperature(flow_layer.temperature, 1.0)

        # Snapshot sim dims before update for resize detection
        prev_sw, prev_sh = self._fluid_flow.sim_width, self._fluid_flow.sim_height

        # Update 3D fluid simulation (handles its own reallocation internally)
        self._fluid_flow.update()

        # Resize visualizer if sim dims changed
        if self._fluid_flow.sim_width != prev_sw or self._fluid_flow.sim_height != prev_sh:
            self._visualizer.allocate(self._fluid_flow.sim_width, self._fluid_flow.sim_height)

        # Update visualization
        self._visualizer.update(self._get_draw_texture())

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
        textures = {
            Fluid3DDrawMode.DENSITY: self._fluid_flow.density,
            Fluid3DDrawMode.DENSITY_RAW: self._fluid_flow.density,
        }
        return textures.get(self.config.draw_mode, self._fluid_flow.density)

    def _colorize_density(self) -> None:
        """Removed: colour is now baked into the density volume during injection.

        The RGBA16F colour volume at sim resolution is composited with the R16F
        density volume by Composite3D, producing an already-coloured 2D output.
        No post-processing colorisation step is required.
        """
        pass  # No-op — kept for API compatibility
