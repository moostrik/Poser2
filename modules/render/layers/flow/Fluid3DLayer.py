"""Fluid3D Layer - 3D Navier-Stokes fluid simulation with cross-camera flow inputs.

Volumetric extension of FluidLayer. Receives 2D velocity, density, and
temperature from all FlowLayers and runs a full 3D Navier-Stokes simulation
with gaussian depth injection.  The composited 2D density output feeds into
the same DensityColorize pipeline as FluidLayer.
"""

# Standard library imports
from __future__ import annotations
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import *  # type: ignore
import numpy as np

# Local application imports
from modules.gl import Texture, Style, Fbo
from modules.render.layers.LayerBase import LayerBase, Blit
from modules.DataHub import DataHub, Stage
from modules.pose.Frame import Frame, MotionGate, Similarity

from modules.settings import Setting, BaseSettings
from modules.flow import Visualizer, VisualisationFieldConfig
from modules.flow.fluid3d import FluidFlow3D, FluidFlow3DConfig
from modules.render.shaders import DensityColorize

from modules.utils.HotReloadMethods import HotReloadMethods

from .FlowLayer import FlowLayer


class Fluid3DDrawMode(IntEnum):
    """Draw modes for Fluid3DLayer.

    Only fields with a 2D composited output are exposed:
        DENSITY      - Colorized density (composited 3D->2D, then per-channel colored)
        DENSITY_RAW  - Raw composited density before colorization (RGBA16F)
    """
    DENSITY = 0
    DENSITY_RAW = auto()


class Fluid3DLayerSettings(BaseSettings):
    """Configuration for Fluid3DLayer (3D fluid simulation)."""
    fps = Setting(30.0, min=1.0, max=240.0)
    num_players = Setting(3, min=1, max=8)
    draw_mode = Setting(Fluid3DDrawMode.DENSITY)
    blend_mode = Setting(Style.BlendMode.ADD)
    simulation_scale = Setting(0.5, min=0.1, max=2.0)

    visualisation: VisualisationFieldConfig
    fluid_flow:    FluidFlow3DConfig


class Fluid3DLayer(LayerBase):
    """3D fluid simulation layer with cross-camera flow inputs.

    Receives velocity, density, and temperature from all FlowLayers and runs
    a volumetric Navier-Stokes fluid simulation:
    1. Aggregate inputs from connected FlowLayers
    2. Inject velocity, per-channel density, temperature into 3D volume
    3. Run 3D advection, pressure projection, viscosity, buoyancy, vorticity
    4. Composite 3D density -> 2D output
    5. Colorize per-channel density with track colors

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

    def __init__(self, cam_id: int, data_hub: DataHub,
                 flow_layers: dict[int, FlowLayer],
                 colors: list[tuple[float, float, float, float]],
                 config: Fluid3DLayerSettings | None = None) -> None:
        """Initialize 3D fluid layer.

        Args:
            cam_id: Camera ID for this layer's output
            data_hub: Data hub for pose data
            flow_layers: Dict of all FlowLayers (cam_id -> FlowLayer)
            colors: Per-camera colors for density colorization
            config: Layer configuration
        """
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._flow_layers: dict[int, FlowLayer] = flow_layers
        self.config: Fluid3DLayerSettings = config or Fluid3DLayerSettings()

        self._delta_time: float = 1 / self.config.fps

        self._fluid_flow: FluidFlow3D = FluidFlow3D(
            self.config.simulation_scale, self.config.fluid_flow
        )
        self._visualizer: Visualizer = Visualizer(self.config.visualisation)

        # Colorization resources
        self._colorized_fbo: Fbo = Fbo()
        self._density_colorize_shader: DensityColorize = DensityColorize()

        # Track colors for per-channel density colorization (up to 4 channels)
        self._colors: list[tuple[float, float, float, float]] = list(colors[:4])
        while len(self._colors) < 4:
            self._colors.append((0.0, 0.0, 0.0, 0.0))

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

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
        sim_width = int(width * self.config.simulation_scale)
        sim_height = int(height * self.config.simulation_scale)

        self._fluid_flow.allocate(sim_width, sim_height, width, height)
        self._visualizer.allocate(sim_width, sim_height)

        # Colorization FBO at simulation resolution
        self._colorized_fbo.allocate(sim_width, sim_height, GL_RGBA16F)
        self._density_colorize_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._fluid_flow.deallocate()
        self._visualizer.deallocate()
        self._colorized_fbo.deallocate()
        self._density_colorize_shader.deallocate()

    def reset(self) -> None:
        """Reset fluid simulation state."""
        self._fluid_flow.reset()

    # ========== Processing ==========

    def update(self) -> None:
        """Update 3D fluid simulation with inputs from all flow layers."""
        # Get motion data from pose
        pose: Frame | None = self._data_hub.get_pose(Stage.LERP, self._cam_id)
        similarities: np.ndarray = (
            pose.similarity.values if pose is not None
            else np.full((self.config.num_players,), 0.0)
        )
        motion_gates: np.ndarray = (
            pose.motion_gate.values if pose is not None
            else np.full((self.config.num_players,), 0.0)
        )
        motion: float = pose.angle_motion.value if pose is not None else 0.0
        m_s = similarities

        self.config.fluid_flow.den_lifetime = 18.0 - (pow(motion, 2.0) * 14.0)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Aggregate inputs from all flow layers (cross-camera)
        vel_strength: float
        den_strength: float
        for cam_id, flow_layer in self._flow_layers.items():
            if cam_id == self._cam_id:
                vel_strength = self._delta_time * motion
                den_strength = motion * 0.02
            else:
                vel_strength = self._delta_time * (m_s[cam_id])
                den_strength = 0.02 * (m_s[cam_id])

            # Add velocity from each flow layer
            self._fluid_flow.add_velocity(flow_layer.velocity, vel_strength)

            # Add density to per-camera channel (R=cam0, G=cam1, B=cam2, A=cam3)
            channel: int = cam_id % 4
            self._fluid_flow.add_density_channel(
                flow_layer.magnitude, channel, den_strength
            )
            self._fluid_flow.clamp_density(0.0, 1.2)

            # Add temperature from each flow layer
            self._fluid_flow.add_temperature(flow_layer.temperature, 0.0)

        # Update 3D fluid simulation
        self._fluid_flow.update(self._delta_time)

        # Colorize density channels with track colors
        self._colorize_density()

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
            Fluid3DDrawMode.DENSITY: self._colorized_fbo.texture,
            Fluid3DDrawMode.DENSITY_RAW: self._fluid_flow.density,
        }
        return textures.get(self.config.draw_mode, self._fluid_flow.density)

    def _colorize_density(self) -> None:
        """Colorize density channels with track colors.

        Maps each RGBA density channel to a corresponding track color and
        composites them additively. Works unchanged because FluidFlow3D.density
        returns a 2D Texture (composited from 3D volume).
        """
        self._density_colorize_shader.reload()
        self._colorized_fbo.begin()
        self._density_colorize_shader.use(self._fluid_flow.density, self._colors)
        self._colorized_fbo.end()
