"""Fluid Layer - Navier-Stokes fluid simulation with cross-camera flow inputs."""

# Standard library imports
from __future__ import annotations
from enum import IntEnum, auto
from dataclasses import dataclass, field

# Third-party imports
from OpenGL.GL import *  # type: ignore
import numpy as np

# Local application imports
from modules.gl import Texture, Style, Fbo
from modules.render.layers.LayerBase import LayerBase, Blit
from modules.DataHub import DataHub, Stage
from modules.pose.Frame import Frame, MotionGate, Similarity

from modules.flow import Visualizer, VisualisationFieldConfig
from modules.flow.fluid import FluidFlow, FluidFlowConfig
from modules.render.shaders import DensityColorize

from modules.utils.HotReloadMethods import HotReloadMethods

from .FlowLayer import FlowLayer


class FluidDrawMode(IntEnum):
    """Draw modes for FluidLayer.

    Outputs from fluid simulation:
        VELOCITY - Advected velocity field (RG16F)
        DENSITY - Colorized density field (RGBA16F, colored by track)
        DENSITY_RAW - Raw density field before colorization (RGBA16F)
        PRESSURE - Pressure field (R32F)
        TEMPERATURE - Temperature field (R32F)
        DIVERGENCE - Velocity divergence (R32F)
        VORTICITY - Vorticity curl (R32F)
        BUOYANCY - Buoyancy force (RG32F)
        OBSTACLE - Obstacle mask (R8)
    """
    VELOCITY = 0
    DENSITY = auto()
    DENSITY_RAW = auto()
    PRESSURE = auto()
    TEMPERATURE = auto()
    DIVERGENCE = auto()
    VORTICITY = auto()
    BUOYANCY = auto()
    OBSTACLE = auto()


@dataclass
class FluidLayerConfig:
    """Configuration for FluidLayer (fluid simulation)."""
    fps: float = 60.0
    num_players: int = 3
    draw_mode: FluidDrawMode = FluidDrawMode.DENSITY
    blend_mode: Style.BlendMode = Style.BlendMode.ADDITIVE
    simulation_scale: float = 0.5

    # Nested configs
    visualisation: VisualisationFieldConfig = field(default_factory=VisualisationFieldConfig)
    fluid_flow: FluidFlowConfig = field(default_factory=FluidFlowConfig)


class FluidLayer(LayerBase):
    """Fluid simulation layer with cross-camera flow inputs.

    Receives velocity, density, and temperature from all FlowLayers and runs
    Navier-Stokes fluid simulation:
    1. Aggregate inputs from connected FlowLayers
    2. Add velocity, density, temperature to fluid
    3. Run advection, pressure projection, viscosity, buoyancy
    4. Output simulated fields

    Cross-camera support:
        Each FluidLayer receives a dict of all FlowLayers, enabling
        cross-camera influence (e.g., motion from camera 0 affecting
        fluid in camera 1's output).

    Usage:
        # In RenderManager:
        flow_layers = {i: FlowLayer(...) for i in range(num_cams)}
        fluid = FluidLayer(cam_id, data_hub, flow_layers, colors, config)

        # Each frame:
        fluid.update()
        density_texture = fluid.density
    """

    def __init__(self, cam_id: int, data_hub: DataHub, flow_layers: dict[int, FlowLayer], colors: list[tuple[float, float, float, float]], config: FluidLayerConfig | None = None) -> None:
        """Initialize fluid layer.

        Args:
            cam_id: Camera ID for this layer's output
            data_hub: Data hub for pose data
            flow_layers: Dict of all FlowLayers (cam_id -> FlowLayer)
            colors: Per-camera colors for density colorization (passed from RenderManager)
            config: Layer configuration
        """
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._flow_layers: dict[int, FlowLayer] = flow_layers
        self.config: FluidLayerConfig = config or FluidLayerConfig()

        self._delta_time: float = 1 / self.config.fps

        self._fluid_flow: FluidFlow = FluidFlow(self.config.simulation_scale, self.config.fluid_flow)
        self._visualizer: Visualizer = Visualizer(self.config.visualisation)

        # Colorization resources (rendering concern, not simulation)
        self._colorized_fbo: Fbo = Fbo()
        self._density_colorize_shader: DensityColorize = DensityColorize()

        # Track colors for per-channel density colorization (up to 4 channels)
        self._colors: list[tuple[float, float, float, float]] = list(colors[:4])
        # Pad with transparent black if fewer than 4 colors
        while len(self._colors) < 4:
            self._colors.append((0.0, 0.0, 0.0, 0.0))

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Visualization output texture."""
        return self._visualizer.texture

    @property
    def velocity(self) -> Texture:
        """Advected velocity field (RG16F)."""
        return self._fluid_flow.velocity

    @property
    def density(self) -> Texture:
        """Advected density field (RGBA16F)."""
        return self._fluid_flow.density

    @property
    def pressure(self) -> Texture:
        """Pressure field (R32F)."""
        return self._fluid_flow.pressure

    @property
    def temperature(self) -> Texture:
        """Temperature field (R32F)."""
        return self._fluid_flow.temperature

    @property
    def divergence(self) -> Texture:
        """Velocity divergence (R32F)."""
        return self._fluid_flow.divergence

    @property
    def vorticity(self) -> Texture:
        """Vorticity curl (R32F)."""
        return self._fluid_flow.vorticity_curl

    @property
    def buoyancy(self) -> Texture:
        """Buoyancy force (RG32F)."""
        return self._fluid_flow.buoyancy

    @property
    def obstacle(self) -> Texture:
        """Obstacle mask (R8)."""
        return self._fluid_flow.obstacle

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate fluid simulation buffers.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (formats determined by FluidFlow)
        """
        # Fluid simulation: low-res simulation, high-res density output
        sim_width = int(width * self.config.simulation_scale)
        sim_height = int(height * self.config.simulation_scale)

        self._fluid_flow.allocate(sim_width, sim_height, width, height)
        self._visualizer.allocate(sim_width, sim_height)

        # Allocate colorization FBO at simulation resolution
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
        """Update fluid simulation with inputs from all flow layers."""
        # Configuration overrides (for hot-reload testing)
        self.config.fluid_flow.vel_speed = 0.1
        self.config.fluid_flow.vel_decay = 6.0

        self.config.fluid_flow.vel_vorticity = 10
        self.config.fluid_flow.vel_vorticity_radius = 3.0
        self.config.fluid_flow.vel_viscosity = 8.0
        self.config.fluid_flow.vel_viscosity_iter = 40

        self.config.fluid_flow.den_speed = 1.1
        self.config.fluid_flow.den_decay = 12

        self.config.fluid_flow.tmp_speed = 0.33
        self.config.fluid_flow.tmp_decay = 3.0

        self.config.fluid_flow.prs_speed = 0.0
        self.config.fluid_flow.prs_decay = 0.0
        self.config.fluid_flow.prs_iterations = 40

        self.config.fluid_flow.tmp_buoyancy = 0.0
        self.config.fluid_flow.tmp_weight = -10.0

        self.config.draw_mode = FluidDrawMode.DENSITY
        self.config.visualisation.toggle_scalar = True
        self.config.visualisation.spacing = 4.0
        self.config.visualisation.element_width = 0.5


        # Get motion data from pose
        pose: Frame | None = self._data_hub.get_pose(Stage.LERP, self._cam_id)
        similarities: np.ndarray = pose.similarity.values if pose is not None else np.full((self.config.num_players,), 0.0)
        motion_gates: np.ndarray = pose.motion_gate.values if pose is not None else np.full((self.config.num_players,), 0.0)
        motion: float = pose.angle_motion.value if pose is not None else 0.0
        m_s = similarities * motion_gates  # Modulate similarity by motion gate
        # print(f"FluidLayer cam_id {self._cam_id} similarities: {similarities}, motion_gates: {motion_gates}")  # DEBUG

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Aggregate inputs from all flow layers (cross-camera)
        vel_strength: float
        den_strength: float
        for cam_id, flow_layer in self._flow_layers.items():
            if cam_id == self._cam_id:
                vel_strength = 0.1
                den_strength = motion
            else:
                vel_strength = 0.1 * (m_s[cam_id]) # m_s[cam_id]  # Cross-camera influence modulated by similarity and motion gate
                den_strength = (m_s[cam_id])   # Cross-camera influence modulated by similarity, motion gate, and motion value

            # Add velocity from each flow layer
            self._fluid_flow.add_velocity(flow_layer.velocity, vel_strength)

            # Add density to per-camera channel (R=cam0, G=cam1, B=cam2, A=cam3)
            channel: int = cam_id % 4  # Map camera to RGBA channel
            self._fluid_flow.add_density_channel(flow_layer.density, channel, den_strength)
            self._fluid_flow.clamp_density(0.0, 1.1)


            # Add temperature from each flow layer
            self._fluid_flow.add_temperature(flow_layer.temperature, 0.0)

        # Update fluid simulation
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
            FluidDrawMode.VELOCITY: self._fluid_flow.velocity,
            FluidDrawMode.DENSITY: self._colorized_fbo.texture,
            FluidDrawMode.DENSITY_RAW: self._fluid_flow.density,
            FluidDrawMode.PRESSURE: self._fluid_flow.pressure,
            FluidDrawMode.TEMPERATURE: self._fluid_flow.temperature,
            FluidDrawMode.DIVERGENCE: self._fluid_flow.divergence,
            FluidDrawMode.VORTICITY: self._fluid_flow.vorticity_curl,
            FluidDrawMode.BUOYANCY: self._fluid_flow.buoyancy,
            FluidDrawMode.OBSTACLE: self._fluid_flow.obstacle,
        }
        return textures.get(self.config.draw_mode, self._fluid_flow.density)

    def _colorize_density(self) -> None:
        """Colorize density channels with track colors.

        Maps each RGBA density channel to a corresponding track color and
        composites them additively.
        """
        self._colorized_fbo.begin()
        self._density_colorize_shader.use(self._fluid_flow.density, self._colors)
        self._colorized_fbo.end()
