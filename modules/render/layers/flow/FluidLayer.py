"""Fluid Layer - Navier-Stokes fluid simulation with cross-camera flow inputs."""

# Standard library imports
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import *  # type: ignore
import numpy as np

# Local application imports
from modules.gl import Texture, Style, Fbo
from modules.render.layers.LayerBase import LayerBase, Blit
from modules.data_hub import DataHub, Stage
from modules.pose.Frame import Frame

from modules.settings import Field, Settings
from modules.flow import Visualizer, VisualisationFieldConfig, FluidFlow, FluidFlowConfig
from modules.render.shaders import DensityColorize

from modules.utils.HotReloadMethods import HotReloadMethods

from .FlowLayer import FlowLayer
from modules.render.color_settings import ColorSettings


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


class FluidLayerSettings(Settings):
    """Configuration for FluidLayer (fluid simulation)."""
    num_players: Field[int] =               Field(3, min=1, max=8, access=Field.INIT)
    draw_mode: Field[FluidDrawMode] =       Field(FluidDrawMode.DENSITY)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)

    fluid_flow:    FluidFlowConfig
    visualisation: VisualisationFieldConfig


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

    def __init__(self, cam_id: int, data_hub: DataHub, flow_layers: dict[int, FlowLayer], settings: FluidLayerSettings, color_settings: ColorSettings) -> None:
        """Initialize fluid layer.

        Args:
            cam_id: Camera ID for this layer's output
            data_hub: Data hub for pose data
            flow_layers: Dict of all FlowLayers (cam_id -> FlowLayer)
            color_settings: Shared color settings for density colorization
            config: Layer configuration
        """
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._flow_layers: dict[int, FlowLayer] = flow_layers
        self.settings: FluidLayerSettings = settings
        self._color_settings: ColorSettings = color_settings

        self._fluid_flow: FluidFlow = FluidFlow(self.settings.fluid_flow)
        self._visualizer: Visualizer = Visualizer(self.settings.visualisation)

        # Colorization resources (rendering concern, not simulation)
        self._colorized_fbo: Fbo = Fbo()
        self._density_colorize_shader: DensityColorize = DensityColorize()

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
        self._fluid_flow.allocate(width, height)
        self._visualizer.allocate(self._fluid_flow.sim_width, self._fluid_flow.sim_height)

        # Colorization FBO at density (full output) resolution
        self._colorized_fbo.allocate(width, height, GL_RGBA16F)
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
        # Get motion data from pose
        pose: Frame | None = self._data_hub.get_pose(Stage.LERP, self._cam_id)
        similarities: np.ndarray = pose.similarity.values if pose is not None else np.full((self.settings.num_players,), 0.0)
        motion_gates: np.ndarray = pose.motion_gate.values if pose is not None else np.full((self.settings.num_players,), 0.0)
        motion: float = pose.angle_motion.value if pose is not None else 0.0
        m_s = similarities

        self.settings.fluid_flow.density.fade_time = 18.0 - (pow(motion, 2.0) * 14.0)  # More motion = faster decay

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

            # Add density to per-camera channel (R=cam0, G=cam1, B=cam2, A=cam3)
            channel: int = cam_id % 4
            self._fluid_flow.add_density_channel(flow_layer.magnitude, channel, den_strength)

            # Add temperature from each flow layer
            self._fluid_flow.add_temperature(flow_layer.temperature, 1.0)

        # Snapshot sim dims before update for resize detection
        prev_sw, prev_sh = self._fluid_flow.sim_width, self._fluid_flow.sim_height

        # Update fluid simulation (handles its own reallocation internally)
        self._fluid_flow.update()

        # Resize visualizer if sim dims changed
        if self._fluid_flow.sim_width != prev_sw or self._fluid_flow.sim_height != prev_sh:
            self._visualizer.allocate(self._fluid_flow.sim_width, self._fluid_flow.sim_height)

        # Colorize density channels with track colors
        self._colorize_density()

        # Update visualization
        self._visualizer.update(self._get_draw_texture())

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        Style.push_style()
        Style.set_blend_mode(self.settings.blend_mode)
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
        return textures.get(self.settings.draw_mode, self._fluid_flow.density)

    def _colorize_density(self) -> None:
        """Colorize density channels with track colors.

        Maps each RGBA density channel to a corresponding track color and
        composites them additively.
        """
        # Build padded color list (up to 4 channels) from settings each frame
        colors: list[tuple[float, float, float, float]] = list(self._color_settings.track_color_tuples[:4])
        while len(colors) < 4:
            colors.append((0.0, 0.0, 0.0, 0.0))

        self._density_colorize_shader.reload()
        self._colorized_fbo.begin()
        self._density_colorize_shader.use(self._fluid_flow.density, colors)
        self._colorized_fbo.end()
