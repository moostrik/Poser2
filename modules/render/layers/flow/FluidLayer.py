"""Fluid Layer - Navier-Stokes fluid simulation with cross-camera flow inputs."""

# Standard library imports
from __future__ import annotations
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Blit
from modules.DataHub import DataHub

from modules.flow import Visualizer, VisualisationFieldConfig
from modules.flow.fluid import FluidFlow, FluidFlowConfig

from modules.utils.HotReloadMethods import HotReloadMethods

if TYPE_CHECKING:
    from .FlowLayer import FlowLayer


class FluidDrawMode(IntEnum):
    """Draw modes for FluidLayer.

    Outputs from fluid simulation:
        VELOCITY - Advected velocity field (RG16F)
        DENSITY - Advected density field (RGBA16F)
        PRESSURE - Pressure field (R32F)
        TEMPERATURE - Temperature field (R32F)
        DIVERGENCE - Velocity divergence (R32F)
        VORTICITY - Vorticity curl (R32F)
        BUOYANCY - Buoyancy force (RG32F)
        OBSTACLE - Obstacle mask (R8)
    """
    VELOCITY = 0
    DENSITY = auto()
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
    draw_mode: FluidDrawMode = FluidDrawMode.DENSITY
    blend_mode: Style.BlendMode = Style.BlendMode.ADDITIVE
    simulation_scale: float = 0.25

    # Scale factors for inputs
    velocity_scale: float = 1.0
    density_scale: float = 1.0
    temperature_scale: float = 1.0

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
        fluid = FluidLayer(cam_id, data_hub, flow_layers, config)

        # Each frame:
        fluid.update()
        density_texture = fluid.density
    """

    def __init__(self, cam_id: int, data_hub: DataHub, flow_layers: dict[int, FlowLayer], config: FluidLayerConfig | None = None ) -> None:
        """Initialize fluid layer.

        Args:
            cam_id: Camera ID for this layer's output
            data_hub: Data hub for pose data
            flow_layers: Dict of all FlowLayers (cam_id -> FlowLayer)
            config: Layer configuration
        """
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._flow_layers: dict[int, FlowLayer] = flow_layers
        self.config: FluidLayerConfig = config or FluidLayerConfig()

        self._delta_time: float = 1 / self.config.fps

        self._fluid_flow: FluidFlow = FluidFlow(self.config.simulation_scale, self.config.fluid_flow)
        self._visualizer: Visualizer = Visualizer(self.config.visualisation)

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

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._fluid_flow.deallocate()
        self._visualizer.deallocate()

    def reset(self) -> None:
        """Reset fluid simulation state."""
        self._fluid_flow.reset()

    # ========== Processing ==========

    def update(self) -> None:
        """Update fluid simulation with inputs from all flow layers."""
        # Configuration overrides (for hot-reload testing)
        self.config.fluid_flow.vel_speed = 0.9
        self.config.fluid_flow.vel_decay = 6.0
        self.config.fluid_flow.vel_vorticity = 0.0
        self.config.fluid_flow.vel_vorticity_radius = 20.0
        self.config.fluid_flow.vel_viscosity = 10
        self.config.fluid_flow.vel_viscosity_iter = 20

        self.config.fluid_flow.den_speed = 1.0
        self.config.fluid_flow.den_decay = 6.0
        self.config.fluid_flow.tmp_speed = 0.33
        self.config.fluid_flow.tmp_decay = 3.0
        self.config.fluid_flow.prs_speed = 0.0
        self.config.fluid_flow.prs_decay = 0.0
        self.config.fluid_flow.prs_iterations = 40

        self.config.fluid_flow.tmp_buoyancy = 0.0
        self.config.fluid_flow.tmp_weight = -10.0

        self.config.draw_mode = FluidDrawMode.DENSITY

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Aggregate inputs from all flow layers (cross-camera)
        for cam_id, flow_layer in self._flow_layers.items():
            if cam_id != self._cam_id:
                continue

            # Scale factor: own camera gets full strength, others can be scaled
            scale = 1.0 if cam_id == self._cam_id else 1.0

            # Add velocity from each flow layer
            velocity_strength = self.config.velocity_scale * scale
            self._fluid_flow.add_velocity(flow_layer.velocity, 1.0)

            # Add density from each flow layer
            density_strength = self.config.density_scale * scale
            self._fluid_flow.add_density(flow_layer.density)

            # Add temperature from each flow layer
            temperature_strength = self.config.temperature_scale * scale
            self._fluid_flow.add_temperature(flow_layer.temperature)

        # Update fluid simulation
        self._fluid_flow.update(self._delta_time)

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
        textures = {
            FluidDrawMode.VELOCITY: self._fluid_flow.velocity,
            FluidDrawMode.DENSITY: self._fluid_flow.density,
            FluidDrawMode.PRESSURE: self._fluid_flow.pressure,
            FluidDrawMode.TEMPERATURE: self._fluid_flow.temperature,
            FluidDrawMode.DIVERGENCE: self._fluid_flow.divergence,
            FluidDrawMode.VORTICITY: self._fluid_flow.vorticity_curl,
            FluidDrawMode.BUOYANCY: self._fluid_flow.buoyancy,
            FluidDrawMode.OBSTACLE: self._fluid_flow.obstacle,
        }
        return textures.get(self.config.draw_mode, self._fluid_flow.density)
