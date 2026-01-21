"""Fluid Flow - 2D Navier-Stokes fluid simulation.

Implements velocity, density, temperature, and pressure fields with:
- Semi-Lagrangian advection
- Pressure projection (incompressibility)
- Viscosity diffusion
- Vorticity confinement
- Buoyancy forces

Ported from ofxFlowTools ftFluidFlow.h/cpp
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, SwapFbo, Fbo
from .. import FlowBase, FlowConfigBase, FlowUtil
from .shaders import (
    Advect, Divergence, Gradient, JacobiPressure, JacobiDiffusion,
    VorticityCurl, VorticityForce, Buoyancy, ObstacleOffset, AddBoolean
)

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class FluidFlowConfig(FlowConfigBase):
    """Configuration for fluid simulation."""

    # Velocity parameters
    vel_speed: float = field(
        default=0.3,
        metadata={"min": -100, "max": 100.0, "label": "Velocity Speed",
                  "description": "Velocity advection speed multiplier"}
    )
    vel_dissipation: float = field(
        default=0.1,
        metadata={"min": -100, "max": 100.0, "label": "Velocity Dissipation",
                  "description": "Velocity energy loss per frame"}
    )
    vel_vorticity: float = field(
        default=0.0,
        metadata={"min": -100, "max": 100, "label": "Vorticity",
                  "description": "Vortex confinement strength (turbulence)"}
    )
    vel_viscosity: float = field(
        default=0.0,
        metadata={"min": -100, "max": 100, "label": "Viscosity",
                  "description": "Fluid thickness/diffusion"}
    )
    vel_viscosity_iter: int = field(
        default=20,
        metadata={"min": 0, "max": 100, "label": "Viscosity Iterations",
                  "description": "Jacobi iterations for diffusion solve"}
    )

    # Pressure parameters
    prs_speed: float = field(
        default=0.0,
        metadata={"min": -100, "max": 100, "label": "Pressure Speed",
                  "description": "Pressure advection speed (usually 0)"}
    )
    prs_dissipation: float = field(
        default=0.1,
        metadata={"min": -100, "max": 100, "label": "Pressure Dissipation",
                  "description": "Pressure decay rate"}
    )
    prs_iterations: int = field(
        default=40,
        metadata={"min": 0, "max": 100, "label": "Pressure Iterations",
                  "description": "Jacobi iterations for pressure solve"}
    )

    # Density parameters
    den_speed: float = field(
        default=0.3,
        metadata={"min": -100, "max": 100.0, "label": "Density Speed",
                  "description": "Density advection speed"}
    )
    den_dissipation: float = field(
        default=0.1,
        metadata={"min": -100, "max": 100.0, "label": "Density Dissipation",
                  "description": "Density fade rate"}
    )

    # Temperature parameters
    tmp_speed: float = field(
        default=0.3,
        metadata={"min": -100, "max": 100.0, "label": "Temperature Speed",
                  "description": "Temperature advection speed"}
    )
    tmp_dissipation: float = field(
        default=0.1,
        metadata={"min": -100, "max": 100.0, "label": "Temperature Dissipation",
                  "description": "Temperature dissipation rate"}
    )
    tmp_buoyancy: float = field(
        default=0.0,
        metadata={"min": -100, "max": 100, "label": "Buoyancy",
                  "description": "Buoyancy force strength (sigma)"}
    )
    tmp_weight: float = field(
        default=0.2,
        metadata={"min": -100, "max": 100, "label": "Temperature Weight",
                  "description": "Density weight in buoyancy"}
    )
    tmp_ambient: float = field(
        default=0.2,
        metadata={"min": -100, "max": 100, "label": "Ambient Temperature",
                  "description": "Reference/ambient temperature"}
    )


class FluidFlow(FlowBase):
    """2D Navier-Stokes fluid simulation.

    Inherits from FlowBase:
        - _input_fbo → velocity field (RG32F)
        - _output_fbo → density field (RGBA32F)

    Additional fields:
        - temperature (R32F)
        - pressure (R32F)
        - obstacles (R8)

    Update pipeline:
        1. Advect density, velocity, temperature, pressure
        2. Apply viscosity (if enabled)
        3. Apply vorticity confinement (if enabled)
        4. Apply buoyancy (if enabled)
        5. Compute divergence and solve for pressure
        6. Subtract pressure gradient (make divergence-free)
    """

    def __init__(self, config: FluidFlowConfig | None = None) -> None:
        super().__init__()

        self.config: FluidFlowConfig = config or FluidFlowConfig()

        # Define formats for FlowBase
        self._input_internal_format = GL_RG32F     # Velocity (inherited as _input_fbo)
        self._output_internal_format = GL_RGBA32F  # Density (inherited as _output_fbo)

        # Additional simulation fields (SwapFbo for ping-pong)
        self._temperature_fbo: SwapFbo = SwapFbo()
        self._pressure_fbo: SwapFbo = SwapFbo()
        self._obstacle_fbo: SwapFbo = SwapFbo()

        # Intermediate result FBOs (single buffer, no ping-pong)
        self._divergence_fbo: Fbo = Fbo()
        self._vorticity_curl_fbo: Fbo = Fbo()
        self._vorticity_force_fbo: Fbo = Fbo()
        self._buoyancy_fbo: Fbo = Fbo()
        self._obstacle_offset_fbo: Fbo = Fbo()

        # Simulation parameters
        self._grid_scale: int = 1  # Always 1 in standard setup
        self._simulation_width: int = 0
        self._simulation_height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0

        # Shaders
        self._advect_shader: Advect = Advect()
        self._divergence_shader: Divergence = Divergence()
        self._gradient_shader: Gradient = Gradient()
        self._jacobi_pressure_shader: JacobiPressure = JacobiPressure()
        self._jacobi_diffusion_shader: JacobiDiffusion = JacobiDiffusion()
        self._vorticity_curl_shader: VorticityCurl = VorticityCurl()
        self._vorticity_force_shader: VorticityForce = VorticityForce()
        self._buoyancy_shader: Buoyancy = Buoyancy()
        self._obstacle_offset_shader: ObstacleOffset = ObstacleOffset()
        self._add_boolean_shader: AddBoolean = AddBoolean()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Properties (Domain-specific API) ==========

    @property
    def velocity(self) -> Texture:
        """RG32F velocity field."""
        return self._input

    @property
    def density(self) -> Texture:
        """RGBA32F density/color field."""
        return self._output

    @property
    def temperature(self) -> Texture:
        """R32F temperature field."""
        return self._temperature_fbo.texture

    @property
    def pressure(self) -> Texture:
        """R32F pressure field."""
        return self._pressure_fbo.texture

    @property
    def divergence(self) -> Texture:
        """R32F divergence field (intermediate result)."""
        return self._divergence_fbo.texture

    @property
    def vorticity_curl(self) -> Texture:
        """R32F vorticity curl field."""
        return self._vorticity_curl_fbo.texture

    @property
    def buoyancy(self) -> Texture:
        """RG32F buoyancy force field."""
        return self._buoyancy_fbo.texture

    @property
    def obstacle(self) -> Texture:
        """R8 obstacle mask."""
        return self._obstacle_fbo.texture

    # ========== Allocation ==========

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate fluid simulation FBOs.

        Args:
            width: Simulation resolution (velocity, temperature, pressure, obstacles)
            height: Simulation resolution
            output_width: Density resolution (defaults to width for single-resolution)
            output_height: Density resolution
        """
        self._simulation_width = width
        self._simulation_height = height
        self._density_width = output_width if output_width is not None else width
        self._density_height = output_height if output_height is not None else height

        # Allocate base FBOs (velocity RG32F, density RGBA32F)
        super().allocate(width, height, self._density_width, self._density_height)

        # Allocate simulation fields
        self._temperature_fbo.allocate(width, height, GL_R32F)
        FlowUtil.zero(self._temperature_fbo)

        self._pressure_fbo.allocate(width, height, GL_R32F)
        FlowUtil.zero(self._pressure_fbo)

        self._obstacle_fbo.allocate(width, height, GL_R8)
        FlowUtil.zero(self._obstacle_fbo)

        # Allocate intermediate FBOs
        self._divergence_fbo.allocate(width, height, GL_R32F)
        FlowUtil.zero(self._divergence_fbo)

        self._vorticity_curl_fbo.allocate(width, height, GL_R32F)
        FlowUtil.zero(self._vorticity_curl_fbo)

        self._vorticity_force_fbo.allocate(width, height, GL_RG32F)
        FlowUtil.zero(self._vorticity_force_fbo)

        self._buoyancy_fbo.allocate(width, height, GL_RG32F)
        FlowUtil.zero(self._buoyancy_fbo)

        self._obstacle_offset_fbo.allocate(width, height, GL_RGBA8)
        FlowUtil.zero(self._obstacle_offset_fbo)

        # Allocate shaders
        self._advect_shader.allocate()
        self._divergence_shader.allocate()
        self._gradient_shader.allocate()
        self._jacobi_pressure_shader.allocate()
        self._jacobi_diffusion_shader.allocate()
        self._vorticity_curl_shader.allocate()
        self._vorticity_force_shader.allocate()
        self._buoyancy_shader.allocate()
        self._obstacle_offset_shader.allocate()
        self._add_boolean_shader.allocate()

        # Initialize obstacles with border
        self._init_obstacle()

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self._temperature_fbo.deallocate()
        self._pressure_fbo.deallocate()
        self._obstacle_fbo.deallocate()
        self._divergence_fbo.deallocate()
        self._vorticity_curl_fbo.deallocate()
        self._vorticity_force_fbo.deallocate()
        self._buoyancy_fbo.deallocate()
        self._obstacle_offset_fbo.deallocate()

        self._advect_shader.deallocate()
        self._divergence_shader.deallocate()
        self._gradient_shader.deallocate()
        self._jacobi_pressure_shader.deallocate()
        self._jacobi_diffusion_shader.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._buoyancy_shader.deallocate()
        self._obstacle_offset_shader.deallocate()
        self._add_boolean_shader.deallocate()

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        super().reset()
        FlowUtil.zero(self._temperature_fbo)
        FlowUtil.zero(self._pressure_fbo)
        FlowUtil.zero(self._divergence_fbo)
        # Don't reset obstacles (preserve border)

    # ========== Input Methods ==========

    def set_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Set velocity field."""
        FlowUtil.set(self._input_fbo, texture, strength)

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to velocity field."""
        FlowUtil.add(self._input_fbo, texture, strength)

    def set_density(self, texture: Texture) -> None:
        """Set density field."""
        FlowUtil.blit(self._output_fbo, texture)

    def add_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to density field."""
        FlowUtil.add(self._output_fbo, texture, strength)

    def set_temperature(self, texture: Texture) -> None:
        """Set temperature field."""
        FlowUtil.blit(self._temperature_fbo, texture)

    def add_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to temperature field."""
        FlowUtil.add(self._temperature_fbo, texture, strength)

    def set_pressure(self, texture: Texture) -> None:
        """Set pressure field."""
        FlowUtil.blit(self._pressure_fbo, texture)

    def add_pressure(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to pressure field."""
        FlowUtil.add(self._pressure_fbo, texture, strength)

    # ========== Update Pipeline ==========

    def update(self, delta_time: float = 1.0) -> None:
        """Update fluid simulation (8-step pipeline).

        Args:
            delta_time: Time step (typically 1.0 for per-frame)
        """
        if not self._allocated:
            return

        # Compute scale factor for dual-resolution support
        # When density resolution != simulation resolution, scale velocity vectors accordingly
        density_scale = self._density_width / self._simulation_width if self._simulation_width > 0 else 1.0

        # ===== STEP 1: DENSITY ADVECT & DISSIPATE =====
        advect_den_step = delta_time * self.config.den_speed * self._grid_scale
        dissipate_den = 1.0 - delta_time * self.config.den_dissipation

        self._output_fbo.swap()
        self._output_fbo.begin()
        self._advect_shader.reload()
        self._advect_shader.use(
            self._output_fbo.back_texture,  # Source density
            self._input_fbo.texture,        # Velocity
            self._obstacle_fbo.texture,     # Obstacles
            self._grid_scale,
            advect_den_step,
            dissipate_den
        )
        self._output_fbo.end()

        # ===== STEP 2: VELOCITY ADVECT & DISSIPATE =====
        advect_vel_step = delta_time * self.config.vel_speed * self._grid_scale
        dissipate_vel = 1.0 - delta_time * self.config.vel_dissipation

        self._input_fbo.swap()
        self._input_fbo.begin()
        self._advect_shader.use(
            self._input_fbo.back_texture,   # Source velocity (self-advection)
            self._input_fbo.back_texture,   # Velocity
            self._obstacle_fbo.texture,     # Obstacles
            self._grid_scale,
            advect_vel_step,
            dissipate_vel
        )
        self._input_fbo.end()

        # ===== STEP 3: VELOCITY DIFFUSE (viscosity) =====
        if self.config.vel_viscosity > 0.0:
            viscosity_step = 0.25 * self.config.vel_viscosity
            for _ in range(self.config.vel_viscosity_iter):
                self._input_fbo.swap()
                self._input_fbo.begin()
                self._jacobi_diffusion_shader.use(
                    self._input_fbo.back_texture,
                    self._obstacle_fbo.texture,
                    self._obstacle_offset_fbo.texture,
                    self._grid_scale,
                    viscosity_step
                )
                self._input_fbo.end()

        # ===== STEP 4: VELOCITY VORTICITY CONFINEMENT =====
        if self.config.vel_vorticity > 0.0:
            vorticity_step = self.config.vel_vorticity * self._grid_scale

            # 4a. Compute vorticity curl
            self._vorticity_curl_fbo.begin()
            self._vorticity_curl_shader.use(
                self._input_fbo.texture,
                self._obstacle_fbo.texture,
                self._grid_scale
            )
            self._vorticity_curl_fbo.end()

            # 4b. Compute confinement force
            self._vorticity_force_fbo.begin()
            self._vorticity_force_shader.use(
                self._vorticity_curl_fbo.texture,
                self._grid_scale,
                vorticity_step
            )
            self._vorticity_force_fbo.end()

            # 4c. Add force to velocity
            self.add_velocity(self._vorticity_force_fbo.texture)

        # ===== STEP 5: TEMPERATURE ADVECT & DISSIPATE =====
        advect_tmp_step = delta_time * self.config.tmp_speed * self._grid_scale
        dissipate_tmp = 1.0 - delta_time * self.config.tmp_dissipation

        self._temperature_fbo.swap()
        self._temperature_fbo.begin()
        self._advect_shader.use(
            self._temperature_fbo.back_texture,  # Source temperature
            self._input_fbo.texture,            # Velocity
            self._obstacle_fbo.texture,         # Obstacles
            self._grid_scale,
            advect_tmp_step,
            dissipate_tmp
        )
        self._temperature_fbo.end()

        # ===== STEP 6: TEMPERATURE BUOYANCY =====
        if self.config.tmp_buoyancy > 0.0 and self.config.tmp_weight > 0.0:
            self._buoyancy_fbo.begin()
            self._buoyancy_shader.use(
                self._input_fbo.texture,
                self._temperature_fbo.texture,
                self._output_fbo.texture,
                self.config.tmp_buoyancy,
                self.config.tmp_weight,
                self.config.tmp_ambient
            )
            self._buoyancy_fbo.end()

            # Add buoyancy force to velocity
            self.add_velocity(self._buoyancy_fbo.texture)

        # ===== STEP 7: PRESSURE ADVECT & DISSIPATE =====
        advect_prs_step = delta_time * self.config.prs_speed * self._grid_scale
        dissipate_prs = 1.0 - self.config.prs_dissipation * self.config.prs_dissipation

        self._pressure_fbo.swap()
        self._pressure_fbo.begin()
        self._advect_shader.use(
            self._pressure_fbo.back_texture,  # Source pressure
            self._input_fbo.texture,          # Velocity
            self._obstacle_fbo.texture,       # Obstacles
            self._grid_scale,
            advect_prs_step,
            dissipate_prs
        )
        self._pressure_fbo.end()

        # ===== STEP 8: PRESSURE PROJECTION (Make divergence-free) =====
        # 8a. Compute divergence
        self._divergence_fbo.begin()
        self._divergence_shader.use(
            self._input_fbo.texture,
            self._obstacle_fbo.texture,
            self._obstacle_offset_fbo.texture,
            self._grid_scale
        )
        self._divergence_fbo.end()

        # 8b. Solve Poisson equation for pressure (Jacobi iterations)
        for _ in range(self.config.prs_iterations):
            self._pressure_fbo.swap()
            self._pressure_fbo.begin()
            self._jacobi_pressure_shader.use(
                self._pressure_fbo.back_texture,
                self._divergence_fbo.texture,
                self._obstacle_fbo.texture,
                self._obstacle_offset_fbo.texture,
                self._grid_scale
            )
            self._pressure_fbo.end()

        # 8c. Subtract pressure gradient from velocity
        self._input_fbo.swap()
        self._input_fbo.begin()
        self._gradient_shader.use(
            self._input_fbo.back_texture,
            self._pressure_fbo.texture,
            self._obstacle_fbo.texture,
            self._obstacle_offset_fbo.texture,
            self._grid_scale
        )
        self._input_fbo.end()

    # ========== Obstacle Initialization ==========

    def _init_obstacle(self) -> None:
        """Initialize obstacles with 1-pixel border."""
        # Fill with white (1.0 = obstacle)
        FlowUtil.one(self._obstacle_fbo)

        # Draw black rectangle (0.0 = fluid) with 1-pixel border
        border = 1
        width = self._obstacle_fbo.width
        height = self._obstacle_fbo.height

        self._obstacle_fbo.begin()
        glColor4f(0.0, 0.0, 0.0, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(border, border)
        glVertex2f(width - border, border)
        glVertex2f(width - border, height - border)
        glVertex2f(border, height - border)
        glEnd()
        self._obstacle_fbo.end()

        # Compute obstacle offset (neighbor flags)
        self._obstacle_offset_fbo.begin()
        self._obstacle_offset_shader.use(self._obstacle_fbo.texture)
        self._obstacle_offset_fbo.end()

    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle mask with texture.

        Args:
            texture: Obstacle mask (any channel > 0 = obstacle)
        """
        # Reset to border
        self._init_obstacle()

        # Add user obstacles
        self._obstacle_fbo.swap()
        self._obstacle_fbo.begin()
        self._add_boolean_shader.use(
            self._obstacle_fbo.back_texture,
            texture
        )
        self._obstacle_fbo.end()

        # Update offset
        self._obstacle_offset_fbo.begin()
        self._obstacle_offset_shader.use(self._obstacle_fbo.texture)
        self._obstacle_offset_fbo.end()

    def add_obstacle(self, texture: Texture) -> None:
        """Add to obstacle mask (boolean OR).

        Args:
            texture: Obstacle mask to add
        """
        self._obstacle_fbo.swap()
        self._obstacle_fbo.begin()
        self._add_boolean_shader.use(
            self._obstacle_fbo.back_texture,
            texture
        )
        self._obstacle_fbo.end()

        # Update offset
        self._obstacle_offset_fbo.begin()
        self._obstacle_offset_shader.use(self._obstacle_fbo.texture)
        self._obstacle_offset_fbo.end()
