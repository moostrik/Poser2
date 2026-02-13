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
from .. import FlowBase, FlowUtil, ConfigBase
from .shaders import (
    Advect, Divergence, Gradient, JacobiPressure, JacobiPressureCompute,
    JacobiDiffusion, JacobiDiffusionCompute,
    VorticityCurl, VorticityForce, Buoyancy, ObstacleOffset, ObstacleBorder, AddBoolean
)

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class FluidFlowConfig(ConfigBase):
    """Configuration for fluid simulation."""

    # Velocity parameters
    vel_speed: float = field(
        default=0.3,
        metadata={"min": 0.0, "max": 10.0, "label": "Velocity Speed",
                  "description": "Velocity advection speed multiplier"}
    )
    vel_decay: float = field(
        default=3.0,
        metadata={"min": 0.01, "max": 60.0, "label": "Velocity Decay Time",
                  "description": "Time in seconds for velocity to decay to 1%"}
    )
    vel_vorticity: float = field(
        default=0.0,
        metadata={"min": 0.0, "max": 60.0, "label": "Vorticity",
                  "description": "Vortex confinement strength (adds turbulence)"}
    )
    vel_vorticity_radius: float = field(
        default=1.0,
        metadata={"min": 1.0, "max":30.0, "label": "Vorticity Radius",
                  "description": "Curl sampling radius in texels (larger = bigger swirls)"}
    )
    vel_viscosity: float = field(
        default=0.0,
        metadata={"min": 0, "max": 4, "label": "Viscosity",
                  "description": "Fluid thickness/resistance to flow"}
    )
    vel_viscosity_iter: int = field(
        default=20,
        metadata={"min": 1, "max": 60, "label": "Viscosity Iterations",
                  "description": "Solver iterations for viscosity (higher = more accurate)"}
    )

    # Pressure parameters
    prs_speed: float = field(
        default=0.33,
        metadata={"min": 0.0, "max": 10.0, "label": "Pressure Speed",
                  "description": "Pressure advection speed (usually 0 for physical accuracy)"}
    )
    prs_decay: float = field(
        default=0.3,  # Fast decay for pressure
        metadata={"min": 0.01, "max": 60.0, "label": "Pressure Decay Time",
                  "description": "Time in seconds for pressure to decay to 1%"}
    )
    prs_iterations: int = field(
        default=40,
        metadata={"min": 1, "max": 60, "label": "Pressure Iterations",
                  "description": "Solver iterations for pressure (higher = more incompressible)"}
    )

    # Density parameters
    den_speed: float = field(
        default=0.3,
        metadata={"min": 0.0, "max": 10.0, "label": "Density Speed",
                  "description": "Density advection speed"}
    )
    den_decay: float = field(
        default=3.0,
        metadata={"min": 0.01, "max": 60.0, "label": "Density Decay Time",
                  "description": "Time in seconds for density to decay to 1%"}
    )

    # Temperature parameters
    tmp_speed: float = field(
        default=0.3,
        metadata={"min": 0.0, "max": 10.0, "label": "Temperature Speed",
                  "description": "Temperature advection speed"}
    )
    tmp_decay: float = field(
        default=3.0,
        metadata={"min": 0.01, "max": 60.0, "label": "Temperature Decay Time",
                  "description": "Time in seconds for temperature to decay to 1%"}
    )
    tmp_buoyancy: float = field(
        default=0.0,
        metadata={"min": 0.0, "max": 10.0, "label": "Buoyancy",
                  "description": "Thermal buoyancy coefficient (σ): hot air rises"}
    )
    tmp_weight: float = field(
        default=0.25,
        metadata={"min": 0.0, "max": 2.0, "label": "Density Weight Ratio",
                  "description": "Ratio of gravity/settling effect vs thermal lift (0.25 = 25% of buoyancy)"}
    )
    tmp_ambient: float = field(
        default=0.2,
        metadata={"min": 0.0, "max": 1.0, "label": "Ambient Temperature",
                  "description": "Reference temperature (buoyancy = 0 at this temp)"}
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

    def __init__(self, sim_scale: float = 0.25,config: FluidFlowConfig | None = None) -> None:
        super().__init__()

        self.config: FluidFlowConfig = config or FluidFlowConfig()

        # Define formats for FlowBase
        self._input_internal_format = GL_RG16F     # Velocity (inherited as _input_fbo)
        self._output_internal_format = GL_RGBA16F  # Density (inherited as _output_fbo)

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
        self._width: int = 0
        self._height: int = 0
        self._simulation_scale: float = sim_scale
        self._simulation_width: int = 0
        self._simulation_height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0

        # Shaders
        self._advect_shader: Advect = Advect()
        self._divergence_shader: Divergence = Divergence()
        self._gradient_shader: Gradient = Gradient()
        self._jacobi_pressure_shader: JacobiPressure = JacobiPressure()
        self._jacobi_pressure_compute: JacobiPressureCompute = JacobiPressureCompute()
        self._jacobi_diffusion_shader: JacobiDiffusion = JacobiDiffusion()
        self._jacobi_diffusion_compute: JacobiDiffusionCompute = JacobiDiffusionCompute()

        # Use compute shaders for iterative solvers (faster)
        self._use_compute_pressure: bool = True
        self._use_compute_diffusion: bool = True
        self._vorticity_curl_shader: VorticityCurl = VorticityCurl()
        self._vorticity_force_shader: VorticityForce = VorticityForce()
        self._buoyancy_shader: Buoyancy = Buoyancy()
        self._obstacle_offset_shader: ObstacleOffset = ObstacleOffset()
        self._obstacle_border_shader: ObstacleBorder = ObstacleBorder()
        self._add_boolean_shader: AddBoolean = AddBoolean()

        # hot_reload = HotReloadMethods(self.__class__, True, True)

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
        self._width = width
        self._height = height

        self._simulation_width = int(width * self._simulation_scale)
        self._simulation_height = int(height * self._simulation_scale)

        self._density_width = output_width if output_width is not None else width
        self._density_height = output_height if output_height is not None else height

        # Compute aspect ratio for isotropic simulation
        self._aspect: float = width / height if height > 0 else 1.0

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
        self._jacobi_pressure_compute.allocate()
        self._jacobi_diffusion_shader.allocate()
        self._jacobi_diffusion_compute.allocate()
        self._vorticity_curl_shader.allocate()
        self._vorticity_force_shader.allocate()
        self._buoyancy_shader.allocate()
        self._obstacle_offset_shader.allocate()
        self._obstacle_border_shader.allocate()
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
        self._jacobi_pressure_compute.deallocate()
        self._jacobi_diffusion_shader.deallocate()
        self._jacobi_diffusion_compute.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._buoyancy_shader.deallocate()
        self._obstacle_offset_shader.deallocate()
        self._obstacle_border_shader.deallocate()
        self._add_boolean_shader.deallocate()

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        super().reset()
        FlowUtil.zero(self._temperature_fbo)
        FlowUtil.zero(self._pressure_fbo)
        FlowUtil.zero(self._divergence_fbo)
        # Don't reset obstacles (preserve border)

    # ========== Input Methods ==========

    # ----- Velocity -----
    def set_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Set velocity field."""
        FlowUtil.set(self._input_fbo, texture, strength)

    def set_velocity_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set velocity in a specific region."""
        FlowUtil.set_region(self._input_fbo, texture, x, y, w, h, strength)

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to velocity field."""
        FlowUtil.add(self._input_fbo, texture, strength)

    def add_velocity_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add velocity to a specific region."""
        FlowUtil.add_region(self._input_fbo, texture, x, y, w, h, strength)

    # ----- Density -----
    def set_density(self, texture: Texture) -> None:
        """Set density field."""
        FlowUtil.blit(self._output_fbo, texture)

    def set_density_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set density in a specific region."""
        FlowUtil.set_region(self._output_fbo, texture, x, y, w, h, strength)

    def set_density_channel(self, texture: Texture, channel: int) -> None:
        """Set single-channel texture to one of the density channels."""
        FlowUtil.set_channel(self._output_fbo, texture, channel)

    def set_density_channel_region(self, texture: Texture, channel: int, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set density channel in a specific region."""
        FlowUtil.set_channel_region(self._output_fbo, texture, channel, x, y, w, h, strength)

    def add_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to density field."""
        FlowUtil.add(self._output_fbo, texture, strength)

    def add_density_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add density to a specific region."""
        FlowUtil.add_region(self._output_fbo, texture, x, y, w, h, strength)

    def add_density_channel(self, texture: Texture, channel: int, strength: float = 1.0) -> None:
        """Add single-channel texture to one of the density channels."""
        FlowUtil.add_channel(self._output_fbo, texture, channel, strength)

    def add_density_channel_region(self, texture: Texture, channel: int, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add density to a specific channel at a specific region."""
        FlowUtil.add_channel_region(self._output_fbo, texture, channel, x, y, w, h, strength)

    def clamp_density(self, min_value: float = 0.0, max_value: float = 1.0) -> None:
        """Clamp density values to a specified range."""
        FlowUtil.clamp(self._output_fbo, min_value, max_value)

    # ----- Temperature -----
    def set_temperature(self, texture: Texture) -> None:
        """Set temperature field."""
        FlowUtil.blit(self._temperature_fbo, texture)

    def set_temperature_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set temperature in a specific region."""
        FlowUtil.set_region(self._temperature_fbo, texture, x, y, w, h, strength)

    def add_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to temperature field."""
        FlowUtil.add(self._temperature_fbo, texture, strength)

    def add_temperature_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add temperature to a specific region."""
        FlowUtil.add_region(self._temperature_fbo, texture, x, y, w, h, strength)

    # ----- Pressure -----
    def set_pressure(self, texture: Texture) -> None:
        """Set pressure field."""
        FlowUtil.blit(self._pressure_fbo, texture)

    def set_pressure_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set pressure in a specific region."""
        FlowUtil.set_region(self._pressure_fbo, texture, x, y, w, h, strength)

    def add_pressure(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to pressure field."""
        FlowUtil.add(self._pressure_fbo, texture, strength)

    def add_pressure_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add pressure to a specific region."""
        FlowUtil.add_region(self._pressure_fbo, texture, x, y, w, h, strength)

    # ========== Update Pipeline ==========

    def update(self, delta_time: float = 1.0) -> None:
        """Update fluid simulation (8-step pipeline).

        Args:
            delta_time: Time step (typically 1.0 for per-frame)
        """
        if not self._allocated:
            return

        # self._aspect = 1.0
        self._aspect = self._width / self._height if self._height > 0 else 1.0

        # ===== STEP 1: DENSITY ADVECT & DISSIPATE =====
        advect_den_step: float = delta_time * self._simulation_scale * self.config.den_speed
        dissipate_den: float = FluidFlow._calculate_dissipation(delta_time, self.config.den_decay)

        self._output_fbo.swap()
        self._output_fbo.begin()
        self._advect_shader.use(
            self._output_fbo.back_texture,  # Source density
            self._input_fbo.texture,        # Velocity
            self._obstacle_fbo.texture,     # Obstacles
            self._simulation_scale,
            advect_den_step,
            dissipate_den
        )
        self._output_fbo.end()

        # ===== STEP 2: VELOCITY ADVECT & DISSIPATE =====
        advect_vel_step: float = delta_time * self._simulation_scale * self.config.vel_speed
        dissipate_vel: float = FluidFlow._calculate_dissipation(delta_time, self.config.vel_decay)

        self._input_fbo.swap()
        self._input_fbo.begin()
        self._advect_shader.use(
            self._input_fbo.back_texture,   # Source velocity (self-advection)
            self._input_fbo.back_texture,   # Velocity
            self._obstacle_fbo.texture,     # Obstacles
            self._simulation_scale,
            advect_vel_step,
            dissipate_vel
        )
        self._input_fbo.end()

        # ===== STEP 3: VELOCITY DIFFUSE (viscosity) =====
        if self.config.vel_viscosity > 0.0:
            # Scale viscosity by simulation_scale² for resolution independence
            viscosity_dt: float = self.config.vel_viscosity * (self._simulation_scale ** 2) * delta_time

            if self._use_compute_diffusion:
                # Compute shader: multi-iteration with automatic ping-pong
                result = self._jacobi_diffusion_compute.solve(
                    self._input_fbo.texture,
                    self._input_fbo.back_texture,
                    self._obstacle_fbo.texture,
                    self._obstacle_offset_fbo.texture,
                    self._simulation_scale,
                    self._aspect,
                    viscosity_dt,
                    total_iterations=self.config.vel_viscosity_iter,
                    iterations_per_dispatch=5
                )
                # Ensure the correct buffer is active after solve
                if result != self._input_fbo.texture:
                    self._input_fbo.swap()
            else:
                # Fragment shader fallback: one iteration per FBO swap
                for _ in range(self.config.vel_viscosity_iter):
                    self._input_fbo.swap()
                    self._input_fbo.begin()
                    self._jacobi_diffusion_shader.use(
                        self._input_fbo.back_texture,
                        self._obstacle_fbo.texture,
                        self._obstacle_offset_fbo.texture,
                        self._simulation_scale,
                        self._aspect,
                        viscosity_dt
                    )
                    self._input_fbo.end()

        # ===== STEP 4: VELOCITY VORTICITY CONFINEMENT =====
        if self.config.vel_vorticity > 0.0 and self.config.vel_vorticity_radius > 0.0:
            self._vorticity_curl_shader.reload()
            self._vorticity_force_shader.reload()

            vorticity_radius: float = self.config.vel_vorticity_radius * self._simulation_scale
            vorticity_force: float = (self.config.vel_vorticity * delta_time)

            # 4a. Compute vorticity curl
            self._vorticity_curl_fbo.begin()
            self._vorticity_curl_shader.use(
                self._input_fbo.texture,
                self._obstacle_fbo.texture,
                self._simulation_scale,
                self._aspect,
                vorticity_radius
            )
            self._vorticity_curl_fbo.end()

            # 4b. Compute confinement force
            self._vorticity_force_fbo.begin()
            self._vorticity_force_shader.use(
                self._vorticity_curl_fbo.texture,
                self._simulation_scale,
                self._aspect,
                vorticity_force
            )
            self._vorticity_force_fbo.end()

            # 4c. Add force to velocity
            self.add_velocity(self._vorticity_force_fbo.texture)

        # ===== STEP 5 & 6: TEMPERATURE ADVECT & BUOYANCY =====
        # Only compute temperature if buoyancy is enabled
        if self.config.tmp_buoyancy == 0.0:
            FlowUtil.zero(self._temperature_fbo)
        else:
            # 5a. Advect temperature
            advect_tmp_step: float = delta_time * self._simulation_scale * self.config.tmp_speed
            dissipate_tmp: float = FluidFlow._calculate_dissipation(delta_time, self.config.tmp_decay)

            self._temperature_fbo.swap()
            self._temperature_fbo.begin()
            self._advect_shader.use(
                self._temperature_fbo.back_texture,  # Source temperature
                self._input_fbo.texture,            # Velocity
                self._obstacle_fbo.texture,         # Obstacles
                self._simulation_scale,
                advect_tmp_step,
                dissipate_tmp
            )
            self._temperature_fbo.end()

            # 6. Compute and apply buoyancy force
            # F = σ(T - T_ambient) - κρ  where κ = weight_ratio * σ
            # Both terms scaled by delta_time * simulation_scale for resolution independence
            sigma: float = delta_time * self._simulation_scale * self.config.tmp_buoyancy
            kappa: float = delta_time * self._simulation_scale * self.config.tmp_weight  # Weight as ratio of thermal effect

            self._buoyancy_fbo.begin()
            self._buoyancy_shader.use(
                self._input_fbo.texture,
                self._temperature_fbo.texture,
                self._output_fbo.texture,
                sigma,
                kappa,
                self.config.tmp_ambient
            )
            self._buoyancy_fbo.end()

            # Add buoyancy force to velocity
            self.add_velocity(self._buoyancy_fbo.texture)
            # Reset temperature when buoyancy disabled to prevent stale data

        # ===== STEP 7: PRESSURE ADVECT & DISSIPATE =====
        # # Only advect pressure for artistic effects (non-physical)
        # When prs_speed = 0, pressure is purely from projection (physical)
        if self.config.prs_speed > 0.0:
            advect_prs_step: float = delta_time * self._simulation_scale * self.config.prs_speed
            dissipate_prs: float = FluidFlow._calculate_dissipation(delta_time, self.config.prs_decay)

            self._pressure_fbo.swap()
            self._pressure_fbo.begin()
            self._advect_shader.use(
                self._pressure_fbo.back_texture,  # Source pressure
                self._input_fbo.texture,          # Velocity
                self._obstacle_fbo.texture,       # Obstacles
                self._simulation_scale,
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
            self._simulation_scale,
            self._aspect
        )
        self._divergence_fbo.end()

        self._use_compute_pressure = True

        # 8b. Solve Poisson equation for pressure (Jacobi iterations)
        if self._use_compute_pressure:
            # Compute shader: multi-iteration with automatic ping-pong
            result = self._jacobi_pressure_compute.solve(
                self._pressure_fbo.texture,
                self._pressure_fbo.back_texture,
                self._divergence_fbo.texture,
                self._obstacle_fbo.texture,
                self._obstacle_offset_fbo.texture,
                self._simulation_scale,
                self._aspect,
                total_iterations=self.config.prs_iterations,
                iterations_per_dispatch=5
            )
            # Ensure the correct buffer is active after solve
            if result != self._pressure_fbo.texture:
                self._pressure_fbo.swap()
        else:
            # Fragment shader fallback: one iteration per FBO swap
            for _ in range(self.config.prs_iterations):
                self._pressure_fbo.swap()
                self._pressure_fbo.begin()
                self._jacobi_pressure_shader.use(
                    self._pressure_fbo.back_texture,
                    self._divergence_fbo.texture,
                    self._obstacle_fbo.texture,
                    self._obstacle_offset_fbo.texture,
                    self._simulation_scale,
                    self._aspect
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
            self._simulation_scale,
            self._aspect
        )
        self._input_fbo.end()

    # ========== Obstacle Initialization ==========

    def _init_obstacle(self) -> None:
        """Initialize obstacles with 1-pixel border using shader."""
        border = 1
        width: int = self._obstacle_fbo.width
        height: int = self._obstacle_fbo.height

        self._obstacle_fbo.begin()
        self._obstacle_border_shader.use(width, height, border)
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

    @staticmethod
    def _calculate_dissipation(delta_time: float, decay_time: float) -> float:
        """Calculate frame-rate independent decay multiplier.

        Args:
            delta_time: Frame delta time (1/fps)
            decay_time: Time in seconds to reach 1% of original value

        Returns:
            Multiplier to apply to field this frame (e.g., 0.99 = 1% loss)
        """
        return pow(0.01, delta_time / max(0.001, decay_time))