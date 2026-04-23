"""FluidFlow - 2D Navier-Stokes fluid simulation.

All-fragment pipeline operating on GL_TEXTURE_2D FBOs with ping-pong
buffers for iterative solvers.

Pipeline:
    1. Dampen velocity (magnitude clamping)
    2. Advect velocity (self-advection)
    3. Apply viscosity (Jacobi diffusion solver)
    4. Confine vorticity (curl + confinement force)
    5. Advect temperature
    6. Apply buoyancy force
    7. Advect pressure (optional, non-physical)
    8. Enforce incompressibility (divergence -> Jacobi solve -> gradient subtraction)
    9. Advect density (semi-Lagrangian)
   10. Dampen density (magnitude clamping)

Boundary conditions via per-field wrap modes:
    Velocity/density:  GL_CLAMP_TO_BORDER(0,0,0,0)  -> no-slip / Dirichlet
    Pressure/temp:     GL_CLAMP_TO_EDGE              -> zero-gradient / Neumann
    Obstacle:          GL_CLAMP_TO_BORDER(1,1,1,1)   -> OOB = obstacle
"""
from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, SwapFbo, Fbo
from .. import FlowUtil
from .fluid_config import FluidFlowSettings, VelocitySettings, DensitySettings
from .shaders import (
    Advect, Divergence, Gradient,
    JacobiPressure, JacobiPressureCompute, JacobiDiffusion, JacobiDiffusionCompute,
    VorticityCurl, VorticityForce, Buoyancy, AddBoolean
)

from modules.utils import HotReloadMethods


# ---------------------------------------------------------------------------
# 2D Fluid Simulation
# ---------------------------------------------------------------------------

class FluidFlow:
    """2D Navier-Stokes fluid simulation.

    Does NOT inherit FlowBase -- manages its own SwapFbo ping-pong
    buffers directly.  Provides a compatible public API with the 3D
    FluidFlow3D where applicable.

    Fields:
        velocity:             SwapFbo RG16F    (u, v)
        density:              SwapFbo RGBA16F   (r, g, b, a)
        temperature:          SwapFbo R16F
        pressure:             SwapFbo R16F
        simulation_obstacle:  SwapFbo R8        (sim resolution)
        density_obstacle:     SwapFbo R8        (density resolution)

    Intermediate (single-buffer):
        divergence:       Fbo R16F
        vorticity_curl:   Fbo R16F
        vorticity_force:  Fbo RG16F
        buoyancy:         Fbo RG16F
    """

    def __init__(self, config: FluidFlowSettings | None = None) -> None:
        self.config: FluidFlowSettings = config or FluidFlowSettings()

        # ---- Simulation dimensions and state ---
        self._simulation_width: int = 0
        self._simulation_height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0
        self._aspect: float = 1.0

        self._reference_dt: float = 1.0 / 60.0  # iteration scaling baseline (60fps)
        self._dt: float = self._reference_dt

        self._has_obstacles: bool = False

        self._allocated: bool = False

        self._reset_pending: bool = False
        self._reallocate_pending: bool = False

        # ---- Simulation fields (SwapFbo for ping-pong) ----
        # Velocity: CLAMP_TO_BORDER(0) = no-slip walls
        self._velocity_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Density: CLAMP_TO_BORDER(0) = nothing leaks out
        self._density_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Temperature: CLAMP_TO_EDGE = insulated walls (Neumann)
        self._temperature_fbo: SwapFbo = SwapFbo()
        # Pressure: CLAMP_TO_EDGE = zero-gradient walls (Neumann)
        self._pressure_fbo: SwapFbo = SwapFbo()
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._simulation_obstacle_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 1.0, 1.0, 1.0))
        # Full-resolution obstacle for density advection (same wrap)
        self._density_obstacle_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 1.0, 1.0, 1.0))

        # Intermediate result FBOs (single buffer, no ping-pong)
        self._divergence_fbo: Fbo = Fbo()
        # Curl is sampled at neighbor offsets by VorticityForce — border must return zero
        self._vorticity_curl_fbo: Fbo = Fbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._vorticity_force_fbo: Fbo = Fbo()
        self._buoyancy_fbo: Fbo = Fbo()

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
        self._add_boolean_shader: AddBoolean = AddBoolean()

        # Bind settings actions
        self.config.bind(FluidFlowSettings.reset_sim, lambda _: self._request_reset())
        self.config.bind(FluidFlowSettings.width, lambda _: self._request_reallocate())
        self.config.bind(FluidFlowSettings.height, lambda _: self._request_reallocate())

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Allocation ==========

    def allocate(self, width: int, height: int) -> None:
        """Allocate all FBOs and shaders.

        Args:
            width: Full output resolution width
            height: Full output resolution height
        """
        self._density_width = width
        self._density_height = height
        self._update_simulation_dimensions()

        self._allocate_simulation_fields()
        self._allocate_density_fields()

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
        self._add_boolean_shader.allocate()

        self._allocated = True

        # DEBUG: inject test obstacle shapes
        # from .debug_utils import upload_debug_obstacle
        # upload_debug_obstacle(self, self._simulation_width, self._simulation_height)

    def _update_simulation_dimensions(self) -> None:
        """Recompute simulation dimensions from current config."""
        self._simulation_width = self._align32(self.config.width)
        self._simulation_height = self._align32(self.config.height)
        self._aspect = self._simulation_width / self._simulation_height if self._simulation_height > 0 else 1.0
        self._grid_scale = self._simulation_width / self._density_width if self._density_width > 0 else 1.0

    def _allocate_density_fields(self) -> None:
        """(Re)allocate density-resolution FBOs (full output resolution)."""
        self._density_fbo.allocate(self._density_width, self._density_height, GL_RGBA16F)
        self._density_obstacle_fbo.allocate(self._density_width, self._density_height, GL_R8)

    def _allocate_simulation_fields(self) -> None:
        """(Re)allocate simulation-resolution FBOs."""
        sim_w = self._simulation_width
        sim_h = self._simulation_height

        # Primary simulation fields
        self._velocity_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._temperature_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._pressure_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._simulation_obstacle_fbo.allocate(sim_w, sim_h, GL_R8)
        FlowUtil.blit(self._simulation_obstacle_fbo, self._density_obstacle_fbo.texture)

        # Intermediate FBOs
        self._divergence_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._vorticity_curl_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._vorticity_force_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._buoyancy_fbo.allocate(sim_w, sim_h, GL_RG16F)

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._velocity_fbo.deallocate()
        self._density_fbo.deallocate()
        self._temperature_fbo.deallocate()
        self._pressure_fbo.deallocate()
        self._simulation_obstacle_fbo.deallocate()
        self._density_obstacle_fbo.deallocate()
        self._divergence_fbo.deallocate()
        self._vorticity_curl_fbo.deallocate()
        self._vorticity_force_fbo.deallocate()
        self._buoyancy_fbo.deallocate()

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
        self._add_boolean_shader.deallocate()

        self._allocated = False

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders."""
        self._advect_shader.reload()
        self._divergence_shader.reload()
        self._gradient_shader.reload()
        self._jacobi_pressure_shader.reload()
        self._jacobi_pressure_compute.reload()
        self._jacobi_diffusion_shader.reload()
        self._jacobi_diffusion_compute.reload()
        self._vorticity_curl_shader.reload()
        self._vorticity_force_shader.reload()
        self._buoyancy_shader.reload()
        self._add_boolean_shader.reload()

    # ========== Update Pipeline ==========

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        if not self._allocated:
            return
        self._velocity_fbo.clear_all()
        self._density_fbo.clear_all()
        self._temperature_fbo.clear_all()
        self._pressure_fbo.clear_all()

    def update(self) -> None:
        """Run one frame of the 2D fluid simulation pipeline."""
        if not self._allocated:
            return

        self._reload_shaders()
        self._handle_deferred_actions()

        # Per-frame state
        self._dt = 1.0 / max(1, self.config.fps)

        # Dampen velocity (clean input for all steps)
        vel: VelocitySettings = self.config.velocity
        self._dampen(self._velocity_fbo, vel.dampen_threshold, vel.dampen_time, self._dt, include_alpha=False)

        # Simulation steps
        self._advect_velocity()
        self._apply_viscosity()
        self._confine_vorticity()
        self._advect_temperature()
        self._apply_buoyancy()
        self._advect_pressure()
        self._enforce_incompressibility()
        self._advect_density()

        # Dampen density (clean output)
        den: DensitySettings = self.config.density
        self._dampen(self._density_fbo, den.dampen_threshold, den.dampen_time, self._dt, include_alpha=True)


    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture, strength: float = 1.0) -> None:
        """Add force to velocity in-place (no input_strength or dt scaling)."""
        FlowUtil.add(self._velocity_fbo, force, strength)

    def _advect_velocity(self) -> None:
        """Self-advect & dissipate velocity field."""
        advect_step = self._dt * self.config.velocity.self_advection
        dissipation = self._calculate_dissipation(self._dt, self.config.velocity.fade_time)

        self._velocity_fbo.swap()
        self._velocity_fbo.begin()
        self._advect_shader.use(
            self._velocity_fbo.back_texture,   # Source velocity (self-advection)
            self._velocity_fbo.back_texture,   # Velocity
            self._simulation_obstacle_fbo.texture,     # Obstacles
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._velocity_fbo.end()

    def _apply_viscosity(self) -> None:
        """Diffuse velocity via Jacobi viscosity solver."""
        if self.config.velocity.viscosity <= 0.0:
            return

        viscosity_dt = self.config.velocity.viscosity * (self._grid_scale ** 2) * self._dt
        iterations = self._scale_iterations(self.config.velocity.viscosity_iter)

        if self._use_compute_diffusion:
            result = self._jacobi_diffusion_compute.solve(
                self._velocity_fbo.texture,
                self._velocity_fbo.back_texture,
                self._simulation_obstacle_fbo.texture,
                self._grid_scale,
                self._aspect,
                viscosity_dt,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles
            )
            if result != self._velocity_fbo.texture:
                self._velocity_fbo.swap()
        else:
            for _ in range(iterations):
                self._velocity_fbo.swap()
                self._velocity_fbo.begin()
                self._jacobi_diffusion_shader.use(
                    self._velocity_fbo.back_texture,
                    self._simulation_obstacle_fbo.texture,
                    self._grid_scale,
                    self._aspect,
                    viscosity_dt,
                    has_obstacles=self._has_obstacles
                )
                self._velocity_fbo.end()

    def _confine_vorticity(self) -> None:
        """Vorticity confinement (curl → force → add to velocity)."""
        if self.config.velocity.vorticity <= 0.0 or self.config.velocity.vorticity_radius <= 0.0:
            return

        vorticity_radius = self.config.velocity.vorticity_radius * self._grid_scale
        vorticity_force = self.config.velocity.vorticity * self._dt

        # a. Compute curl
        self._vorticity_curl_fbo.begin()
        self._vorticity_curl_shader.use(
            self._velocity_fbo.texture,
            self._simulation_obstacle_fbo.texture,
            self._grid_scale,
            self._aspect,
            vorticity_radius,
            has_obstacles=self._has_obstacles
        )
        self._vorticity_curl_fbo.end()

        # b. Compute confinement force
        self._vorticity_force_fbo.begin()
        self._vorticity_force_shader.use(
            self._vorticity_curl_fbo.texture,
            self._simulation_obstacle_fbo.texture,
            self._grid_scale,
            self._aspect,
            vorticity_force,
            has_obstacles=self._has_obstacles
        )
        self._vorticity_force_fbo.end()

        # c. Add force to velocity
        self._add_force_to_velocity(self._vorticity_force_fbo.texture)

    def _advect_temperature(self) -> None:
        """Advect & dissipate the temperature scalar field."""
        if self.config.temperature.buoyancy == 0.0:
            self._temperature_fbo.clear_all()
            return

        advect_step = self._dt * self.config.speed
        dissipation = self._calculate_dissipation(self._dt, self.config.temperature.fade_time)

        self._temperature_fbo.swap()
        self._temperature_fbo.begin()
        self._advect_shader.use(
            self._temperature_fbo.back_texture,
            self._velocity_fbo.texture,
            self._simulation_obstacle_fbo.texture,
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._temperature_fbo.end()

    def _apply_buoyancy(self) -> None:
        """Apply buoyancy force to velocity: F = σ(T − T_ambient) − κρ."""
        if self.config.temperature.buoyancy == 0.0:
            return

        sigma = self._dt * self._grid_scale * self.config.temperature.buoyancy
        kappa = self._dt * self._grid_scale * self.config.temperature.weight

        self._buoyancy_fbo.begin()
        self._buoyancy_shader.use(
            self._velocity_fbo.texture,
            self._temperature_fbo.texture,
            self._density_fbo.texture,
            self._simulation_obstacle_fbo.texture,
            sigma,
            kappa,
            self.config.temperature.ambient,
            has_obstacles=self._has_obstacles
        )
        self._buoyancy_fbo.end()

        self._add_force_to_velocity(self._buoyancy_fbo.texture)

    def _advect_pressure(self) -> None:
        """Advect & dissipate pressure (optional, non-physical)."""
        if self.config.pressure.speed <= 0.0:
            return

        advect_step = self._dt * self.config.pressure.speed
        dissipation = self._calculate_dissipation(self._dt, self.config.pressure.fade_time)

        self._pressure_fbo.swap()
        self._pressure_fbo.begin()
        self._advect_shader.use(
            self._pressure_fbo.back_texture,
            self._velocity_fbo.texture,
            self._simulation_obstacle_fbo.texture,
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._pressure_fbo.end()

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence → Jacobi solve → gradient subtraction)."""
        # a. Compute divergence
        self._divergence_fbo.begin()
        self._divergence_shader.use(
            self._velocity_fbo.texture,
            self._simulation_obstacle_fbo.texture,
            self._grid_scale,
            self._aspect,
            has_obstacles=self._has_obstacles
        )
        self._divergence_fbo.end()

        # b. Solve Poisson equation for pressure
        iterations = self._scale_iterations(self.config.pressure.iterations)

        if self._use_compute_pressure:
            result = self._jacobi_pressure_compute.solve(
                self._pressure_fbo.texture,
                self._pressure_fbo.back_texture,
                self._divergence_fbo.texture,
                self._simulation_obstacle_fbo.texture,
                self._grid_scale,
                self._aspect,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles
            )
            if result != self._pressure_fbo.texture:
                self._pressure_fbo.swap()
        else:
            for _ in range(iterations):
                self._pressure_fbo.swap()
                self._pressure_fbo.begin()
                self._jacobi_pressure_shader.use(
                    self._pressure_fbo.back_texture,
                    self._divergence_fbo.texture,
                    self._simulation_obstacle_fbo.texture,
                    self._grid_scale,
                    self._aspect,
                    has_obstacles=self._has_obstacles
                )
                self._pressure_fbo.end()

        # c. Subtract pressure gradient from velocity
        self._velocity_fbo.swap()
        self._velocity_fbo.begin()
        self._gradient_shader.use(
            self._velocity_fbo.back_texture,
            self._pressure_fbo.texture,
            self._simulation_obstacle_fbo.texture,
            self._grid_scale,
            self._aspect,
            has_obstacles=self._has_obstacles
        )
        self._velocity_fbo.end()

    def _advect_density(self) -> None:
        """Advect & dissipate density field."""
        advect_step = self._dt * (self.config.speed + self.config.density.speed_offset)
        dissipation = self._calculate_dissipation(self._dt, self.config.density.fade_time)

        self._density_fbo.swap()
        self._density_fbo.begin()
        self._advect_shader.use(
            self._density_fbo.back_texture,    # Source density
            self._velocity_fbo.texture,        # Velocity
            self._density_obstacle_fbo.texture, # Obstacles (full density resolution)
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._density_fbo.end()

    def _dampen(self, fbo: SwapFbo, threshold: float, dampen_time: float,
               delta_time: float, include_alpha: bool) -> None:
        """Exponential drag on magnitude excess above threshold.

        Args:
            fbo: Field to dampen (ping-pong swap handled by FlowUtil)
            threshold: Magnitude below which values are untouched
            dampen_time: Seconds for excess to decay to ~1%. 0=off
            delta_time: Frame delta time for frame-rate independence
            include_alpha: True for density (RGBA magnitude), False for velocity (RGB)
        """
        if dampen_time <= 0.0:
            return
        factor = pow(0.01, delta_time / dampen_time)
        FlowUtil.dampen(fbo, threshold, factor, include_alpha)

    # ========== Deferred ==========

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            old_w, old_h = self._simulation_width, self._simulation_height
            self._update_simulation_dimensions()
            if self._simulation_width != old_w or self._simulation_height != old_h:
                self._allocate_simulation_fields()

        if self._reset_pending:
            self._reset_pending = False
            self.reset()

    def _request_reset(self) -> None:
        """Thread-safe reset request — deferred to next update() on the GL thread."""
        self._reset_pending = True

    def _request_reallocate(self) -> None:
        """Thread-safe reallocation request — deferred to next update() on the GL thread."""
        self._reallocate_pending = True

    # ========== Internal helpers ==========

    def _scale_iterations(self, base_iterations: int) -> int:
        """Scale solver iterations proportionally to dt.

        GUI iteration count represents quality at 60fps. At other frame rates,
        iterations scale to approximate consistent solver convergence.
        """
        ratio = self._dt / self._reference_dt
        return max(1, int(base_iterations * ratio + 0.5))

    @staticmethod
    def _align32(v: int) -> int:
        """Round up to the nearest multiple of 32."""
        return (v + 31) & ~31

    @staticmethod
    def _calculate_dissipation(delta_time: float, decay_time: float) -> float:
        """Calculate frame-rate independent decay multiplier.

        Returns pow(0.01, dt / decay_time) -- field reaches 1% after decay_time seconds.
        """
        return pow(0.01, delta_time / max(0.001, decay_time))

    # ========== Input Methods ==========
    def set_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Set velocity field."""
        if not self._allocated:
            return
        FlowUtil.set(self._velocity_fbo, texture, strength)

    def set_velocity_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set velocity in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._velocity_fbo, texture, x, y, w, h, strength)

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to velocity field. Applies config velocity.input_strength and delta_time."""
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.velocity.input_strength * dt
        FlowUtil.add(self._velocity_fbo, texture, effective)

    def add_velocity_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add velocity to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._velocity_fbo, texture, x, y, w, h, strength)

    # ----- Density -----
    def set_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Set density field."""
        if not self._allocated:
            return
        FlowUtil.set(self._density_fbo, texture, strength)

    def set_density_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set density in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._density_fbo, texture, x, y, w, h, strength)

    def set_density_channel(self, texture: Texture, channel: int) -> None:
        """Set single-channel texture to one of the density channels."""
        if not self._allocated:
            return
        FlowUtil.set_channel(self._density_fbo, texture, channel)

    def set_density_channel_region(self, texture: Texture, channel: int, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set density channel in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_channel_region(self._density_fbo, texture, channel, x, y, w, h, strength)

    def add_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to density field. Applies config density.input_strength and delta_time."""
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.density.input_strength * dt
        FlowUtil.add(self._density_fbo, texture, effective)

    def add_density_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add density to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._density_fbo, texture, x, y, w, h, strength)

    def add_density_channel(self, texture: Texture, channel: int, strength: float = 1.0) -> None:
        """Add single-channel texture to one of the density channels. Applies config density.input_strength and delta_time."""
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.density.input_strength * dt
        FlowUtil.add_channel(self._density_fbo, texture, channel, effective)

    def add_density_channel_region(self, texture: Texture, channel: int, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add density to a specific channel at a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_channel_region(self._density_fbo, texture, channel, x, y, w, h, strength)

    # ----- Temperature -----
    def set_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Set temperature field."""
        if not self._allocated:
            return
        FlowUtil.set(self._temperature_fbo, texture, strength)

    def set_temperature_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set temperature in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._temperature_fbo, texture, x, y, w, h, strength)

    def add_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to temperature field. Applies config temperature.input_strength and delta_time."""
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.temperature.input_strength * dt
        FlowUtil.add(self._temperature_fbo, texture, effective)

    def add_temperature_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add temperature to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._temperature_fbo, texture, x, y, w, h, strength)

    # ----- Pressure -----
    def set_pressure(self, texture: Texture, strength: float = 1.0) -> None:
        """Set pressure field."""
        if not self._allocated:
            return
        FlowUtil.set(self._pressure_fbo, texture, strength)

    def set_pressure_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set pressure in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._pressure_fbo, texture, x, y, w, h, strength)

    def add_pressure(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to pressure field."""
        if not self._allocated:
            return
        FlowUtil.add(self._pressure_fbo, texture, strength)

    def add_pressure_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add pressure to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._pressure_fbo, texture, x, y, w, h, strength)

    # ----- Obstacles -----
    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle mask with texture (boolean OR with empty field).

        Updates both sim-resolution and density-resolution obstacle buffers.

        Args:
            texture: Obstacle mask (any channel > 0 = obstacle)
        """
        if not self._allocated:
            return
        FlowUtil.zero(self._simulation_obstacle_fbo)
        self._simulation_obstacle_fbo.swap()
        self._simulation_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._simulation_obstacle_fbo.back_texture, texture)
        self._simulation_obstacle_fbo.end()

        FlowUtil.zero(self._density_obstacle_fbo)
        self._density_obstacle_fbo.swap()
        self._density_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._density_obstacle_fbo.back_texture, texture)
        self._density_obstacle_fbo.end()

        self._has_obstacles = True

    def add_obstacle(self, texture: Texture) -> None:
        """Add to obstacle mask (boolean OR).

        Updates both sim-resolution and density-resolution obstacle buffers.

        Args:
            texture: Obstacle mask to add
        """
        if not self._allocated:
            return
        self._simulation_obstacle_fbo.swap()
        self._simulation_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._simulation_obstacle_fbo.back_texture, texture)
        self._simulation_obstacle_fbo.end()

        self._density_obstacle_fbo.swap()
        self._density_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._density_obstacle_fbo.back_texture, texture)
        self._density_obstacle_fbo.end()

        self._has_obstacles = True

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        if not self._allocated:
            return
        FlowUtil.zero(self._simulation_obstacle_fbo)
        FlowUtil.zero(self._density_obstacle_fbo)
        self._has_obstacles = False

    # ========== Properties ==========

    @property
    def allocated(self) -> bool:
        return self._allocated

    @property
    def velocity(self) -> Texture:
        """RG16F velocity field."""
        return self._velocity_fbo.texture

    @property
    def density(self) -> Texture:
        """RGBA16F density/color field."""
        return self._density_fbo.texture

    @property
    def temperature(self) -> Texture:
        """R16F temperature field."""
        return self._temperature_fbo.texture

    @property
    def pressure(self) -> Texture:
        """R16F pressure field."""
        return self._pressure_fbo.texture

    @property
    def divergence(self) -> Texture:
        """R16F divergence field (intermediate)."""
        return self._divergence_fbo.texture

    @property
    def vorticity_curl(self) -> Texture:
        """R16F vorticity curl field."""
        return self._vorticity_curl_fbo.texture

    @property
    def buoyancy(self) -> Texture:
        """RG16F buoyancy force field."""
        return self._buoyancy_fbo.texture

    @property
    def obstacle(self) -> Texture:
        """R8 obstacle mask."""
        return self._density_obstacle_fbo.texture

    @property
    def sim_width(self) -> int:
        """Current simulation resolution width (aligned to 16)."""
        return self._simulation_width

    @property
    def sim_height(self) -> int:
        """Current simulation resolution height (aligned to 16)."""
        return self._simulation_height
