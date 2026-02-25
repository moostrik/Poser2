"""Fluid Flow - 2D Navier-Stokes fluid simulation.

Implements velocity, density, temperature, and pressure fields with:
- Semi-Lagrangian advection
- Pressure projection (incompressibility)
- Viscosity diffusion
- Vorticity confinement
- Buoyancy forces

Ported from ofxFlowTools ftFluidFlow.h/cpp
"""
from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, SwapFbo, Fbo
from modules.settings import Field, Settings, Widget
from .. import FlowBase, FlowUtil
from .fluid_config import VelocityConfig, DensityConfig, TemperatureConfig, PressureConfig
from .shaders import (
    Advect, Divergence, Gradient,
    JacobiPressure, JacobiPressureCompute, JacobiDiffusion, JacobiDiffusionCompute,
    VorticityCurl, VorticityForce, Buoyancy, AddBoolean
)

from modules.utils.HotReloadMethods import HotReloadMethods


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FluidFlowConfig(Settings):
    """Configuration for fluid simulation.

    Speed model:
        One 'speed' parameter controls all passive scalar transport (density,
        temperature, pressure). At speed=1.0, a velocity value of 1.0 moves
        fluid across the full texture width in 1 second.

        velocity.self_advection is a separate stability parameter that controls
        how much the velocity field advects itself. Keep low to prevent numerical
        blowup from the self-advection feedback loop.

        density.speed_offset adds to base speed for density transport only.
        0 = physically coupled to velocity (density rides the flow).
        Positive = density flings further. Negative = density lags.

    Fade_time model:
        Exponential frame-rate-independent decay: multiplier = 0.01^(dt/fade_time).
        fade_time=3.0 means the field retains ~1% after 3 seconds.
    """

    # ---- Actions ----
    reset_sim = Field(False, widget=Widget.button, description="Reset all simulation fields to zero")

    # ---- Global ----
    simulation_scale = Field(0.5, min=0.1, max=2.0, description="Resolution scale for simulation buffers")
    fps = Field(60, min=1, max=240, access=Field.READ, description="Current average FPS for dt calculation (bound from WindowManager)")
    speed = Field(0.5, min=0.0, max=5.0, description="Base fluid transport rate")

    # ---- Field groups ----
    velocity:    VelocityConfig
    density:     DensityConfig
    temperature: TemperatureConfig
    pressure:    PressureConfig


# ---------------------------------------------------------------------------
# 2D Fluid Simulation
# ---------------------------------------------------------------------------

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

        # ---- Simulation dimensions and state ---
        self._width: int = 0
        self._height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0
        self._dt: float = 1.0 / 60.0
        self._has_obstacles: bool = False
        self._reset_pending: bool = False
        self._reallocate_pending: bool = False

        # Define formats for FlowBase
        self._input_internal_format = GL_RG16F     # Velocity (inherited as _input_fbo)
        self._output_internal_format = GL_RGBA16F  # Density (inherited as _output_fbo)

        # Override FlowBase FBOs with border-color wrap for boundary conditions:
        # Velocity: CLAMP_TO_BORDER(0) = no-slip (velocity->0 at walls)
        # Density:  CLAMP_TO_BORDER(0) = nothing leaks out
        self._input_fbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._output_fbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))

        # Additional simulation fields (SwapFbo for ping-pong)
        self._temperature_fbo: SwapFbo = SwapFbo()  # CLAMP_TO_EDGE = insulated walls (Neumann)
        self._pressure_fbo: SwapFbo = SwapFbo()      # CLAMP_TO_EDGE = zero-gradient walls (Neumann)
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds reads as obstacle
        self._obstacle_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 1.0, 1.0, 1.0))
        # Full-resolution obstacle for density advection (same wrap)
        self._density_obstacle_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 1.0, 1.0, 1.0))

        # Intermediate result FBOs (single buffer, no ping-pong)
        self._divergence_fbo: Fbo = Fbo()
        self._vorticity_curl_fbo: Fbo = Fbo()
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
        self.config.bind(FluidFlowConfig.reset_sim, lambda _: self._request_reset())
        self.config.bind(FluidFlowConfig.simulation_scale, lambda _: self._request_reallocate())

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Allocation ==========

    def allocate(self, width: int, height: int) -> None:
        """Allocate fluid simulation FBOs.

        Args:
            width: Full output resolution width
            height: Full output resolution height
        """
        self._density_width = width
        self._density_height = height

        sim_scale = self.config.simulation_scale
        self._width = self._align16(int(width * sim_scale))
        self._height = self._align16(int(height * sim_scale))

        # Compute aspect ratio for isotropic simulation
        self._aspect: float = self._width / self._height if self._height > 0 else 1.0

        self._allocate_fbos()

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

        # DEBUG: inject test obstacle shapes
        self._debug_inject_obstacle_shapes()

    def _debug_inject_obstacle_shapes(self) -> None:
        """Draw circle, triangle, cross, and line into the obstacle buffer for visual verification."""
        import numpy as np

        w, h = self._width, self._height
        mask = np.zeros((h, w), dtype=np.uint8)

        # Coordinate grids (row=y, col=x)
        yy, xx = np.ogrid[0:h, 0:w]

        # ---- Circle (top-left quadrant) ----
        cx, cy, r = w // 4, h * 3 // 4, min(w, h) // 8
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask[dist_sq <= r * r] = 255

        # ---- Triangle (top-right quadrant) ----
        tx, ty = w * 3 // 4, h * 3 // 4  # center
        s = min(w, h) // 7  # half-size
        # Equilateral triangle pointing up: 3 edge tests
        # Vertices: top (tx, ty+s), bottom-left (tx-s, ty-s*0.6), bottom-right (tx+s, ty-s*0.6)
        top_y = ty + s
        bot_y = ty - int(s * 0.6)
        # Left edge: top -> bottom-left
        # Right edge: top -> bottom-right
        # Bottom edge: bottom-left -> bottom-right
        for y_px in range(max(0, bot_y), min(h, top_y + 1)):
            for x_px in range(max(0, tx - s), min(w, tx + s + 1)):
                # Barycentric test
                t_frac = (y_px - bot_y) / max(1, top_y - bot_y)
                half_w = s * (1.0 - t_frac)
                if abs(x_px - tx) <= half_w:
                    mask[y_px, x_px] = 255

        # ---- Cross (bottom-left quadrant) ----
        cx2, cy2 = w // 4, h // 4
        arm = min(w, h) // 8
        thick = max(2, min(w, h) // 40)
        # Horizontal bar
        mask[max(0, cy2 - thick):min(h, cy2 + thick + 1),
             max(0, cx2 - arm):min(w, cx2 + arm + 1)] = 255
        # Vertical bar
        mask[max(0, cy2 - arm):min(h, cy2 + arm + 1),
             max(0, cx2 - thick):min(w, cx2 + thick + 1)] = 255

        # ---- Diagonal line (bottom-right quadrant) ----
        lx, ly = w * 3 // 4, h // 4  # center
        length = min(w, h) // 6
        thick_l = max(2, min(w, h) // 50)
        for t in np.linspace(-1, 1, length * 4):
            px = int(lx + t * length * 0.5)
            py = int(ly + t * length * 0.3)
            for dx in range(-thick_l, thick_l + 1):
                for dy in range(-thick_l, thick_l + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        mask[ny, nx] = 255

        # Upload to a temporary texture and inject
        temp = Texture()
        temp.allocate(w, h, GL_R8)
        glBindTexture(GL_TEXTURE_2D, temp.tex_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, mask)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.set_obstacle(temp)
        temp.deallocate()

    def _allocate_fbos(self) -> None:
        """(Re)allocate all simulation FBOs at current dimensions.

        Uses self._width, self._height (and self._density_width/
        self._density_height for the density/output FBOs via FlowBase).
        Shaders are not touched because they carry no resolution-dependent state.
        """
        # Base FBOs (velocity RG32F, density RGBA32F)
        super().allocate(self._width, self._height, self._density_width, self._density_height)

        # Simulation fields
        self._temperature_fbo.allocate(self._width, self._height, GL_R16F)
        FlowUtil.zero(self._temperature_fbo)

        self._pressure_fbo.allocate(self._width, self._height, GL_R16F)
        FlowUtil.zero(self._pressure_fbo)

        self._obstacle_fbo.allocate(self._width, self._height, GL_R8)
        FlowUtil.zero(self._obstacle_fbo)

        self._density_obstacle_fbo.allocate(self._density_width, self._density_height, GL_R8)
        FlowUtil.zero(self._density_obstacle_fbo)

        self._has_obstacles = False

        # Intermediate FBOs
        self._divergence_fbo.allocate(self._width, self._height, GL_R16F)
        FlowUtil.zero(self._divergence_fbo)

        self._vorticity_curl_fbo.allocate(self._width, self._height, GL_R16F)
        FlowUtil.zero(self._vorticity_curl_fbo)

        self._vorticity_force_fbo.allocate(self._width, self._height, GL_RG16F)
        FlowUtil.zero(self._vorticity_force_fbo)

        self._buoyancy_fbo.allocate(self._width, self._height, GL_RG16F)
        FlowUtil.zero(self._buoyancy_fbo)

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self._temperature_fbo.deallocate()
        self._pressure_fbo.deallocate()
        self._obstacle_fbo.deallocate()
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

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders (checks file timestamps internally)."""
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
        super().reset()
        FlowUtil.zero(self._temperature_fbo)
        FlowUtil.zero(self._pressure_fbo)
        FlowUtil.zero(self._divergence_fbo)
        # Don't reset obstacles (preserve border)

    def update(self) -> None:
        """Run one frame of the 2D fluid simulation pipeline."""
        if not self._allocated:
            return

        self._handle_deferred_actions()
        self._reload_shaders()

        # Per-frame state
        self._dt = 1.0 / max(1, self.config.fps)
        self._aspect = self._width / self._height if self._height > 0 else 1.0

        # Dampen inputs before simulation
        vel = self.config.velocity
        den = self.config.density
        self._dampen(self._input_fbo,  vel.dampen_threshold, vel.dampen_time, self._dt, include_alpha=False)
        self._dampen(self._output_fbo, den.dampen_threshold, den.dampen_time, self._dt, include_alpha=True)

        # Simulation steps
        self._advect_density()
        self._advect_velocity()
        self._diffuse_velocity()
        self._apply_vorticity()
        self._advect_temperature_and_buoyancy()
        self._advect_pressure()
        self._project_pressure()

    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Add internal force to velocity (no input_strength or dt scaling).

        Use for simulation-internal forces (vorticity confinement, buoyancy)
        that already carry their own dt scaling. Matches FluidFlow3D's
        _add_force_to_velocity for 2D/3D parity.
        """
        FlowUtil.add(self._input_fbo, texture, strength)

    def _advect_density(self) -> None:
        """Step 1: Advect & dissipate density field."""
        advect_step = self._dt * (self.config.speed + self.config.density.speed_offset)
        dissipation = self._calculate_dissipation(self._dt, self.config.density.fade_time)

        self._output_fbo.swap()
        self._output_fbo.begin()
        self._advect_shader.use(
            self._output_fbo.back_texture,  # Source density
            self._input_fbo.texture,        # Velocity
            self._density_obstacle_fbo.texture,  # Obstacles (full density resolution)
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._output_fbo.end()

    def _advect_velocity(self) -> None:
        """Step 2: Self-advect & dissipate velocity field."""
        advect_step = self._dt * self.config.velocity.self_advection
        dissipation = self._calculate_dissipation(self._dt, self.config.velocity.fade_time)

        self._input_fbo.swap()
        self._input_fbo.begin()
        self._advect_shader.use(
            self._input_fbo.back_texture,   # Source velocity (self-advection)
            self._input_fbo.back_texture,   # Velocity
            self._obstacle_fbo.texture,     # Obstacles
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._input_fbo.end()

    def _diffuse_velocity(self) -> None:
        """Step 3: Viscosity diffusion (Jacobi solver)."""
        if self.config.velocity.viscosity <= 0.0:
            return

        viscosity_dt = self.config.velocity.viscosity * (self.config.simulation_scale ** 2) * self._dt

        if self._use_compute_diffusion:
            result = self._jacobi_diffusion_compute.solve(
                self._input_fbo.texture,
                self._input_fbo.back_texture,
                self._obstacle_fbo.texture,
                self.config.simulation_scale,
                self._aspect,
                viscosity_dt,
                total_iterations=self.config.velocity.viscosity_iter,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles
            )
            if result != self._input_fbo.texture:
                self._input_fbo.swap()
        else:
            for _ in range(self.config.velocity.viscosity_iter):
                self._input_fbo.swap()
                self._input_fbo.begin()
                self._jacobi_diffusion_shader.use(
                    self._input_fbo.back_texture,
                    self._obstacle_fbo.texture,
                    self.config.simulation_scale,
                    self._aspect,
                    viscosity_dt,
                    has_obstacles=self._has_obstacles
                )
                self._input_fbo.end()

    def _apply_vorticity(self) -> None:
        """Step 4: Vorticity confinement (curl → force → add to velocity)."""
        if self.config.velocity.vorticity <= 0.0 or self.config.velocity.vorticity_radius <= 0.0:
            return

        vorticity_radius = self.config.velocity.vorticity_radius * self.config.simulation_scale
        vorticity_force = self.config.velocity.vorticity * self._dt

        # 4a. Compute vorticity curl
        self._vorticity_curl_fbo.begin()
        self._vorticity_curl_shader.use(
            self._input_fbo.texture,
            self._obstacle_fbo.texture,
            self.config.simulation_scale,
            self._aspect,
            vorticity_radius,
            has_obstacles=self._has_obstacles
        )
        self._vorticity_curl_fbo.end()

        # 4b. Compute confinement force
        self._vorticity_force_fbo.begin()
        self._vorticity_force_shader.use(
            self._vorticity_curl_fbo.texture,
            self._obstacle_fbo.texture,
            self.config.simulation_scale,
            self._aspect,
            vorticity_force,
            has_obstacles=self._has_obstacles
        )
        self._vorticity_force_fbo.end()

        # 4c. Add force to velocity
        self._add_force_to_velocity(self._vorticity_force_fbo.texture)

    def _advect_temperature_and_buoyancy(self) -> None:
        """Steps 5–6: Advect temperature & apply buoyancy force to velocity."""
        if self.config.temperature.buoyancy == 0.0:
            FlowUtil.zero(self._temperature_fbo)
            return

        # 5. Advect temperature
        advect_step = self._dt * self.config.speed
        dissipation = self._calculate_dissipation(self._dt, self.config.temperature.fade_time)

        self._temperature_fbo.swap()
        self._temperature_fbo.begin()
        self._advect_shader.use(
            self._temperature_fbo.back_texture,
            self._input_fbo.texture,
            self._obstacle_fbo.texture,
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._temperature_fbo.end()

        # 6. Buoyancy: F = σ(T - T_ambient) - κρ
        sigma = self._dt * self.config.simulation_scale * self.config.temperature.buoyancy
        kappa = self._dt * self.config.simulation_scale * self.config.temperature.weight

        self._buoyancy_fbo.begin()
        self._buoyancy_shader.use(
            self._input_fbo.texture,
            self._temperature_fbo.texture,
            self._output_fbo.texture,
            self._obstacle_fbo.texture,
            sigma,
            kappa,
            self.config.temperature.ambient,
            has_obstacles=self._has_obstacles
        )
        self._buoyancy_fbo.end()

        self._add_force_to_velocity(self._buoyancy_fbo.texture)

    def _advect_pressure(self) -> None:
        """Step 7: Advect & dissipate pressure (optional, non-physical)."""
        if self.config.pressure.speed <= 0.0:
            return

        advect_step = self._dt * self.config.pressure.speed
        dissipation = self._calculate_dissipation(self._dt, self.config.pressure.fade_time)

        self._pressure_fbo.swap()
        self._pressure_fbo.begin()
        self._advect_shader.use(
            self._pressure_fbo.back_texture,
            self._input_fbo.texture,
            self._obstacle_fbo.texture,
            self._aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._pressure_fbo.end()

    def _project_pressure(self) -> None:
        """Step 8: Pressure projection (divergence → Jacobi solve → gradient subtraction)."""
        # 8a. Compute divergence
        self._divergence_fbo.begin()
        self._divergence_shader.use(
            self._input_fbo.texture,
            self._obstacle_fbo.texture,
            self.config.simulation_scale,
            self._aspect,
            has_obstacles=self._has_obstacles
        )
        self._divergence_fbo.end()

        # 8b. Solve Poisson equation for pressure
        if self._use_compute_pressure:
            result = self._jacobi_pressure_compute.solve(
                self._pressure_fbo.texture,
                self._pressure_fbo.back_texture,
                self._divergence_fbo.texture,
                self._obstacle_fbo.texture,
                self.config.simulation_scale,
                self._aspect,
                total_iterations=self.config.pressure.iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles
            )
            if result != self._pressure_fbo.texture:
                self._pressure_fbo.swap()
        else:
            for _ in range(self.config.pressure.iterations):
                self._pressure_fbo.swap()
                self._pressure_fbo.begin()
                self._jacobi_pressure_shader.use(
                    self._pressure_fbo.back_texture,
                    self._divergence_fbo.texture,
                    self._obstacle_fbo.texture,
                    self.config.simulation_scale,
                    self._aspect,
                    has_obstacles=self._has_obstacles
                )
                self._pressure_fbo.end()

        # 8c. Subtract pressure gradient from velocity
        self._input_fbo.swap()
        self._input_fbo.begin()
        self._gradient_shader.use(
            self._input_fbo.back_texture,
            self._pressure_fbo.texture,
            self._obstacle_fbo.texture,
            self.config.simulation_scale,
            self._aspect,
            has_obstacles=self._has_obstacles
        )
        self._input_fbo.end()

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
        if self._reset_pending:
            self._reset_pending = False
            self.reset()

        if self._reallocate_pending:
            self._reallocate_pending = False
            sim_scale = self.config.simulation_scale
            new_w = self._align16(int(self._density_width * sim_scale))
            new_h = self._align16(int(self._density_height * sim_scale))
            if new_w != self._width or new_h != self._height:
                self._width = new_w
                self._height = new_h
                self._aspect = self._width / self._height if self._height > 0 else 1.0
                self._allocate_fbos()

    def _request_reset(self) -> None:
        """Thread-safe reset request — deferred to next update() on the GL thread."""
        self._reset_pending = True

    def _request_reallocate(self) -> None:
        """Thread-safe reallocation request — deferred to next update() on the GL thread."""
        self._reallocate_pending = True

    # ========== Internal helpers ==========

    @staticmethod
    def _align16(v: int) -> int:
        """Round up to the nearest multiple of 16."""
        return (v + 15) & ~15

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

    # ========== Obstacles ==========

    # ========== Input Methods ==========

    # ----- Velocity -----
    def set_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Set velocity field."""
        FlowUtil.set(self._input_fbo, texture, strength)

    def set_velocity_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set velocity in a specific region."""
        FlowUtil.set_region(self._input_fbo, texture, x, y, w, h, strength)

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to velocity field. Applies config velocity.input_strength and delta_time."""
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.velocity.input_strength * dt
        FlowUtil.add(self._input_fbo, texture, effective)

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
        """Add to density field. Applies config density.input_strength and delta_time."""
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.density.input_strength * dt
        FlowUtil.add(self._output_fbo, texture, effective)

    def add_density_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add density to a specific region."""
        FlowUtil.add_region(self._output_fbo, texture, x, y, w, h, strength)

    def add_density_channel(self, texture: Texture, channel: int, strength: float = 1.0) -> None:
        """Add single-channel texture to one of the density channels. Applies config density.input_strength and delta_time."""
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.density.input_strength * dt
        FlowUtil.add_channel(self._output_fbo, texture, channel, effective)

    def add_density_channel_region(self, texture: Texture, channel: int, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add density to a specific channel at a specific region."""
        FlowUtil.add_channel_region(self._output_fbo, texture, channel, x, y, w, h, strength)

    # ----- Temperature -----
    def set_temperature(self, texture: Texture) -> None:
        """Set temperature field."""
        FlowUtil.blit(self._temperature_fbo, texture)

    def set_temperature_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set temperature in a specific region."""
        FlowUtil.set_region(self._temperature_fbo, texture, x, y, w, h, strength)

    def add_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to temperature field. Applies config temperature.input_strength and delta_time."""
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.temperature.input_strength * dt
        FlowUtil.add(self._temperature_fbo, texture, effective)

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

    # ---- Obstacles -----
    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle mask with texture (boolean OR with empty field).

        Updates both sim-resolution and density-resolution obstacle buffers.

        Args:
            texture: Obstacle mask (any channel > 0 = obstacle)
        """
        FlowUtil.zero(self._obstacle_fbo)
        self._obstacle_fbo.swap()
        self._obstacle_fbo.begin()
        self._add_boolean_shader.use(self._obstacle_fbo.back_texture, texture)
        self._obstacle_fbo.end()

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
        self._obstacle_fbo.swap()
        self._obstacle_fbo.begin()
        self._add_boolean_shader.use(self._obstacle_fbo.back_texture, texture)
        self._obstacle_fbo.end()

        self._density_obstacle_fbo.swap()
        self._density_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._density_obstacle_fbo.back_texture, texture)
        self._density_obstacle_fbo.end()

        self._has_obstacles = True

    # ========== Properties ==========

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
        return self._density_obstacle_fbo.texture

    @property
    def sim_width(self) -> int:
        """Current simulation resolution width (aligned to 16)."""
        return self._width

    @property
    def sim_height(self) -> int:
        """Current simulation resolution height (aligned to 16)."""
        return self._height
