"""FluidFlow3D - 3D Navier-Stokes fluid simulation using volumetric textures.

All-compute pipeline operating on GL_TEXTURE_3D volumes with trilinear
filtering between depth layers. Produces a composited 2D output for
downstream rendering compatibility.

Pipeline:
    1. Advect density (3D semi-Lagrangian)
    2. Advect velocity (3D self-advection)
    3. Diffuse velocity (3D Jacobi viscosity solver)
    4. Vorticity confinement (3D curl + confinement force)
    5. Temperature advection + buoyancy force
    6. Pressure advection (optional, non-physical)
    7. Pressure projection (divergence -> Jacobi solve -> gradient subtraction)
    8. Composite 3D density -> 2D output

Boundary conditions via per-field wrap modes (no explicit border obstacles):
    Velocity/density:  GL_CLAMP_TO_BORDER(0,0,0,0)  -> no-slip / Dirichlet
    Pressure/temp:     GL_CLAMP_TO_EDGE              -> zero-gradient / Neumann
    Obstacle:          GL_CLAMP_TO_BORDER(1,0,0,0)   -> OOB = obstacle
"""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from modules.gl.Texture3D import Texture3D, SwapTexture3D
from modules.settings import Field, Settings, Widget
from ..fluid.fluid_config import VelocityConfig, DensityConfig, TemperatureConfig, PressureConfig
from .shaders import (
    Advect3D, Divergence3D, Gradient3D,
    JacobiPressure3D, JacobiDiffusion3D,
    VorticityCurl3D, VorticityForce3D, Buoyancy3D,
    Inject3D, InjectChannel3D, Clamp3D, Dampen3D, Composite3D, Add3D,
    InjectBinary3D
)

# Combined memory barrier bits (cast to int for Pylance compatibility)
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


from modules.utils.HotReloadMethods import HotReloadMethods


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DepthConfig(Settings):
    """3D depth/volume-specific parameters."""
    layers: Field[int]              = Field(4,    min=1,    max=64,     description="Number of depth layers in the 3D volume")
    scale: Field[float]             = Field(1.0,  min=0.5,  max=5.0,    description="Manual multiplier on auto-computed Z grid spacing (width/depth_layers)")
    composite_mode: Field[int]      = Field(1,    min=0,    max=2,      description="3D->2D compositing: 0=alpha, 1=additive, 2=max")
    injection_layer: Field[float]   = Field(0.1,  min=0.0,  max=1.0,    description="Normalized depth for 2D->3D injection center")
    injection_spread: Field[float]  = Field(0.15, min=0.01, max=0.5,    description="Gaussian sigma for depth spread during injection")

class FluidFlow3DConfig(Settings):
    """Configuration for 3D fluid simulation.

    Same speed/fade_time/vorticity/buoyancy model as FluidFlowConfig,
    with additional depth-specific parameters.
    """

    # ---- Actions ----
    reset_sim: Field[bool] = Field(False, widget=Widget.button, description="Reset all simulation fields to zero")

    # ---- Global ----
    simulation_scale: Field[float]  = Field(0.5,    min=0.1,    max=2.0,    description="Resolution scale for simulation buffers")
    fps: Field[int]                 = Field(60,     min=1,      max=240,    description="Current average FPS for dt calculation (bound from WindowManager)", access=Field.READ)
    speed: Field[float]             = Field(1.0,    min=0.0,    max=5.0,    description="Base fluid transport rate")

    # ---- Field groups ----
    depth:       DepthConfig
    velocity:    VelocityConfig
    density:     DensityConfig
    temperature: TemperatureConfig
    pressure:    PressureConfig


# ---------------------------------------------------------------------------
# 3D Fluid Simulation
# ---------------------------------------------------------------------------

class FluidFlow3D:
    """3D Navier-Stokes fluid simulation using volumetric compute shaders.

    Does NOT inherit FlowBase -- the internal representation is entirely
    different (SwapTexture3D instead of SwapFbo). Provides a compatible
    public API with the 2D FluidFlow where applicable.

    Fields:
        velocity:    SwapTexture3D RGBA16F  (u, v, w, spare)
        density:     SwapTexture3D RGBA16F  (r, g, b, a)
        temperature: SwapTexture3D R16F
        pressure:    SwapTexture3D R16F
        obstacle:    Texture3D     R8       (no swap needed -- set externally)

    Intermediate (single-buffer):
        divergence:       Texture3D R16F
        curl:             Texture3D RGBA16F  (wx, wy, wz, 0)
        vorticity_force:  Texture3D RGBA16F  (Fx, Fy, Fz, 0)
        buoyancy_force:   Texture3D RGBA16F  (Fx, Fy, Fz, 0)

    2D composited output:
        _output_texture:  Texture RGBA16F    (composited from 3D density)
    """

    def __init__(self, config: FluidFlow3DConfig | None = None) -> None:
        self.config: FluidFlow3DConfig = config or FluidFlow3DConfig()

        # ---- Simulation dimensions and state ---
        self._width: int = 0
        self._height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0

        self._depth: int = 0
        self._aspect: float = 1.0
        self._auto_depth_scale: float = 1.0

        self._dt: float = 1.0 / 60.0

        self._grid_scale: float = 0.5
        self._depth_scale: float = 1.0

        self._allocated: bool = False
        self._reset_pending: bool = False
        self._reallocate_pending: bool = False

        # ---- Volumetric fields (SwapTexture3D for ping-pong) ----
        # Velocity: CLAMP_TO_BORDER(0) = no-slip walls
        self._velocity: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Density: CLAMP_TO_BORDER(0) = nothing leaks out
        self._density: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Temperature: CLAMP_TO_EDGE = insulated walls (Neumann)
        self._temperature: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_EDGE)
        # Pressure: CLAMP_TO_EDGE = zero-gradient walls (Neumann)
        self._pressure: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_EDGE)
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._simulation_obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        # Density-resolution obstacle (for density advection when simulation_scale < 1)
        self._density_obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        self._has_obstacles: bool = False

        # ---- Intermediate volumes (single buffer, no ping-pong) ----
        self._divergence_vol: Texture3D = Texture3D()
        self._curl_vol: Texture3D = Texture3D()
        self._vorticity_force_vol: Texture3D = Texture3D()
        self._buoyancy_force_vol: Texture3D = Texture3D()

        # ---- 2D composited output (for downstream render layers) ----
        self._output_texture: Texture = Texture()

        # ---- Compute shaders ----
        self._advect_shader: Advect3D = Advect3D()
        self._divergence_shader: Divergence3D = Divergence3D()
        self._gradient_shader: Gradient3D = Gradient3D()
        self._jacobi_pressure_shader: JacobiPressure3D = JacobiPressure3D()
        self._jacobi_diffusion_shader: JacobiDiffusion3D = JacobiDiffusion3D()
        self._vorticity_curl_shader: VorticityCurl3D = VorticityCurl3D()
        self._vorticity_force_shader: VorticityForce3D = VorticityForce3D()
        self._buoyancy_shader: Buoyancy3D = Buoyancy3D()
        self._inject_shader: Inject3D = Inject3D()
        self._inject_channel_shader: InjectChannel3D = InjectChannel3D()
        self._clamp_shader: Clamp3D = Clamp3D()
        self._dampen_shader: Dampen3D = Dampen3D()
        self._composite_shader: Composite3D = Composite3D()
        self._add_shader: Add3D = Add3D()
        self._inject_binary_shader: InjectBinary3D = InjectBinary3D()

        # Bind settings actions
        self.config.bind(FluidFlow3DConfig.reset_sim, lambda _: self._request_reset())
        self.config.depth.bind(DepthConfig.layers, lambda _: self._request_reallocate())
        self.config.bind(FluidFlow3DConfig.simulation_scale, lambda _: self._request_reallocate())

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Allocation ==========

    def allocate(self, width: int, height: int) -> None:
        """Allocate all 3D volumes and shaders.

        Args:
            width: Full output resolution width (XY)
            height: Full output resolution height (XY)
        """
        self._density_width = width
        self._density_height = height

        sim_scale = self.config.simulation_scale
        self._width = self._align16(int(width * sim_scale))
        self._height = self._align16(int(height * sim_scale))
        self._aspect = self._width / self._height if self._height > 0 else 1.0
        self._depth = self.config.depth.layers

        self._allocate_volumes()

        # 2D composited output
        self._output_texture.allocate(self._density_width, self._density_height, GL_RGBA16F)

        # Allocate shaders
        self._advect_shader.allocate()
        self._divergence_shader.allocate()
        self._gradient_shader.allocate()
        self._jacobi_pressure_shader.allocate()
        self._jacobi_diffusion_shader.allocate()
        self._vorticity_curl_shader.allocate()
        self._vorticity_force_shader.allocate()
        self._buoyancy_shader.allocate()
        self._inject_shader.allocate()
        self._inject_channel_shader.allocate()
        self._clamp_shader.allocate()
        self._dampen_shader.allocate()
        self._composite_shader.allocate()
        self._add_shader.allocate()
        self._inject_binary_shader.allocate()

        self._allocated = True

        # DEBUG: inject test obstacle shapes
        # self._debug_inject_obstacle_shapes()

    def _debug_inject_obstacle_shapes(self) -> None:
        """Draw circle, triangle, cross, and line into the obstacle volume for visual verification.

        Reuses the same 2D mask pattern as FluidFlow._debug_inject_obstacle_shapes,
        projected through all depth layers via set_obstacle() → InjectBinary3D.
        """
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
        top_y = ty + s
        bot_y = ty - int(s * 0.6)
        for y_px in range(max(0, bot_y), min(h, top_y + 1)):
            for x_px in range(max(0, tx - s), min(w, tx + s + 1)):
                t_frac = (y_px - bot_y) / max(1, top_y - bot_y)
                half_w = s * (1.0 - t_frac)
                if abs(x_px - tx) <= half_w:
                    mask[y_px, x_px] = 255

        # ---- Cross (bottom-left quadrant) ----
        cx2, cy2 = w // 4, h // 4
        arm = min(w, h) // 8
        thick = max(2, min(w, h) // 40)
        mask[max(0, cy2 - thick):min(h, cy2 + thick + 1),
             max(0, cx2 - arm):min(w, cx2 + arm + 1)] = 255
        mask[max(0, cy2 - arm):min(h, cy2 + arm + 1),
             max(0, cx2 - thick):min(w, cx2 + thick + 1)] = 255

        # ---- Diagonal line (bottom-right quadrant) ----
        lx, ly = w * 3 // 4, h // 4
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

        # Upload to a temporary 2D texture and inject through all layers
        from modules.gl import Texture
        temp = Texture()
        temp.allocate(w, h, GL_R8)
        glBindTexture(GL_TEXTURE_2D, temp.tex_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, mask)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.set_obstacle(temp)
        temp.deallocate()

    def _allocate_volumes(self) -> None:
        """(Re)allocate all 3D volumetric fields at current dimensions.

        Uses self._width, self._height, self._depth (and self._density_width/
        self._density_height for the density field).  Shaders are not touched
        because they carry no resolution-dependent state.
        """
        self._auto_depth_scale = self._width / max(1, self._depth)
        d = self._depth
        sim_w = self._width
        sim_h = self._height

        # Primary simulation fields
        self._velocity.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._velocity.clear_all()

        self._density.allocate(self._density_width, self._density_height, d, GL_RGBA16F)
        self._density.clear_all()

        self._temperature.allocate(sim_w, sim_h, d, GL_R16F)
        self._temperature.clear_all()

        self._pressure.allocate(sim_w, sim_h, d, GL_R16F)
        self._pressure.clear_all()

        self._simulation_obstacle.allocate(sim_w, sim_h, d, GL_R8)
        self._simulation_obstacle.clear()
        self._density_obstacle.allocate(self._density_width, self._density_height, d, GL_R8)
        self._density_obstacle.clear()
        self._has_obstacles = False

        # Intermediate volumes
        self._divergence_vol.allocate(sim_w, sim_h, d, GL_R16F)
        self._divergence_vol.clear()

        self._curl_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._curl_vol.clear()

        self._vorticity_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._vorticity_force_vol.clear()

        self._buoyancy_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._buoyancy_force_vol.clear()

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._velocity.deallocate()
        self._density.deallocate()
        self._temperature.deallocate()
        self._pressure.deallocate()
        self._density_obstacle.deallocate()
        self._simulation_obstacle.deallocate()
        self._divergence_vol.deallocate()
        self._curl_vol.deallocate()
        self._vorticity_force_vol.deallocate()
        self._buoyancy_force_vol.deallocate()
        self._output_texture.deallocate()

        self._advect_shader.deallocate()
        self._divergence_shader.deallocate()
        self._gradient_shader.deallocate()
        self._jacobi_pressure_shader.deallocate()
        self._jacobi_diffusion_shader.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._buoyancy_shader.deallocate()
        self._inject_shader.deallocate()
        self._inject_channel_shader.deallocate()
        self._clamp_shader.deallocate()
        self._dampen_shader.deallocate()
        self._composite_shader.deallocate()
        self._add_shader.deallocate()
        self._inject_binary_shader.deallocate()

        self._allocated = False

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders."""
        self._advect_shader.reload()
        self._divergence_shader.reload()
        self._gradient_shader.reload()
        self._jacobi_pressure_shader.reload()
        self._jacobi_diffusion_shader.reload()
        self._vorticity_curl_shader.reload()
        self._vorticity_force_shader.reload()
        self._buoyancy_shader.reload()
        self._inject_shader.reload()
        self._inject_channel_shader.reload()
        self._clamp_shader.reload()
        self._dampen_shader.reload()
        self._composite_shader.reload()
        self._add_shader.reload()
        self._inject_binary_shader.reload()

    # ========== Update Pipeline ==========

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        if not self._allocated:
            return
        self._velocity.clear_all()
        self._density.clear_all()
        self._temperature.clear_all()
        self._pressure.clear_all()
        self._divergence_vol.clear()
        self._curl_vol.clear()

    def update(self) -> None:
        """Run one frame of the 3D fluid simulation pipeline."""
        if not self._allocated:
            return

        self._reload_shaders()
        self._handle_deferred_actions()

        # Per-frame state
        self._dt = 1.0 / max(1, self.config.fps)
        self._aspect = self._width / self._height if self._height > 0 else 1.0
        self._depth_scale = self._auto_depth_scale * self.config.depth.scale
        self._grid_scale = self.config.simulation_scale

        # Simulation steps
        self._advect_velocity()
        self._diffuse_velocity()
        self._apply_vorticity()
        self._advect_pressure()
        self._project_pressure()
        self._advect_temperature_and_buoyancy()
        self._advect_density()
        self._composite_output()

        # Dampen inputs before simulation
        vel = self.config.velocity
        den = self.config.density
        self._dampen(self._velocity, vel.dampen_threshold, vel.dampen_time, self._dt, include_alpha=False)
        self._dampen(self._density,  den.dampen_threshold, den.dampen_time, self._dt, include_alpha=True)


    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture3D, strength: float = 1.0) -> None:
        """Add 3D force volume to velocity in-place (no swap needed)."""
        self._add_shader.use(self._velocity.texture, force, strength)
        glMemoryBarrier(_BARRIER_IMAGE)

    def _advect_velocity(self) -> None:
        """Step 2: Self-advect & dissipate velocity field."""
        advect_step = self._dt * self.config.velocity.self_advection
        dissipation = self._calculate_dissipation(self._dt, self.config.velocity.fade_time)

        self._velocity.swap()
        self._advect_shader.advect(
            self._velocity.back_texture,
            self._velocity.texture,
            self._velocity.back_texture,
            self._simulation_obstacle,
            self._aspect, self._depth_scale,
            advect_step, dissipation,
            internal_format=GL_RGBA16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _diffuse_velocity(self) -> None:
        """Step 3: Viscosity diffusion (Jacobi solver)."""
        if self.config.velocity.viscosity <= 0.0:
            return

        viscosity_dt = self.config.velocity.viscosity * (self.config.simulation_scale ** 2) * self._dt

        result = self._jacobi_diffusion_shader.solve(
            self._velocity.texture,
            self._velocity.back_texture,
            self._simulation_obstacle,
            self._grid_scale, self._aspect, self._depth_scale,
            viscosity_dt,
            total_iterations=self.config.velocity.viscosity_iter,
            iterations_per_dispatch=5,
            has_obstacles=self._has_obstacles
        )
        if result != self._velocity.texture:
            self._velocity.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _apply_vorticity(self) -> None:
        """Step 4: Vorticity confinement (curl → force → add to velocity)."""
        if self.config.velocity.vorticity <= 0.0 or self.config.velocity.vorticity_radius <= 0.0:
            return

        vorticity_radius = self.config.velocity.vorticity_radius * self.config.simulation_scale
        vorticity_force = self.config.velocity.vorticity * self._dt

        # 4a. Compute 3D curl
        self._vorticity_curl_shader.use(
            self._velocity.texture,
            self._simulation_obstacle,
            self._curl_vol,
            self._grid_scale, self._aspect, self._depth_scale,
            vorticity_radius,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # 4b. Compute confinement force
        self._vorticity_force_shader.use(
            self._curl_vol,
            self._simulation_obstacle,
            self._vorticity_force_vol,
            self._grid_scale, self._aspect, self._depth_scale,
            vorticity_force,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_IMAGE)

        # 4c. Add force to velocity
        self._add_force_to_velocity(self._vorticity_force_vol)

    def _advect_temperature_and_buoyancy(self) -> None:
        """Steps 5–6: Advect temperature & apply buoyancy force to velocity."""
        if self.config.temperature.buoyancy == 0.0:
            self._temperature.clear_all()
            return

        # 5. Advect temperature
        advect_step = self._dt * self.config.speed
        dissipation = self._calculate_dissipation(self._dt, self.config.temperature.fade_time)

        self._temperature.swap()
        self._advect_shader.advect(
            self._temperature.back_texture,
            self._temperature.texture,
            self._velocity.texture,
            self._simulation_obstacle,
            self._aspect, self._depth_scale,
            advect_step, dissipation,
            internal_format=GL_R16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # 6. Buoyancy: F = sigma*(T - T_ambient) - kappa*density
        sigma = self._dt * self.config.simulation_scale * self.config.temperature.buoyancy
        kappa = self._dt * self.config.simulation_scale * self.config.temperature.weight

        self._buoyancy_shader.use(
            self._temperature.texture,
            self._density.texture,
            self._simulation_obstacle,
            self._buoyancy_force_vol,
            sigma, kappa, self.config.temperature.ambient,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_IMAGE)

        self._add_force_to_velocity(self._buoyancy_force_vol)

    def _advect_pressure(self) -> None:
        """Step 7: Advect & dissipate pressure (optional, non-physical)."""
        if self.config.pressure.speed <= 0.0:
            return

        advect_step = self._dt * self.config.pressure.speed
        dissipation = self._calculate_dissipation(self._dt, self.config.pressure.fade_time)

        self._pressure.swap()
        self._advect_shader.advect(
            self._pressure.back_texture,
            self._pressure.texture,
            self._velocity.texture,
            self._simulation_obstacle,
            self._aspect, self._depth_scale,
            advect_step, dissipation,
            internal_format=GL_R16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _project_pressure(self) -> None:
        """Step 8: Pressure projection (divergence → Jacobi solve → gradient subtraction)."""
        # 8a. Compute divergence
        self._divergence_shader.use(
            self._velocity.texture,
            self._simulation_obstacle,
            self._divergence_vol,
            self._grid_scale, self._aspect, self._depth_scale,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # 8b. Solve Poisson equation for pressure
        result = self._jacobi_pressure_shader.solve(
            self._pressure.texture,
            self._pressure.back_texture,
            self._divergence_vol,
            self._simulation_obstacle,
            self._grid_scale, self._aspect, self._depth_scale,
            total_iterations=self.config.pressure.iterations,
            iterations_per_dispatch=5,
            has_obstacles=self._has_obstacles
        )
        if result != self._pressure.texture:
            self._pressure.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # 8c. Subtract pressure gradient from velocity
        self._velocity.swap()
        self._gradient_shader.use(
            self._velocity.back_texture,
            self._pressure.texture,
            self._simulation_obstacle,
            self._velocity.texture,
            self._grid_scale, self._aspect, self._depth_scale,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_density(self) -> None:
        """Step 1: Advect & dissipate density field."""
        advect_step = self._dt * (self.config.speed + self.config.density.speed_offset)
        dissipation = self._calculate_dissipation(self._dt, self.config.density.fade_time)

        self._density.swap()
        self._advect_shader.advect(
            self._density.back_texture,
            self._density.texture,
            self._velocity.texture,
            self._density_obstacle,
            self._aspect, self._depth_scale,
            advect_step, dissipation,
            internal_format=GL_RGBA16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _dampen(self, volume: SwapTexture3D, threshold: float, dampen_time: float,
               delta_time: float, include_alpha: bool,
               internal_format: int = GL_RGBA16F) -> None:
        """Exponential drag on magnitude excess above threshold.

        Args:
            volume: Volumetric field to dampen (in-place via compute shader)
            threshold: Magnitude below which values are untouched
            dampen_time: Seconds for excess to decay to ~1%. 0=off
            delta_time: Frame delta time for frame-rate independence
            include_alpha: True for density (RGBA magnitude), False for velocity (RGB)
            internal_format: Image format for volume binding
        """
        if dampen_time <= 0.0:
            return
        factor = pow(0.01, delta_time / dampen_time)
        self._dampen_shader.use(
            volume.texture, threshold, factor,
            include_alpha=include_alpha,
            internal_format=internal_format
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def _composite_output(self) -> None:
        """Step 9: Composite 3D density → 2D output texture."""
        self._composite_shader.use(
            self._density.texture,
            self._output_texture,
            self.config.depth.composite_mode
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    # ========== Deferred ==========

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            new_depth = self.config.depth.layers
            sim_scale = self.config.simulation_scale
            new_w = self._align16(int(self._density_width * sim_scale))
            new_h = self._align16(int(self._density_height * sim_scale))
            if new_depth != self._depth or new_w != self._width or new_h != self._height:
                self._width = new_w
                self._height = new_h
                self._aspect = self._width / self._height if self._height > 0 else 1.0
                self._depth = new_depth
                self._allocate_volumes()

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

    @staticmethod
    def _align16(v: int) -> int:
        """Round up to the nearest multiple of 16."""
        return (v + 15) & ~15

    @staticmethod
    def _calculate_dissipation(delta_time: float, decay_time: float) -> float:
        """Calculate frame-rate independent decay multiplier.

        Returns pow(0.01, dt / decay_time) -- field reaches 1% after decay_time seconds.
        """
        return pow(0.01, delta_time / max(0.001, decay_time))

    # ========== Input Methods (2D -> 3D injection) ==========

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D velocity texture into 3D velocity volume.

        Uses gaussian depth spread centered at config.depth.injection_layer.
        Applies config velocity.input_strength and delta_time.
        """
        if not self._allocated:
            return
        effective = strength * self.config.velocity.input_strength * (1.0 / max(1, self.config.fps))
        self._inject_shader.use(
            texture, self._velocity.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            effective, mode=0,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def set_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Set (replace) 3D velocity volume from 2D texture.

        Gaussian depth spread, replace mode.
        """
        if not self._allocated:
            return
        self._velocity.clear_all()
        self._inject_shader.use(
            texture, self._velocity.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            strength, mode=1,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D density texture into 3D density volume. Applies config density.input_strength and delta_time."""
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.density.input_strength * dt
        self._inject_shader.use(
            texture, self._density.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            effective, mode=0,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def set_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Set (replace) 3D density volume from 2D texture."""
        if not self._allocated:
            return
        self._density.clear_all()
        self._inject_shader.use(
            texture, self._density.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            strength, mode=1,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_density_channel(self, texture: Texture, channel: int,
                            strength: float = 1.0) -> None:
        """Inject single-channel 2D texture into one RGBA channel of the 3D density volume.

        Uses gaussian depth spread centered at config.depth.injection_layer.
        Applies config density.input_strength.

        Args:
            texture: 2D source texture (reads .r component)
            channel: Target RGBA channel (0=R, 1=G, 2=B, 3=A)
            strength: Injection strength multiplier
        """
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.density.input_strength * dt
        self._inject_channel_shader.use(
            texture, self._density.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            channel, effective, mode=0,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D temperature texture into 3D temperature volume. Applies config temperature.input_strength and delta_time."""
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.temperature.input_strength * dt
        self._inject_shader.use(
            texture, self._temperature.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            effective, mode=0,
            internal_format=GL_R16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def set_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Set 3D temperature volume from 2D texture."""
        if not self._allocated:
            return
        self._temperature.clear_all()
        self._inject_shader.use(
            texture, self._temperature.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            strength, mode=1,
            internal_format=GL_R16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_pressure(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D pressure texture into 3D pressure volume."""
        if not self._allocated:
            return
        self._inject_shader.use(
            texture, self._pressure.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            strength, mode=0,
            internal_format=GL_R16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    # ----- Obstacles -----
    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle volume with a 2D mask projected through all layers.

        Updates both sim-resolution and density-resolution obstacle volumes.

        Args:
            texture: 2D obstacle mask (any channel > 0.5 = obstacle)
        """
        if not self._allocated:
            return
        self._inject_binary_shader.use(texture, self._simulation_obstacle, mode=1)
        self._inject_binary_shader.use(texture, self._density_obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def add_obstacle(self, texture: Texture) -> None:
        """Add to obstacle volume (boolean OR with existing obstacles).

        Updates both sim-resolution and density-resolution obstacle volumes.

        Args:
            texture: 2D obstacle mask to add (any channel > 0.5 = obstacle)
        """
        if not self._allocated:
            return
        self._inject_binary_shader.use(texture, self._simulation_obstacle, mode=0)
        self._inject_binary_shader.use(texture, self._density_obstacle, mode=0)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        if not self._allocated:
            return
        self._simulation_obstacle.clear()
        self._density_obstacle.clear()
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = False

    # ========== Properties ==========

    @property
    def allocated(self) -> bool:
        return self._allocated

    @property
    def velocity_volume(self) -> Texture3D:
        """RGBA16F velocity volume (u,v,w in xyz)."""
        return self._velocity.texture

    @property
    def density_volume(self) -> Texture3D:
        """RGBA16F density volume."""
        return self._density.texture

    @property
    def temperature_volume(self) -> Texture3D:
        """R16F temperature volume."""
        return self._temperature.texture

    @property
    def pressure_volume(self) -> Texture3D:
        """R16F pressure volume."""
        return self._pressure.texture

    @property
    def obstacle_volume(self) -> Texture3D:
        """R8 obstacle mask volume (sim resolution)."""
        return self._simulation_obstacle

    @property
    def density_obstacle_volume(self) -> Texture3D:
        """R8 obstacle mask volume (density resolution)."""
        return self._density_obstacle

    @property
    def divergence_volume(self) -> Texture3D:
        """R16F divergence volume (intermediate)."""
        return self._divergence_vol

    @property
    def curl_volume(self) -> Texture3D:
        """RGBA16F curl/vorticity vector volume (intermediate)."""
        return self._curl_vol

    @property
    def density(self) -> Texture:
        """2D composited density output for downstream render layers."""
        return self._output_texture

    @property
    def velocity(self) -> Texture3D:
        """Alias for velocity_volume (primary field access)."""
        return self._velocity.texture

    @property
    def depth(self) -> int:
        """Number of depth layers."""
        return self._depth

    @property
    def sim_width(self) -> int:
        """Current simulation resolution width (aligned to 16)."""
        return self._width

    @property
    def sim_height(self) -> int:
        """Current simulation resolution height (aligned to 16)."""
        return self._height
