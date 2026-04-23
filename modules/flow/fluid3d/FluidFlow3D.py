"""FluidFlow3D - 3D Navier-Stokes fluid simulation using volumetric textures.

All-compute pipeline operating on GL_TEXTURE_3D volumes with trilinear
filtering between depth layers. Produces a composited 2D output for
downstream rendering compatibility.

Density is split into two volumes:
  • R16F  density (output resolution) — scalar brightness V = max(r,g,b)
  • RGBA16F colour  (sim resolution)  — RGB colour, bilinearly upsampled

Pipeline:
    1. Dampen velocity (magnitude clamping)
    2. Advect velocity (3D self-advection)
    3. Apply viscosity (3D Jacobi diffusion solver)
    4. Confine vorticity (3D curl + confinement force)
    5. Advect temperature
    6. Apply buoyancy force
    7. Advect pressure (optional, non-physical)
    8. Enforce incompressibility (divergence -> Jacobi solve -> gradient subtraction)
    9. Advect colour   (RGBA16F RGB,  3D semi-Lagrangian, sim resolution)
   10. Fused: advect density (R16F, output res) + colour lookup + 2D composite

Boundary conditions via per-field wrap modes (no explicit border obstacles):
    Velocity/density:  GL_CLAMP_TO_BORDER(0,0,0,0)  -> no-slip / Dirichlet
    Pressure/temp:     GL_CLAMP_TO_EDGE              -> zero-gradient / Neumann
    Obstacle:          GL_CLAMP_TO_BORDER(1,0,0,0)   -> OOB = obstacle
"""
from OpenGL.GL import *  # type: ignore
import time

from modules.gl import SwapFbo, Texture, Texture3D, SwapTexture3D
from ..fluid.fluid_config import FluidFlowSettings, DepthSettings, VelocitySettings
from ..fluid.shaders import AddBoolean
from ..FlowUtil import FlowUtil
from .shaders import (
    Advect3D, Divergence3D, Gradient3D,
    JacobiPressure3D, JacobiDiffusion3D,
    VorticityCurl3D, VorticityForce3D, Buoyancy3D,
    Inject3D, InjectChannel3D, InjectValue3D,
    Clamp3D, Dampen3D, Composite3D, AdvectComposite3D, Add3D,
    Blit3D, InjectBinary3D
)

# Combined memory barrier bits (cast to int for Pylance compatibility)
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


from modules.utils import HotReloadMethods

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3D Fluid Simulation
# ---------------------------------------------------------------------------

class FluidFlow3D:
    """3D Navier-Stokes fluid simulation using volumetric compute shaders.

    Does NOT inherit FlowBase -- the internal representation is entirely
    different (SwapTexture3D instead of SwapFbo). Provides a compatible
    public API with the 2D FluidFlow where applicable.

    Fields:
        velocity:    SwapTexture3D RGBA16F  (u, v, w, spare)   — sim resolution
        density:     SwapTexture3D R16F     scalar brightness   — output resolution
        color:       SwapTexture3D RGBA16F  RGB colour          — sim resolution
        temperature: SwapTexture3D R16F                        — sim resolution
        pressure:    SwapTexture3D R16F                        — sim resolution
        obstacle:    Texture3D     R8       (no swap needed -- set externally)

    Intermediate (single-buffer):
        divergence:       Texture3D R16F
        curl:             Texture3D RGBA16F  (wx, wy, wz, 0)
        vorticity_force:  Texture3D RGBA16F  (Fx, Fy, Fz, 0)
        buoyancy_force:   Texture3D RGBA16F  (Fx, Fy, Fz, 0)

    2D composited output:
        _output_texture:  Texture RGBA16F    (composited from density * colour)
    """

    def __init__(self, config: FluidFlowSettings | None = None) -> None:
        self.config: FluidFlowSettings = config or FluidFlowSettings()

        # ---- Simulation dimensions and state ---
        self._simulation_width: int = 0
        self._simulation_height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0
        self._depth: int = 0
        self._aspect: float = 1.0

        self._reference_dt: float = 1.0 / 60.0  # iteration scaling baseline (60fps)
        self._dt: float = self._reference_dt

        self._depth_aspect: float = 1.0

        self._has_obstacles: bool = False

        self._allocated: bool = False

        self._reset_pending: bool = False
        self._reallocate_pending: bool = False

        # ---- GPU profiling ----
        self._profile_enabled: bool = True
        self._profile_frame_count: int = 0
        self._profile_interval: int = 120  # print every N frames
        self._profile_accum: dict[str, float] = {}
        self._profile_queries: list[int] = []  # reusable query object pool

        # ---- Volumetric fields (SwapTexture3D for ping-pong) ----
        # Velocity: CLAMP_TO_BORDER(0) = no-slip walls
        self._velocity: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Density: CLAMP_TO_BORDER(0) = R16F scalar brightness at output resolution
        self._density: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Colour: CLAMP_TO_BORDER(0) = RGBA16F colour at sim resolution
        self._color: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Temperature: CLAMP_TO_EDGE = insulated walls (Neumann)
        self._temperature: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_EDGE)
        # Pressure: CLAMP_TO_EDGE = zero-gradient walls (Neumann)
        self._pressure: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_EDGE)
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._simulation_obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        # Density-resolution obstacle (for density advection when grid_scale < 1)
        self._density_obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        # 2D obstacle source (authoritative, never affected by depth/sim_scale changes)
        self._obstacle_source: SwapFbo = SwapFbo(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))

        # ---- Intermediate volumes (single buffer, no ping-pong) ----
        self._divergence_vol: Texture3D = Texture3D()
        # Curl is sampled at neighbor offsets by VorticityForce3D — border must return zero
        self._curl_vol: Texture3D = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
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
        self._inject_value_shader: InjectValue3D = InjectValue3D()
        self._clamp_shader: Clamp3D = Clamp3D()
        self._dampen_shader: Dampen3D = Dampen3D()
        self._composite_shader: Composite3D = Composite3D()
        self._advect_composite_shader: AdvectComposite3D = AdvectComposite3D()
        self._add_shader: Add3D = Add3D()
        self._blit_shader: Blit3D = Blit3D()
        self._inject_binary_shader: InjectBinary3D = InjectBinary3D()
        self._add_boolean_shader: AddBoolean = AddBoolean()

        # Bind settings actions
        self.config.bind(FluidFlowSettings.reset_sim, lambda _: self._request_reset())
        self.config.depth.bind(DepthSettings.depth, lambda _: self._request_reallocate())
        self.config.bind(FluidFlowSettings.width, lambda _: self._request_reallocate())
        self.config.bind(FluidFlowSettings.height, lambda _: self._request_reallocate())

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
        self._update_simulation_dimensions()

        self._allocate_simulation_fields()
        self._allocate_density_fields()

        # 2D obstacle source (full output resolution, depth-independent)
        self._obstacle_source.allocate(self._density_width, self._density_height, GL_R8)

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
        self._inject_value_shader.allocate()
        self._clamp_shader.allocate()
        self._dampen_shader.allocate()
        self._composite_shader.allocate()
        self._advect_composite_shader.allocate()
        self._add_shader.allocate()
        self._blit_shader.allocate()
        self._inject_binary_shader.allocate()
        self._add_boolean_shader.allocate()

        self._allocated = True

        # DEBUG: inject test obstacle shapes
        # from ..fluid.debug_utils import upload_debug_obstacle
        # upload_debug_obstacle(self, self._simulation_width, self._simulation_height)

    def _update_simulation_dimensions(self) -> None:
        """Recompute simulation dimensions from current config."""
        self._simulation_width = self._align32(self.config.width)
        self._simulation_height = self._align32(self.config.height)
        self._aspect = self._simulation_width / self._simulation_height if self._simulation_height > 0 else 1.0
        self._grid_scale = self._simulation_width / self._density_width if self._density_width > 0 else 1.0
        self._depth = self.config.depth.depth
        self._depth_aspect = (self._simulation_width / max(1, self._depth)) * self.config.depth.scale

    def _allocate_density_fields(self) -> None:
        """(Re)allocate density-resolution volumes (full output resolution x depth)."""
        self._density.allocate(self._density_width, self._density_height, self._depth, GL_R16F)
        self._density_obstacle.allocate(self._density_width, self._density_height, self._depth, GL_R8)
        self._regenerate_obstacle_volumes()

    def _allocate_simulation_fields(self) -> None:
        """(Re)allocate simulation-resolution 3D volumes."""
        d = self._depth
        sim_w = self._simulation_width
        sim_h = self._simulation_height

        # Primary simulation fields
        self._velocity.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._color.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._temperature.allocate(sim_w, sim_h, d, GL_R16F)
        self._pressure.allocate(sim_w, sim_h, d, GL_R16F)
        self._simulation_obstacle.allocate(sim_w, sim_h, d, GL_R8)
        self._regenerate_obstacle_volumes()

        # Intermediate volumes
        self._divergence_vol.allocate(sim_w, sim_h, d, GL_R16F)
        self._curl_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._vorticity_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._buoyancy_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)

    def _regenerate_obstacle_volumes(self) -> None:
        """Project 2D obstacle source into both 3D obstacle volumes."""
        if not self._has_obstacles:
            return
        self._inject_binary_shader.use(self._obstacle_source.texture, self._simulation_obstacle, mode=1)
        self._inject_binary_shader.use(self._obstacle_source.texture, self._density_obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._velocity.deallocate()
        self._density.deallocate()
        self._color.deallocate()
        self._temperature.deallocate()
        self._pressure.deallocate()
        self._density_obstacle.deallocate()
        self._simulation_obstacle.deallocate()
        self._obstacle_source.deallocate()
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
        self._inject_value_shader.deallocate()
        self._clamp_shader.deallocate()
        self._dampen_shader.deallocate()
        self._composite_shader.deallocate()
        self._advect_composite_shader.deallocate()
        self._add_shader.deallocate()
        self._blit_shader.deallocate()
        self._inject_binary_shader.deallocate()
        self._add_boolean_shader.deallocate()

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
        self._inject_value_shader.reload()
        self._clamp_shader.reload()
        self._dampen_shader.reload()
        self._composite_shader.reload()
        self._advect_composite_shader.reload()
        self._add_shader.reload()
        self._blit_shader.reload()
        self._inject_binary_shader.reload()

    # ========== Update Pipeline ==========

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        if not self._allocated:
            return
        self._velocity.clear_all()
        self._density.clear_all()
        self._color.clear_all()
        self._temperature.clear_all()
        self._pressure.clear_all()

    def update(self) -> None:
        """Run one frame of the 3D fluid simulation pipeline."""
        if not self._allocated:
            return

        self._reload_shaders()
        self._handle_deferred_actions()

        # Per-frame state
        self._dt = 1.0 / max(1, self.config.fps)
        self._depth_aspect = (self._simulation_width / max(1, self._depth)) * self.config.depth.scale

        profiling = self._profile_enabled

        # Dampen velocity (clean input for all steps)
        vel: VelocitySettings = self.config.velocity
        self._profile_begin("dampen_vel", profiling)
        self._dampen(self._velocity, vel.dampen_threshold, vel.dampen_time, self._dt, include_alpha=False)
        self._profile_end("dampen_vel", profiling)

        # Simulation steps
        self._profile_begin("advect_vel", profiling)
        self._advect_velocity()
        self._profile_end("advect_vel", profiling)

        self._profile_begin("viscosity", profiling)
        self._apply_viscosity()
        self._profile_end("viscosity", profiling)

        self._profile_begin("vorticity", profiling)
        self._confine_vorticity()
        self._profile_end("vorticity", profiling)

        self._profile_begin("advect_temp", profiling)
        self._advect_temperature()
        self._profile_end("advect_temp", profiling)

        self._profile_begin("buoyancy", profiling)
        self._apply_buoyancy()
        self._profile_end("buoyancy", profiling)

        self._profile_begin("advect_pres", profiling)
        self._advect_pressure()
        self._profile_end("advect_pres", profiling)

        self._profile_begin("incompress", profiling)
        self._enforce_incompressibility()
        self._profile_end("incompress", profiling)

        self._profile_begin("advect_color", profiling)
        self._advect_color()
        self._profile_end("advect_color", profiling)

        # Density dampen disabled — not meaningful for 3D and wastes a full-volume pass

        # Fused: advect density inline + colour lookup + 2D composite in one pass
        self._profile_begin("advect_composite", profiling)
        self._advect_density_and_composite()
        self._profile_end("advect_composite", profiling)

        # if profiling:
        #     self._profile_report()

    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture3D, strength: float = 1.0) -> None:
        """Add 3D force volume to velocity in-place (no swap needed)."""
        self._add_shader.use(self._velocity.texture, force, strength)
        glMemoryBarrier(_BARRIER_IMAGE)

    def _advect_velocity(self) -> None:
        """Self-advect & dissipate velocity field."""
        advect_step = self._dt * self.config.velocity.self_advection
        dissipation = self._calculate_dissipation(self._dt, self.config.velocity.fade_time)

        self._velocity.swap()
        self._advect_shader.advect(
            self._velocity.back_texture,
            self._velocity.texture,
            self._velocity.back_texture,
            self._simulation_obstacle,
            self._aspect, self._depth_aspect,
            advect_step, dissipation,
            internal_format=GL_RGBA16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _apply_viscosity(self) -> None:
        """Diffuse velocity via Jacobi viscosity solver."""
        if self.config.velocity.viscosity <= 0.0:
            return

        viscosity_dt = self.config.velocity.viscosity * (self._grid_scale ** 2) * self._dt
        iterations = self._scale_iterations(self.config.velocity.viscosity_iter)

        result = self._jacobi_diffusion_shader.solve(
            self._velocity.texture,
            self._velocity.back_texture,
            self._simulation_obstacle,
            self._grid_scale, self._aspect, self._depth_aspect,
            viscosity_dt,
            total_iterations=iterations,
            iterations_per_dispatch=5,
            has_obstacles=self._has_obstacles
        )
        if result != self._velocity.texture:
            self._velocity.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _confine_vorticity(self) -> None:
        """Vorticity confinement (curl → force → add to velocity)."""
        if self.config.velocity.vorticity <= 0.0 or self.config.velocity.vorticity_radius <= 0.0:
            return

        vorticity_radius = self.config.velocity.vorticity_radius * self._grid_scale
        vorticity_force = self.config.velocity.vorticity * self._dt

        # a. Compute curl
        self._vorticity_curl_shader.use(
            self._velocity.texture,
            self._simulation_obstacle,
            self._curl_vol,
            self._grid_scale, self._aspect, self._depth_aspect,
            vorticity_radius,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. Compute confinement force
        self._vorticity_force_shader.use(
            self._curl_vol,
            self._simulation_obstacle,
            self._vorticity_force_vol,
            self._grid_scale, self._aspect, self._depth_aspect,
            vorticity_force,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_IMAGE)

        # c. Add force to velocity
        self._add_force_to_velocity(self._vorticity_force_vol)

    def _advect_temperature(self) -> None:
        """Advect & dissipate the temperature scalar field."""
        if self.config.temperature.buoyancy == 0.0:
            self._temperature.clear_all()
            return

        advect_step = self._dt * self.config.speed
        dissipation = self._calculate_dissipation(self._dt, self.config.temperature.fade_time)

        self._temperature.swap()
        self._advect_shader.advect(
            self._temperature.back_texture,
            self._temperature.texture,
            self._velocity.texture,
            self._simulation_obstacle,
            self._aspect, self._depth_aspect,
            advect_step, dissipation,
            internal_format=GL_R16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _apply_buoyancy(self) -> None:
        """Apply buoyancy force to velocity: F = σ(T − T_ambient) − κρ."""
        if self.config.temperature.buoyancy == 0.0:
            return

        sigma = self._dt * self._grid_scale * self.config.temperature.buoyancy
        kappa = self._dt * self._grid_scale * self.config.temperature.weight

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
        """Advect & dissipate pressure (optional, non-physical)."""
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
            self._aspect, self._depth_aspect,
            advect_step, dissipation,
            internal_format=GL_R16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence → Jacobi solve → gradient subtraction)."""
        # a. Compute divergence
        self._divergence_shader.use(
            self._velocity.texture,
            self._simulation_obstacle,
            self._divergence_vol,
            self._grid_scale, self._aspect, self._depth_aspect,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. Solve Poisson equation for pressure
        iterations = self._scale_iterations(self.config.pressure.iterations)

        result = self._jacobi_pressure_shader.solve(
            self._pressure.texture,
            self._pressure.back_texture,
            self._divergence_vol,
            self._simulation_obstacle,
            self._grid_scale, self._aspect, self._depth_aspect,
            total_iterations=iterations,
            iterations_per_dispatch=5,
            has_obstacles=self._has_obstacles
        )
        if result != self._pressure.texture:
            self._pressure.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # c. Subtract pressure gradient from velocity
        self._velocity.swap()
        self._gradient_shader.use(
            self._velocity.back_texture,
            self._pressure.texture,
            self._simulation_obstacle,
            self._velocity.texture,
            self._grid_scale, self._aspect, self._depth_aspect,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_density_and_composite(self) -> None:
        """Fused: advect density inline + colour lookup + 2D composite.

        Replaces the separate _advect_density() + _composite_output() calls.
        Density advection is performed inline during the composite ray-march,
        eliminating one full output-resolution 3D dispatch and halving density
        bandwidth.
        """
        advect_step = self._dt * (self.config.speed + self.config.density.speed_offset)
        dissipation = self._calculate_dissipation(self._dt, self.config.density.fade_time)

        self._density.swap()
        self._advect_composite_shader.use(
            self._velocity.texture,
            self._density.back_texture,
            self._density.texture,
            self._color.texture,
            self._density_obstacle,
            self._output_texture,
            advect_step,
            self._aspect, self._depth_aspect,
            dissipation,
            mode=self.config.depth.composite_mode,
            absorption=self.config.depth.absorption,
            has_obstacles=self._has_obstacles,
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_color(self) -> None:
        """Advect & dissipate RGB colour field (sim resolution)."""
        advect_step = self._dt * (self.config.speed + self.config.density.speed_offset)
        dissipation = self._calculate_dissipation(self._dt, self.config.density.fade_time)

        self._color.swap()
        self._advect_shader.advect(
            self._color.back_texture,
            self._color.texture,
            self._velocity.texture,
            self._simulation_obstacle,
            self._aspect, self._depth_aspect,
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

    # ========== GPU Profiling ==========

    def _profile_begin(self, name: str, enabled: bool) -> None:
        """Insert a GPU fence before a pass (uses glFinish for accurate timing)."""
        if not enabled:
            return
        glFinish()
        self._profile_accum.setdefault(name + "_start", 0.0)
        self._profile_accum[name + "_start"] = time.perf_counter()

    def _profile_end(self, name: str, enabled: bool) -> None:
        """Record GPU time for a pass."""
        if not enabled:
            return
        glFinish()
        start = self._profile_accum.get(name + "_start", 0.0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._profile_accum[name] = self._profile_accum.get(name, 0.0) + elapsed_ms

    def _profile_report(self) -> None:
        """Print averaged per-pass timings every N frames."""
        self._profile_frame_count += 1
        if self._profile_frame_count < self._profile_interval:
            return

        n = self._profile_frame_count
        pass_names = [
            "dampen_vel", "advect_vel", "viscosity", "vorticity",
            "advect_temp", "buoyancy", "advect_pres", "incompress",
            "advect_color", "advect_composite",
        ]

        total = 0.0
        lines = [f"FluidFlow3D GPU profile ({self._simulation_width}x{self._simulation_height}x{self._depth}, "
                 f"density {self._density_width}x{self._density_height}x{self._depth}, "
                 f"avg over {n} frames):"]
        for name in pass_names:
            avg_ms = self._profile_accum.get(name, 0.0) / n
            total += avg_ms
            bar = "█" * int(avg_ms * 2)  # 1 block per 0.5ms
            lines.append(f"  {name:<14s} {avg_ms:6.2f} ms  {bar}")
        lines.append(f"  {'TOTAL':<14s} {total:6.2f} ms  ({1000.0 / max(total, 0.001):.0f} fps budget)")

        logging.info("\n".join(lines))
        logger.debug("\n".join(lines))

        # Reset accumulators
        self._profile_frame_count = 0
        for key in list(self._profile_accum.keys()):
            if not key.endswith("_start"):
                self._profile_accum[key] = 0.0

    # ========== Deferred ==========

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            old_w, old_h, old_d = self._simulation_width, self._simulation_height, self._depth
            self._update_simulation_dimensions()
            sim_changed = self._simulation_width != old_w or self._simulation_height != old_h
            depth_changed = self._depth != old_d
            if sim_changed or depth_changed:
                self._allocate_simulation_fields()
                if depth_changed:
                    self._allocate_density_fields()

        if self._reset_pending:
            self._reset_pending = False
            self.reset()

    def _request_reset(self) -> None:
        """Thread-safe reset request -- deferred to next update() on the GL thread."""
        self._reset_pending = True

    def _request_reallocate(self) -> None:
        """Thread-safe reallocation request -- deferred to next update() on the GL thread."""
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

    # ========== Input Methods (2D -> 3D injection) ==========

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D velocity texture into 3D velocity volume.

        Uses gaussian depth spread centered at config.z.injection_layer.
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

    def add_velocity_z(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D texture as Z-velocity (W channel) into 3D velocity volume.

        Positive strength pushes fluid from back layer toward front (camera).
        Uses motion magnitude as the per-pixel Z-velocity source.
        """
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        # Negate: negative W at back layer pulls fluid toward front (camera)
        effective = strength * self.config.velocity.input_strength * dt
        self._inject_channel_shader.use(
            texture, self._velocity.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            2, effective, mode=0,  # channel 2 = B component = W (Z-velocity)
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
        """Inject 2D RGBA texture into the split density+colour volumes.

        Brightness V = max(r,g,b) is injected additively into the R16F density
        volume (output resolution).  The full RGBA colour is injected additively
        into the RGBA16F colour volume (sim resolution).
        Both inputs are scaled by config.density.input_strength and delta_time.
        """
        if not self._allocated:
            return
        dt = 1.0 / max(1, self.config.fps)
        effective = strength * self.config.density.input_strength * dt
        # Inject brightness into high-res scalar density volume
        self._inject_value_shader.use(
            texture, self._density.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            effective, mode=0
        )
        # Inject RGBA colour into sim-res colour volume
        self._inject_shader.use(
            texture, self._color.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            effective, mode=0,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def set_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Replace 3D density+colour volumes from a 2D RGBA texture.

        Clears both volumes first, then injects brightness V into the R16F
        density volume and the full RGBA colour into the colour volume.
        """
        if not self._allocated:
            return
        self._density.clear_all()
        self._color.clear_all()
        self._inject_value_shader.use(
            texture, self._density.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            strength, mode=1
        )
        self._inject_shader.use(
            texture, self._color.texture,
            self.config.depth.injection_layer,
            self.config.depth.injection_spread,
            strength, mode=1,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_density_channel(self, texture: Texture, channel: int,
                            strength: float = 1.0) -> None:
        """Deprecated: density is no longer split into per-camera RGBA channels.

        Use add_density() instead — it injects RGBA colour directly.
        This method is kept for API compatibility but injects into the colour
        volume only (ignores the channel argument, treats the texture as RGBA).
        """
        self.add_density(texture, strength)

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

    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle volume with a 2D mask projected through all layers.

        Stores the 2D mask as the authoritative source, then projects into
        both sim-resolution and density-resolution 3D obstacle volumes.

        Args:
            texture: 2D obstacle mask (any channel > 0.5 = obstacle)
        """
        if not self._allocated:
            return
        # Replace 2D source (clear → OR with texture = texture)
        FlowUtil.zero(self._obstacle_source)
        self._obstacle_source.swap()
        self._obstacle_source.begin()
        self._add_boolean_shader.use(self._obstacle_source.back_texture, texture)
        self._obstacle_source.end()
        # Project to 3D volumes
        self._inject_binary_shader.use(self._obstacle_source.texture, self._simulation_obstacle, mode=1)
        self._inject_binary_shader.use(self._obstacle_source.texture, self._density_obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def add_obstacle(self, texture: Texture) -> None:
        """Add to obstacle volume (boolean OR with existing obstacles).

        Updates the 2D source and both 3D obstacle volumes.

        Args:
            texture: 2D obstacle mask to add (any channel > 0.5 = obstacle)
        """
        if not self._allocated:
            return
        # OR into 2D source
        self._obstacle_source.swap()
        self._obstacle_source.begin()
        self._add_boolean_shader.use(self._obstacle_source.back_texture, texture)
        self._obstacle_source.end()
        # Regenerate 3D volumes from updated source
        self._inject_binary_shader.use(self._obstacle_source.texture, self._simulation_obstacle, mode=1)
        self._inject_binary_shader.use(self._obstacle_source.texture, self._density_obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        if not self._allocated:
            return
        FlowUtil.zero(self._obstacle_source)
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
        """R16F scalar brightness volume at output resolution (GL_TEXTURE_3D)."""
        return self._density.texture

    @property
    def color_volume(self) -> Texture3D:
        """RGBA16F colour volume at sim resolution (GL_TEXTURE_3D)."""
        return self._color.texture

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
        """Current simulation resolution width (aligned to 32)."""
        return self._simulation_width

    @property
    def sim_height(self) -> int:
        """Current simulation resolution height (aligned to 32)."""
        return self._simulation_height
