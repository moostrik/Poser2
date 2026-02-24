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

from modules.gl import Texture, SwapFbo, Fbo
from modules.gl.Texture3D import Texture3D, SwapTexture3D
from modules.gl.ComputeShader import ComputeShader
from modules.settings import Field, Settings
from .shaders import (
    Advect3D, Divergence3D, Gradient3D,
    JacobiPressure3D, JacobiDiffusion3D,
    VorticityCurl3D, VorticityForce3D, Buoyancy3D,
    Inject3D, InjectChannel3D, Clamp3D, Composite3D, Add3D
)

# Combined memory barrier bits (cast to int for Pylance compatibility)
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FluidFlow3DConfig(Settings):
    """Configuration for 3D fluid simulation.

    Same speed/lifetime/vorticity/buoyancy model as FluidFlowConfig,
    with additional depth-specific parameters.
    """

    # ---- Depth parameters ----
    depth_layers = Field(16, min=4, max=64, description="Number of depth layers in the 3D volume")
    depth_scale = Field(1.0, min=0.1, max=4.0, description="Z grid spacing relative to XY")
    composite_mode = Field(1, min=0, max=2, description="3D->2D compositing: 0=alpha, 1=additive, 2=max")
    injection_layer = Field(0.5, min=0.0, max=1.0, description="Normalized depth for 2D->3D injection center")
    injection_spread = Field(0.15, min=0.01, max=0.5, description="Gaussian sigma for depth spread during injection")

    # ---- Transport speed ----
    speed = Field(1.0, min=0.0, max=5.0, description="Base fluid transport rate")
    vel_self_advection = Field(0.01, min=0.0, max=0.2, description="How much velocity advects itself")
    den_speed_offset = Field(0.0, min=-5.0, max=5.0, description="Added to base speed for density only")

    # ---- Velocity parameters ----
    vel_lifetime = Field(10.0, min=0.01, max=60.0, description="Seconds until velocity fades to ~1%")
    vel_vorticity = Field(5.0, min=0.0, max=60.0, description="Vortex confinement strength")
    vel_vorticity_radius = Field(3.0, min=1.0, max=30.0, description="Curl sampling radius in texels")
    vel_viscosity = Field(15.0, min=0.0, max=100.0, description="Fluid thickness/resistance to flow")
    vel_viscosity_iter = Field(40, min=1, max=60, description="Solver iterations for viscosity")

    # ---- Pressure parameters ----
    prs_speed = Field(0.0, min=0.0, max=2.0, description="Pressure advection speed")
    prs_lifetime = Field(8.0, min=0.01, max=60.0, description="Seconds until pressure fades to ~1%")
    prs_iterations = Field(40, min=1, max=60, description="Solver iterations for pressure")

    # ---- Density parameters ----
    den_lifetime = Field(30.0, min=0.01, max=60.0, description="Seconds until density fades to ~1%")

    # ---- Temperature parameters ----
    tmp_lifetime = Field(3.0, min=0.01, max=60.0, description="Seconds until temperature fades to ~1%")
    tmp_buoyancy = Field(0.0, min=0.0, max=10.0, description="Thermal buoyancy coefficient: hot air rises")
    tmp_weight = Field(-10.0, min=-20.0, max=2.0, description="Ratio of gravity/settling vs thermal lift")
    tmp_ambient = Field(0.2, min=0.0, max=1.0, description="Reference temperature (buoyancy = 0 at this temp)")


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

    def __init__(self, sim_scale: float = 0.25,
                 config: FluidFlow3DConfig | None = None) -> None:
        self.config: FluidFlow3DConfig = config or FluidFlow3DConfig()

        self._simulation_scale: float = sim_scale
        self._width: int = 0
        self._height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0
        self._depth: int = 0
        self._aspect: float = 1.0
        self._allocated: bool = False

        # ---- Volumetric fields (SwapTexture3D for ping-pong) ----
        # Velocity: CLAMP_TO_BORDER(0) = no-slip walls
        self._velocity: SwapTexture3D = SwapTexture3D(
            wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0)
        )
        # Density: CLAMP_TO_BORDER(0) = nothing leaks out
        self._density: SwapTexture3D = SwapTexture3D(
            wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0)
        )
        # Temperature: CLAMP_TO_EDGE = insulated walls (Neumann)
        self._temperature: SwapTexture3D = SwapTexture3D(
            wrap=GL_CLAMP_TO_EDGE
        )
        # Pressure: CLAMP_TO_EDGE = zero-gradient walls (Neumann)
        self._pressure: SwapTexture3D = SwapTexture3D(
            wrap=GL_CLAMP_TO_EDGE
        )

        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._obstacle: Texture3D = Texture3D(
            interpolation=GL_NEAREST,
            wrap=GL_CLAMP_TO_BORDER,
            border_color=(1.0, 0.0, 0.0, 0.0)
        )

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
        self._composite_shader: Composite3D = Composite3D()
        self._add_shader: Add3D = Add3D()

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
        """R8 obstacle mask volume."""
        return self._obstacle

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

    # ========== Allocation ==========

    def allocate(self, width: int, height: int,
                 output_width: int | None = None,
                 output_height: int | None = None) -> None:
        """Allocate all 3D volumes and shaders.

        Args:
            width: Simulation resolution (XY)
            height: Simulation resolution (XY)
            output_width: Composited 2D output width (defaults to width)
            output_height: Composited 2D output height (defaults to height)
        """
        self._width = width
        self._height = height
        self._density_width = output_width if output_width is not None else width
        self._density_height = output_height if output_height is not None else height
        self._aspect = width / height if height > 0 else 1.0
        self._depth = self.config.depth_layers

        sim_w = self._width
        sim_h = self._height
        d = self._depth

        # Allocate volumetric fields (at incoming simulation resolution)
        self._velocity.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._velocity.clear_all()

        self._density.allocate(self._density_width, self._density_height, d, GL_RGBA16F)
        self._density.clear_all()

        self._temperature.allocate(sim_w, sim_h, d, GL_R16F)
        self._temperature.clear_all()

        self._pressure.allocate(sim_w, sim_h, d, GL_R16F)
        self._pressure.clear_all()

        self._obstacle.allocate(sim_w, sim_h, d, GL_R8)
        self._obstacle.clear()

        # Intermediate volumes
        self._divergence_vol.allocate(sim_w, sim_h, d, GL_R16F)
        self._divergence_vol.clear()

        self._curl_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._curl_vol.clear()

        self._vorticity_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._vorticity_force_vol.clear()

        self._buoyancy_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._buoyancy_force_vol.clear()

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
        self._composite_shader.allocate()
        self._add_shader.allocate()

        self._allocated = True

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._velocity.deallocate()
        self._density.deallocate()
        self._temperature.deallocate()
        self._pressure.deallocate()
        self._obstacle.deallocate()
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
        self._composite_shader.deallocate()
        self._add_shader.deallocate()

        self._allocated = False

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

    # ========== Input Methods (2D -> 3D injection) ==========

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D velocity texture into 3D velocity volume.

        Uses gaussian depth spread centered at config.injection_layer.
        """
        if not self._allocated:
            return
        self._inject_shader.use(
            texture, self._velocity.texture,
            self.config.injection_layer,
            self.config.injection_spread,
            strength, mode=0,
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
            self.config.injection_layer,
            self.config.injection_spread,
            strength, mode=1,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D density texture into 3D density volume."""
        if not self._allocated:
            return
        self._inject_shader.use(
            texture, self._density.texture,
            self.config.injection_layer,
            self.config.injection_spread,
            strength, mode=0,
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
            self.config.injection_layer,
            self.config.injection_spread,
            strength, mode=1,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_density_channel(self, texture: Texture, channel: int,
                            strength: float = 1.0) -> None:
        """Inject single-channel 2D texture into one RGBA channel of the 3D density volume.

        Uses gaussian depth spread centered at config.injection_layer.

        Args:
            texture: 2D source texture (reads .r component)
            channel: Target RGBA channel (0=R, 1=G, 2=B, 3=A)
            strength: Injection strength multiplier
        """
        if not self._allocated:
            return
        self._inject_channel_shader.use(
            texture, self._density.texture,
            self.config.injection_layer,
            self.config.injection_spread,
            channel, strength, mode=0,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def clamp_density(self, min_value: float = 0.0, max_value: float = 1.0) -> None:
        """Clamp 3D density volume voxels to [min_value, max_value].

        Args:
            min_value: Minimum clamp value (applied to all channels)
            max_value: Maximum clamp value (applied to all channels)
        """
        if not self._allocated:
            return
        self._clamp_shader.use(
            self._density.texture, min_value, max_value,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D temperature texture into 3D temperature volume."""
        if not self._allocated:
            return
        self._inject_shader.use(
            texture, self._temperature.texture,
            self.config.injection_layer,
            self.config.injection_spread,
            strength, mode=0,
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
            self.config.injection_layer,
            self.config.injection_spread,
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
            self.config.injection_layer,
            self.config.injection_spread,
            strength, mode=0,
            internal_format=GL_R16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    # ========== Internal helpers ==========

    def _add_force_to_velocity(self, force: Texture3D, strength: float = 1.0) -> None:
        """Add 3D force volume to velocity in-place (no swap needed)."""
        self._add_shader.use(self._velocity.texture, force, strength)
        glMemoryBarrier(_BARRIER_IMAGE)

    @staticmethod
    def _calculate_dissipation(delta_time: float, decay_time: float) -> float:
        """Calculate frame-rate independent decay multiplier.

        Returns pow(0.01, dt / decay_time) -- field reaches 1% after decay_time seconds.
        """
        return pow(0.01, delta_time / max(0.001, decay_time))

    # ========== Update Pipeline ==========

    def update(self, delta_time: float = 1.0) -> None:
        """Run one frame of the 3D fluid simulation pipeline.

        Args:
            delta_time: Time step (typically 1/fps or 1.0 for per-frame)
        """
        if not self._allocated:
            return

        self._aspect = self._width / self._height if self._height > 0 else 1.0
        depth_scale: float = self.config.depth_scale
        grid_scale: float = self._simulation_scale

        # ===== STEP 1: ADVECT DENSITY =====
        advect_den_step = delta_time * (self.config.speed + self.config.den_speed_offset)
        dissipate_den = self._calculate_dissipation(delta_time, self.config.den_lifetime)

        self._density.swap()
        self._advect_shader.advect(
            self._density.back_texture,   # read from previous
            self._density.texture,        # write to current
            self._velocity.texture,       # advected by velocity
            self._obstacle,
            self._aspect, depth_scale,
            advect_den_step, dissipate_den,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # ===== STEP 2: ADVECT VELOCITY (self-advection) =====
        advect_vel_step = delta_time * self.config.vel_self_advection
        dissipate_vel = self._calculate_dissipation(delta_time, self.config.vel_lifetime)

        self._velocity.swap()
        self._advect_shader.advect(
            self._velocity.back_texture,  # source velocity (self-advection)
            self._velocity.texture,       # write result
            self._velocity.back_texture,  # advected by same velocity field
            self._obstacle,
            self._aspect, depth_scale,
            advect_vel_step, dissipate_vel,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # ===== STEP 3: VELOCITY DIFFUSE (viscosity) =====
        if self.config.vel_viscosity > 0.0:
            viscosity_dt = self.config.vel_viscosity * (self._simulation_scale ** 2) * delta_time

            result = self._jacobi_diffusion_shader.solve(
                self._velocity.texture,
                self._velocity.back_texture,
                self._obstacle,
                grid_scale, self._aspect, depth_scale,
                viscosity_dt,
                total_iterations=self.config.vel_viscosity_iter,
                iterations_per_dispatch=5
            )
            # Ensure correct buffer is active after solve
            if result != self._velocity.texture:
                self._velocity.swap()
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # ===== STEP 4: VORTICITY CONFINEMENT =====
        if self.config.vel_vorticity > 0.0 and self.config.vel_vorticity_radius > 0.0:
            vorticity_radius = self.config.vel_vorticity_radius * self._simulation_scale
            vorticity_force = self.config.vel_vorticity * delta_time

            # 4a. Compute 3D curl (vorticity vector)
            self._vorticity_curl_shader.use(
                self._velocity.texture,
                self._obstacle,
                self._curl_vol,
                grid_scale, self._aspect, depth_scale,
                vorticity_radius
            )
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

            # 4b. Compute confinement force
            self._vorticity_force_shader.use(
                self._curl_vol,
                self._vorticity_force_vol,
                grid_scale, self._aspect, depth_scale,
                vorticity_force
            )
            glMemoryBarrier(_BARRIER_IMAGE)

            # 4c. Add force to velocity in-place
            self._add_force_to_velocity(self._vorticity_force_vol)

        # ===== STEP 5 & 6: TEMPERATURE ADVECT & BUOYANCY =====
        if self.config.tmp_buoyancy == 0.0:
            self._temperature.clear_all()
        else:
            # 5a. Advect temperature
            advect_tmp_step = delta_time * self.config.speed
            dissipate_tmp = self._calculate_dissipation(delta_time, self.config.tmp_lifetime)

            self._temperature.swap()
            self._advect_shader.advect(
                self._temperature.back_texture,
                self._temperature.texture,
                self._velocity.texture,
                self._obstacle,
                self._aspect, depth_scale,
                advect_tmp_step, dissipate_tmp,
                internal_format=GL_R16F
            )
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

            # 6. Compute and apply buoyancy force
            # F = sigma*(T - T_ambient) - kappa*density
            sigma = delta_time * self._simulation_scale * self.config.tmp_buoyancy
            kappa = delta_time * self._simulation_scale * self.config.tmp_weight

            self._buoyancy_shader.use(
                self._temperature.texture,
                self._density.texture,
                self._buoyancy_force_vol,
                sigma, kappa, self.config.tmp_ambient
            )
            glMemoryBarrier(_BARRIER_IMAGE)

            # Add buoyancy force to velocity
            self._add_force_to_velocity(self._buoyancy_force_vol)

        # ===== STEP 7: PRESSURE ADVECT (optional, non-physical) =====
        if self.config.prs_speed > 0.0:
            advect_prs_step = delta_time * self.config.prs_speed
            dissipate_prs = self._calculate_dissipation(delta_time, self.config.prs_lifetime)

            self._pressure.swap()
            self._advect_shader.advect(
                self._pressure.back_texture,
                self._pressure.texture,
                self._velocity.texture,
                self._obstacle,
                self._aspect, depth_scale,
                advect_prs_step, dissipate_prs,
                internal_format=GL_R16F
            )
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # ===== STEP 8: PRESSURE PROJECTION (make divergence-free) =====

        # 8a. Compute divergence
        self._divergence_shader.use(
            self._velocity.texture,
            self._obstacle,
            self._divergence_vol,
            grid_scale, self._aspect, depth_scale
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # 8b. Solve Poisson equation for pressure (Jacobi iterations)
        result = self._jacobi_pressure_shader.solve(
            self._pressure.texture,
            self._pressure.back_texture,
            self._divergence_vol,
            self._obstacle,
            grid_scale, self._aspect, depth_scale,
            total_iterations=self.config.prs_iterations,
            iterations_per_dispatch=5
        )
        # Ensure correct buffer is active
        if result != self._pressure.texture:
            self._pressure.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # 8c. Subtract pressure gradient from velocity
        self._velocity.swap()
        self._gradient_shader.use(
            self._velocity.back_texture,    # read old velocity
            self._pressure.texture,          # read pressure
            self._obstacle,
            self._velocity.texture,          # write corrected velocity
            grid_scale, self._aspect, depth_scale
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # ===== STEP 9: COMPOSITE 3D -> 2D =====
        self._composite_shader.use(
            self._density.texture,
            self._output_texture,
            self.config.composite_mode
        )
        glMemoryBarrier(_BARRIER_IMAGE)
