"""AdvectComposite3D - Fused density advection + colour lookup + 2D compositing."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Texture3D, ComputeShader


class AdvectComposite3D(ComputeShader):
    """Fused density advection + colour lookup + 2D compositing.

    Replaces the separate Advect3D (density) + Composite3D dispatches with
    one 2D pass.  For each output pixel the shader iterates over every depth
    layer, inline-advects the R16F density, writes the result back to the
    front buffer (for buoyancy / injection next frame), samples the
    already-advected colour volume, and accumulates the 2D composite.

    Modes:
        0 = Front-to-back alpha compositing
        1 = Additive blending
        2 = Maximum intensity projection
        3 = Emission-absorption (Beer's law volumetric)
        4 = Debug depth spectrum
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(
        self,
        velocity: Texture3D,
        density_back: Texture3D,
        density_front: Texture3D,
        color: Texture3D,
        obstacle: Texture3D,
        output_2d: Texture,
        timestep: float,
        aspect: float,
        depth_scale: float,
        dissipation: float,
        mode: int = 0,
        absorption: float = 4.0,
        has_obstacles: bool = True,
    ) -> None:
        """Run fused advect-density + composite pass.

        Args:
            velocity:      3D velocity field (RGBA16F, sim resolution).
            density_back:  Previous-frame density (R16F, output resolution) — read via sampler.
            density_front: Destination density (R16F, output resolution) — written via imageStore.
            color:         Already-advected colour (RGBA16F, sim resolution).
            obstacle:      Density obstacle mask (R8, output resolution).
            output_2d:     2D composited output (RGBA16F).
            timestep:      dt * speed — advection distance per frame.
            aspect:        Width / height ratio for isotropic XY advection.
            depth_scale:   Z grid spacing relative to XY.
            dissipation:   Exponential decay multiplier.
            mode:          Composite mode (0–4, see class docstring).
            absorption:    Absorption coefficient for Beer's law (mode 3).
            has_obstacles:  Whether obstacle logic is active.
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # ---- Samplers (read-only, trilinear) ----
        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, density_back, "uDensityBack")
        self.bind_texture_3d(2, color, "uColor")
        self.bind_texture_3d(3, obstacle, "uObstacle")

        # ---- Images (write-only) ----
        self.bind_image_3d_write(0, density_front, GL_R16F)
        self.bind_image_write(1, output_2d, GL_RGBA16F)

        # ---- Advection uniforms ----
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        rdx_y = 1.0 / aspect if aspect > 0.0 else 1.0
        rdx_z = 1.0 / depth_scale if depth_scale > 0.0 else 1.0
        glUniform3f(self.get_uniform_loc("uRdx"), 1.0, rdx_y, rdx_z)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        # ---- Composite uniforms ----
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform1i(self.get_uniform_loc("uDepth"), density_back.depth)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output_2d.width, output_2d.height)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)

        # ---- Dispatch 2D (one thread per output pixel) ----
        self.dispatch(output_2d.width, output_2d.height)
