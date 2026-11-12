"""ParticleEmit - new particles from a density source into the ring."""
from __future__ import annotations

from OpenGL.GL import glUniform1f, glUniform1i, glUniform1ui, glUseProgram

from modules.gl import ComputeShader, StorageBuffer, Texture
from .z_position_binding import bind_z_position


class ParticleEmit(ComputeShader):
    """Spawn particles per texel of a density source, their mass its amount, their color its color."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, particles: StorageBuffer, ring: StorageBuffer, count: int, *,
            gain: float, dt: float, rate: float, floor: float, layers: int, spread: float,
            seed: int, z_position: Texture | None = None) -> None:
        """Emit from `source` into the ring of `count` particles.

        Args:
            source: RGBA, the amount times the color, the amount as alpha
            particles: The particles' buffer
            ring: The ring's head (one uint)
            count: The ring's size
            gain: The source's gain × the injection's weight
            dt: The step (s)
            rate: Particles per texel per second where ink is
            floor: The amount below which a texel spawns fewer
            layers: The grid's layers, which place the center and sum the slab for the mass
            spread: Its sigma
            seed: The frame, for the hash
            z_position: The source's z position (red over green); a texel without one, or every texel
                without it, spawns none
        """
        if not self.allocated or not self.shader_program or not source.allocated:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, source, "uSource")
        particles.bind_base(0)
        ring.bind_base(1)
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        glUniform1f(self.get_uniform_loc("uDt"), dt)
        glUniform1f(self.get_uniform_loc("uRate"), rate)
        glUniform1f(self.get_uniform_loc("uFloor"), floor)
        bind_z_position(self, 1, z_position)
        glUniform1i(self.get_uniform_loc("uLayers"), layers)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1ui(self.get_uniform_loc("uCount"), count)
        glUniform1ui(self.get_uniform_loc("uSeed"), seed & 0xFFFFFFFF)
        self.dispatch(source.width, source.height)
        self.barrier_buffer()
