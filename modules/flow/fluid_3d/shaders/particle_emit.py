"""ParticleEmit - new particles from a density source into the ring."""
from __future__ import annotations

from OpenGL.GL import glUniform1f, glUniform1ui, glUseProgram

from modules.gl import ComputeShader, StorageBuffer, Texture


class ParticleEmit(ComputeShader):
    """Spawn particles per texel of a density source, their mass its amount, their color its color."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, particles: StorageBuffer, ring: StorageBuffer, count: int, *,
            gain: float, dt: float, rate: float, floor: float, slab: float, layer: float, spread: float,
            seed: int) -> None:
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
            slab: The injection gaussian's sum over the grid's layers
            layer: The slab's center in the volume's z, 0 to 1
            spread: Its sigma
            seed: The frame, for the hash
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
        glUniform1f(self.get_uniform_loc("uSlab"), slab)
        glUniform1f(self.get_uniform_loc("uLayer"), layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1ui(self.get_uniform_loc("uCount"), count)
        glUniform1ui(self.get_uniform_loc("uSeed"), seed & 0xFFFFFFFF)
        self.dispatch(source.width, source.height)
        self.barrier_buffer()
