"""ParticleSprites - the particles drawn from their buffer as points, streaks or blobs into the depth slices."""
from __future__ import annotations

import ctypes

from OpenGL.GL import (GL_DRAW_INDIRECT_BUFFER, GL_LINES, GL_POINTS, glBindBuffer, glDrawArraysIndirect, glUniform1f, glUniform1i, glUniform1ui, glUniform2f,
                       glUseProgram)

from modules.gl import Shader, StorageBuffer
from .particle_ring import DRAW_OFFSET


class ParticleSprites(Shader):
    """Draw the ring's particles, pulled by gl_VertexID; the target is the bound framebuffer, summed (SUM blend)."""

    POINTS: int = 0
    STREAKS: int = 1
    BLOBS: int = 2

    def use(self, particles: StorageBuffer, ring: StorageBuffer, count: int, mode: int, *, slices: int,
            scale: float, ring_fade: float, output_size: tuple[int, int], size: float, streak: float,
            radius: float, reference_mass: float, spread: float, fade_time: float, max_point_size: float,
            fog: float, tint: float, weighted: bool = False, emission: float = 1.0, absorption: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        particles.bind_base(0)
        ring.bind_base(1)
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform1ui(self.get_uniform_loc("uCount"), count)
        glUniform1i(self.get_uniform_loc("uSlices"), slices)
        glUniform1f(self.get_uniform_loc("uScale"), scale)
        glUniform1f(self.get_uniform_loc("uRingFade"), ring_fade)
        glUniform2f(self.get_uniform_loc("uOutputSize"), float(output_size[0]), float(output_size[1]))
        glUniform1f(self.get_uniform_loc("uSize"), size)
        glUniform1f(self.get_uniform_loc("uStreak"), streak)
        glUniform1f(self.get_uniform_loc("uRadius"), radius)
        glUniform1f(self.get_uniform_loc("uReferenceMass"), reference_mass)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uFadeTime"), fade_time)
        glUniform1f(self.get_uniform_loc("uMaxPointSize"), max_point_size)
        glUniform1f(self.get_uniform_loc("uFog"), fog)
        glUniform1f(self.get_uniform_loc("uTint"), tint)
        glUniform1i(self.get_uniform_loc("uWeighted"), int(weighted))
        glUniform1f(self.get_uniform_loc("uEmission"), emission)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ring.buffer_id)     # over the ring's filled slots
        glDrawArraysIndirect(GL_LINES if mode == self.STREAKS else GL_POINTS, ctypes.c_void_p(DRAW_OFFSET))
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)
