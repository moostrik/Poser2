"""ParticleSurface - the particles as sphere sprites into a depth, a thickness and a color target."""
from __future__ import annotations

import ctypes

from OpenGL.GL import GL_DRAW_INDIRECT_BUFFER, GL_POINTS, glBindBuffer, glDrawArraysIndirect, glUniform1f, glUniform1ui, glUniform2f, glUseProgram

from modules.gl import Shader, StorageBuffer
from .particle_ring import DRAW_OFFSET


class ParticleSurface(Shader):
    """Splat the ring's particles as spheres; the bound framebuffer's three targets blend MIN, SUM, SUM."""

    def use(self, particles: StorageBuffer, ring: StorageBuffer, count: int, *, scale: float, ring_fade: float,
            output_size: tuple[int, int], z_per_pixel: float, radius: float, reference_mass: float, spread: float,
            fade_time: float, max_point_size: float) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        particles.bind_base(0)
        ring.bind_base(1)
        glUniform1ui(self.get_uniform_loc("uCount"), count)
        glUniform1f(self.get_uniform_loc("uScale"), scale)
        glUniform1f(self.get_uniform_loc("uRingFade"), ring_fade)
        glUniform2f(self.get_uniform_loc("uOutputSize"), float(output_size[0]), float(output_size[1]))
        glUniform1f(self.get_uniform_loc("uZPerPixel"), z_per_pixel)
        glUniform1f(self.get_uniform_loc("uRadius"), radius)
        glUniform1f(self.get_uniform_loc("uReferenceMass"), reference_mass)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uFadeTime"), fade_time)
        glUniform1f(self.get_uniform_loc("uMaxPointSize"), max_point_size)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ring.buffer_id)     # over the ring's filled slots
        glDrawArraysIndirect(GL_POINTS, ctypes.c_void_p(DRAW_OFFSET))
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)
