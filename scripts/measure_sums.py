"""Measure Sums: one frame of one side of the depth velocity's measuring summed on the GPU (`measure_sums.comp`), for
`depth_velocity_compare.py`: the sums the row needs per region, read back as a few numbers, and the still part's
histograms, kept on the GPU over the loop and read back once."""

import math

import numpy as np
from OpenGL.GL import (GL_DYNAMIC_READ, GL_R8, GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_STORAGE_BUFFER,
                       GL_TEXTURE0, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, GL_RED,
                       glActiveTexture, glBindBuffer, glBindBufferBase, glBindTexture, glBufferData,
                       glDeleteBuffers, glDispatchCompute, glGenBuffers, glGetBufferSubData, glMemoryBarrier,
                       glTexImage2D, glUniform1f, glUniform1i, glUniform3f, glUseProgram)

from modules.gl import ComputeShader, Texture

SUMS: int = 12
BINS: int = 4096
BIN: float = 1.0 / 128.0           # px per frame a bin
HISTOGRAMS: tuple[str, ...] = ('xy inside', 'xy edge', 'z inside', 'z edge', 'z table')
CORE, EDGE, TABLE, MOVING, MASK = 1, 2, 4, 8, 16


def region_bits(regions: dict, moved: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The frame's regions as one byte a pixel, the shader's bits."""
    bits = np.zeros(mask.shape, np.uint8)
    bits |= regions['core'].astype(np.uint8) * CORE
    bits |= regions['edge'].astype(np.uint8) * EDGE
    bits |= regions['table'].astype(np.uint8) * TABLE
    bits |= (regions['core'] & moved).astype(np.uint8) * MOVING
    bits |= mask.astype(np.uint8) * MASK
    return bits


def p99_of(histogram: np.ndarray) -> float:
    """The 99th percentile of a histogram's values, the middle of its bin; NaN without values."""
    total = int(histogram.sum())
    if total == 0:
        return float('nan')
    index = int(np.searchsorted(np.cumsum(histogram), 0.99 * total))
    return (index + 0.5) * BIN


class RegionTexture(Texture):
    """The regions' bits (R8, the bits over 255), uploaded once a frame."""

    def upload(self, bits: np.ndarray) -> None:
        h, w = bits.shape
        if not self.allocated or self.width != w or self.height != h:
            self.allocate(w, h, GL_R8)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, np.ascontiguousarray(bits))
        glBindTexture(GL_TEXTURE_2D, 0)


class SideSums:
    """One side's buffers: the frame's partials, a workgroup's sums each, and the still part's histograms over the loop."""

    def __init__(self, width: int, height: int) -> None:
        self.groups: int = math.ceil(width / 16) * math.ceil(height / 16)
        self.partials, self.histogram_buffer = (int(name) for name in glGenBuffers(2))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.partials)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.groups * SUMS * 4, None, GL_DYNAMIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.histogram_buffer)
        glBufferData(GL_SHADER_STORAGE_BUFFER, len(HISTOGRAMS) * BINS * 4, np.zeros(len(HISTOGRAMS) * BINS, np.uint32), GL_DYNAMIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def histograms(self) -> dict[str, np.ndarray]:
        """The still part's histograms over the loop so far."""
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.histogram_buffer)
        raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, len(HISTOGRAMS) * BINS * 4)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        counts = np.frombuffer(raw, np.uint32).reshape(len(HISTOGRAMS), BINS)
        return {name: counts[i].copy() for i, name in enumerate(HISTOGRAMS)}

    def deallocate(self) -> None:
        glDeleteBuffers(2, [self.partials, self.histogram_buffer])


class MeasureSums(ComputeShader):
    """The sums of one side's frame into its `SideSums` (one program for every side)."""

    def use(self, buffers: SideSums, velocity: Texture, estimate: Texture, before: Texture, z_position: Texture | None,
            regions: Texture, to_px: tuple[float, float, float], width: float, still: bool) -> np.ndarray:
        """Sum one frame; the frame's sums (SUMS), the workgroups' added."""
        glUseProgram(self.shader_program)
        for unit, texture in enumerate((velocity, estimate, before, z_position, regions)):
            glActiveTexture(int(GL_TEXTURE0) + unit)
            glBindTexture(GL_TEXTURE_2D, texture.tex_id if texture is not None else 0)
        glUniform3f(self.get_uniform_loc('to_px'), *to_px)
        glUniform1f(self.get_uniform_loc('width'), width)
        glUniform1i(self.get_uniform_loc('still'), int(still))
        glUniform1i(self.get_uniform_loc('weighed'), int(z_position is not None))
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.partials)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.histogram_buffer)
        glDispatchCompute(math.ceil(velocity.width / 16), math.ceil(velocity.height / 16), 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers.partials)
        raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, buffers.groups * SUMS * 4)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return np.frombuffer(raw, np.float32).reshape(buffers.groups, SUMS).astype(np.float64).sum(axis=0)
