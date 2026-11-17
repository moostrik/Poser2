"""VelocityPeak: the 99th percentile of a velocity's speed over about a second, counted on the GPU into log bins
and read back from two pixel buffers without waiting. How and why: BRIDGE_3D.md, "The peak"."""

from __future__ import annotations

import ctypes

import numpy as np
from OpenGL.GL import (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED, GL_NEAREST, GL_PIXEL_PACK_BUFFER, GL_R32UI,
                       GL_RED_INTEGER, GL_STREAM_READ, GL_SYNC_GPU_COMMANDS_COMPLETE, GL_TEXTURE_2D, GL_UNSIGNED_INT,
                       glBindBuffer, glBindTexture, glBufferData, glClearTexImage, glClientWaitSync, glDeleteBuffers,
                       glDeleteSync, glFenceSync, glGenBuffers, glGetBufferSubData)
from OpenGL.raw.GL.VERSION.GL_1_0 import glGetTexImage as raw_get_tex_image

from modules.gl import Texture
from modules.settings import BaseSettings, Field
from ..visualization import HistogramCount, histogram_grid

BINS: int = 256
SPEED_LOW: float = 0.01         # widths/s: the first bin's lower edge; below it is still
SPEED_HIGH: float = 10.0        # widths/s: the last bin's upper edge; above it counts in the last bin
PERCENTILE: float = 99.0
READ_INTERVAL: float = 1.0      # s between two reads of the bins
PIXEL_BUFFERS: int = 2


class VelocityPeakSettings(BaseSettings):
    """The peak: whether it is measured, and what it read."""
    enabled = Field(False, description="Measure the p99 speed once a second")
    speed = Field(0.0, access=Field.READ, description="p99 speed of the moving pixels, last second (widths/s); 10 = at least 10")


def speed_percentile(counts: np.ndarray, percentile: float = PERCENTILE, low: float = SPEED_LOW,
                     high: float = SPEED_HIGH) -> float:
    """The speed below which `percentile` % of the counted speeds lie, from the bins' `counts`: within its bin,
    log-interpolated by the share of the bin's count it needs; `high` when it falls in the last bin (at least
    `high`), 0 with nothing counted."""
    counts = np.asarray(counts, dtype=np.float64)
    total: float = float(counts.sum())
    if total <= 0.0:
        return 0.0
    bins: int = len(counts)
    target: float = total * percentile / 100.0
    cumulative: np.ndarray = np.cumsum(counts)
    bin_: int = int(np.searchsorted(cumulative, target, side='left'))
    if bin_ >= bins - 1:
        return high
    before: float = float(cumulative[bin_ - 1]) if bin_ > 0 else 0.0
    share: float = (target - before) / max(float(counts[bin_]), 1.0)
    return low * (high / low) ** ((bin_ + share) / bins)


class VelocityPeak:
    """A velocity's p99 speed over the moving pixels (`VelocityPeakSettings.speed`), while `enabled`; render thread
    only. `update` once per velocity: it counts the velocity's speeds into the bins, about once a second copies
    them into a pixel buffer behind a fence and clears them, and writes the percentile of a copy the GPU has
    finished; nothing waits. Off, nothing runs and `speed` is 0."""

    def __init__(self, settings: VelocityPeakSettings) -> None:
        self.settings: VelocityPeakSettings = settings
        self._bins: Texture = Texture(GL_NEAREST)       # an integer texture must not filter
        self._count: HistogramCount = HistogramCount()
        self._buffers: list[int] = []
        self._fences: list[object | None] = [None] * PIXEL_BUFFERS
        self._next: int = 0                             # the buffer the next copy goes into
        self._elapsed: float = 0.0                      # s the bins have counted
        self._zero: np.ndarray = np.zeros(1, dtype=np.uint32)

    @property
    def allocated(self) -> bool:
        return bool(self._buffers)

    def allocate(self) -> None:
        self._bins.allocate(BINS, 1, GL_R32UI)
        self._count.allocate()
        self._buffers = [int(b) for b in np.atleast_1d(glGenBuffers(PIXEL_BUFFERS))]
        for buffer in self._buffers:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer)
            glBufferData(GL_PIXEL_PACK_BUFFER, 4 * BINS, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._restart()

    def deallocate(self) -> None:
        self._release()
        if self._buffers:
            glDeleteBuffers(len(self._buffers), self._buffers)
        self._buffers = []
        self._bins.deallocate()
        self._count.deallocate()

    def update(self, velocity: Texture, delta_time: float) -> None:
        """Count `velocity`'s speeds (widths/s, xyz) over `delta_time` s; read what the GPU has finished."""
        if not self.allocated or not velocity.allocated:
            return
        if not self.settings.enabled:
            if self._elapsed > 0.0 or self.settings.speed != 0.0:
                self._restart()
                self.settings.speed = 0.0
            return
        self._collect()
        self._count.use(velocity, self._bins, 1.0, histogram_grid(velocity.width, velocity.height), (SPEED_LOW, SPEED_HIGH))
        self._elapsed += max(delta_time, 0.0)
        if self._elapsed >= READ_INTERVAL and self._fences[self._next] is None:
            self._copy(self._next)
            self._next = (self._next + 1) % PIXEL_BUFFERS
            self._elapsed = 0.0

    def _copy(self, slot: int) -> None:
        """The bins into pixel buffer `slot`, behind a fence; the bins cleared to count the next second."""
        glBindTexture(GL_TEXTURE_2D, self._bins.tex_id)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._buffers[slot])
        raw_get_tex_image(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._fences[slot] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        self._clear_bins()

    def _collect(self) -> None:
        """The percentile of each copy the GPU has finished, written to `speed`."""
        for slot, fence in enumerate(self._fences):
            if fence is None or glClientWaitSync(fence, 0, 0) not in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                continue
            glDeleteSync(fence)
            self._fences[slot] = None
            counts = np.zeros(BINS, dtype=np.uint32)
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self._buffers[slot])
            glGetBufferSubData(GL_PIXEL_PACK_BUFFER, 0, counts.nbytes, counts.ctypes.data_as(ctypes.c_void_p))
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
            self.settings.speed = round(speed_percentile(counts), 3)

    def _clear_bins(self) -> None:
        glClearTexImage(self._bins.tex_id, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, self._zero)

    def _restart(self) -> None:
        """Nothing counted, nothing waiting."""
        self._release()
        self._elapsed = 0.0
        if self._bins.allocated:
            self._clear_bins()

    def _release(self) -> None:
        for slot, fence in enumerate(self._fences):
            if fence is not None:
                glDeleteSync(fence)
            self._fences[slot] = None
