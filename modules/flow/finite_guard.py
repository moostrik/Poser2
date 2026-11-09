"""FiniteGuard: a stage's count of the non-finite values its shaders replaced with 0, and its warning.

A shader that writes a stage's field tests each value's bits: a float whose exponent bits are all ones is an
infinity or a NaN, whatever the rest, a test the driver cannot drop the way it may drop `isnan`. Such a value
becomes 0 and adds 1 to its field's slot in the guard's buffer (`atomicAdd`), so a bad value stops where it was
made and the show keeps running. The guard reads the counts about once a second, from the other of its two
buffers once a fence says the GPU is done with it, so nothing waits; a count above 0 is logged as a warning
naming the stage and the field, so nothing fails silently.

The shader side, in each guarded shader:

    layout(std430, binding = 7) buffer FiniteGuard { uint uNonFinite[]; };
    uniform int uGuardSlot;     // the field's slot; -1 no guard

    vec4 guard(vec4 value) {
        if (uGuardSlot < 0) return value;
        if (any(equal(floatBitsToUint(value) & 0x7F800000u, uvec4(0x7F800000u)))) {
            atomicAdd(uNonFinite[uGuardSlot], 1u);
            return vec4(0.0);
        }
        return value;
    }
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from OpenGL.GL import (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED, GL_SYNC_GPU_COMMANDS_COMPLETE, glClientWaitSync,
                       glDeleteSync, glFenceSync, glUniform1i)

from modules.gl import StorageBuffer
from modules.settings import BaseSettings, Field

logger = logging.getLogger(__name__)

GUARD_BINDING: int = 7          # the storage block binding the guarded shaders count into
REPORT_INTERVAL: float = 1.0    # s between two reads of the counts


def guard_report(stage: str, fields: tuple[str, ...], counts: list[int], period: float) -> list[str]:
    """The warnings for one read: one per field whose count is above 0."""
    return [f"{stage}: {count} non-finite {field} values set to 0 in the last {period:.1f} s"
            for field, count in zip(fields, counts) if count > 0]


class FiniteGuardSettings(BaseSettings):
    """A stage's guard, its settings group `guard`."""
    enabled: Field[bool] = Field(True, description="Set non-finite values to 0, count and log them; off once the stage is stable")


@dataclass(frozen=True)
class GuardSlot:
    """One field's place in a stage's guard, handed to the shader that writes the field."""
    guard: FiniteGuard
    index: int

    def use(self, slot_location: int) -> None:
        """Bind the guard's buffer to GUARD_BINDING and set the shader's `uGuardSlot` at `slot_location`; call
        with the shader's program in use."""
        self.guard.bind()
        glUniform1i(slot_location, self.index)


class FiniteGuard:
    """The non-finite counts of one stage's fields; render thread only. `update` once per tick."""

    def __init__(self, stage: str, fields: tuple[str, ...]) -> None:
        self._stage: str = stage
        self._fields: tuple[str, ...] = fields
        self._buffers: tuple[StorageBuffer, StorageBuffer] = (StorageBuffer(), StorageBuffer())
        self._active: int = 0                   # the buffer the shaders count into
        self._fence: object | None = None       # set when the other buffer waits to be read
        self._elapsed: float = 0.0              # s since the last swap
        self._period: float = 0.0               # s the buffer waiting to be read counted over

    def slot(self, field: str) -> GuardSlot:
        return GuardSlot(self, self._fields.index(field))

    @property
    def allocated(self) -> bool:
        return self._buffers[0].allocated

    def allocate(self) -> None:
        for buffer in self._buffers:
            buffer.allocate(4 * len(self._fields))
        self._active = 0
        self._elapsed = 0.0

    def deallocate(self) -> None:
        self._release_fence()
        for buffer in self._buffers:
            buffer.deallocate()

    def bind(self) -> None:
        self._buffers[self._active].bind_base(GUARD_BINDING)

    def update(self, delta_time: float) -> None:
        """Read the counts the GPU has finished, and about once a second swap the buffers to read the next."""
        if not self.allocated:
            return
        self._collect()
        self._elapsed += max(delta_time, 0.0)
        if self._fence is None and self._elapsed >= REPORT_INTERVAL:
            self._fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            self._active = 1 - self._active
            self._buffers[self._active].clear()
            self._period, self._elapsed = self._elapsed, 0.0

    def _collect(self) -> None:
        if self._fence is None:
            return
        if glClientWaitSync(self._fence, 0, 0) not in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
            return
        self._release_fence()
        waiting: StorageBuffer = self._buffers[1 - self._active]
        counts: list[int] = [waiting.read_uint(4 * index) for index in range(len(self._fields))]
        for message in guard_report(self._stage, self._fields, counts, self._period):
            logger.warning(message)

    def _release_fence(self) -> None:
        if self._fence is not None:
            glDeleteSync(self._fence)
        self._fence = None
