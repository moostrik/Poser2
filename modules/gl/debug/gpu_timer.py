import ctypes

from OpenGL.GL import (GL_QUERY_RESULT, GL_QUERY_RESULT_AVAILABLE, GL_TIMESTAMP, glDeleteQueries, glGenQueries,
                       glGetQueryObjectiv, glQueryCounter)
from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v as raw_get_query_ui64     # the wrapped one has no 64-bit output type

from modules.utils import PerformanceTimer

QUERY_COUNT: int = 8    # samples that can wait for the GPU; with all of them waiting the next one is skipped


class GpuTimer:
    """The GPU time of a block of GL work, as samples of a `PerformanceTimer` of that name.

    A block is two GPU timestamps, so blocks nest: a stage's timer can hold its passes' timers.
    Runs only during a timing session or while timer samples have a listener; otherwise `begin` and
    `end` touch no GL. The results are read a few frames later, when the GPU has them, so nothing
    waits. Must be called from the GL thread.
    """

    def __init__(self, name: str) -> None:
        self._timer: PerformanceTimer = PerformanceTimer(name=name, sample_count=200)
        self._queries: list[int] = []      # pairs: begin, end
        self._next: int = 0         # the pair the next block takes
        self._waiting: int = 0      # blocks ended, result not read yet
        self._open: bool = False

    def begin(self) -> None:
        if not PerformanceTimer.session_active() and not PerformanceTimer.has_sample_callbacks():
            return
        if not self._queries:
            self._queries = [int(query) for query in glGenQueries(2 * QUERY_COUNT)]
        self._collect()
        if self._waiting == QUERY_COUNT:
            return
        glQueryCounter(self._queries[2 * self._next], GL_TIMESTAMP)
        self._open = True

    def end(self) -> None:
        if not self._open:
            return
        self._open = False
        glQueryCounter(self._queries[2 * self._next + 1], GL_TIMESTAMP)
        self._next = (self._next + 1) % QUERY_COUNT
        self._waiting += 1

    def deallocate(self) -> None:
        if self._queries:
            glDeleteQueries(2 * QUERY_COUNT, self._queries)
        self._queries = []
        self._next = 0
        self._waiting = 0

    def _collect(self) -> None:
        """Read the results the GPU has ready, oldest first."""
        while self._waiting > 0:
            pair: int = (self._next - self._waiting) % QUERY_COUNT
            began, ended = self._queries[2 * pair], self._queries[2 * pair + 1]
            if not glGetQueryObjectiv(ended, GL_QUERY_RESULT_AVAILABLE):
                return
            nanoseconds: int = _timestamp(ended) - _timestamp(began)
            self._timer.add_time(nanoseconds / 1_000_000.0, report=False)
            self._waiting -= 1


def _timestamp(query: int) -> int:
    """A timestamp query's result, ns."""
    value = ctypes.c_uint64(0)
    raw_get_query_ui64(query, GL_QUERY_RESULT, ctypes.byref(value))
    return int(value.value)
