from OpenGL.GL import (GL_QUERY_RESULT, GL_QUERY_RESULT_AVAILABLE, GL_TIME_ELAPSED, glBeginQuery, glDeleteQueries,
                       glEndQuery, glGenQueries, glGetQueryObjectiv, glGetQueryObjectuiv)

from modules.utils import PerformanceTimer

QUERY_COUNT: int = 8    # samples that can wait for the GPU; with all of them waiting the next one is skipped


class GpuTimer:
    """The GPU time of a block of GL work, as samples of a `PerformanceTimer` of that name.

    Runs only during a timing session; outside one, `begin` and `end` touch no GL. The results of timer
    queries are read a few frames later, when the GPU has them, so nothing waits. Only one block can be
    open at a time. Must be called from the GL thread.
    """

    def __init__(self, name: str) -> None:
        self._timer: PerformanceTimer = PerformanceTimer(name=name, sample_count=200)
        self._queries: list[int] = []
        self._next: int = 0         # the query the next block takes
        self._waiting: int = 0      # blocks ended, result not read yet
        self._open: bool = False

    def begin(self) -> None:
        if not PerformanceTimer.session_active():
            return
        if not self._queries:
            self._queries = [int(query) for query in glGenQueries(QUERY_COUNT)]
        self._collect()
        if self._waiting == QUERY_COUNT:
            return
        glBeginQuery(GL_TIME_ELAPSED, self._queries[self._next])
        self._open = True

    def end(self) -> None:
        if not self._open:
            return
        self._open = False
        glEndQuery(GL_TIME_ELAPSED)
        self._next = (self._next + 1) % QUERY_COUNT
        self._waiting += 1

    def deallocate(self) -> None:
        if self._queries:
            glDeleteQueries(QUERY_COUNT, self._queries)
        self._queries = []
        self._next = 0
        self._waiting = 0

    def _collect(self) -> None:
        """Read the results the GPU has ready, oldest first."""
        while self._waiting > 0:
            query: int = self._queries[(self._next - self._waiting) % QUERY_COUNT]
            if not glGetQueryObjectiv(query, GL_QUERY_RESULT_AVAILABLE):
                return
            nanoseconds: int = glGetQueryObjectuiv(query, GL_QUERY_RESULT)  # 32 bits hold 4 s, far past a frame
            self._timer.add_time(nanoseconds / 1_000_000.0, report=False)
            self._waiting -= 1
