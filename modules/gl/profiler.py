import time
import logging
from collections.abc import Callable

from OpenGL.GL import glFinish  # type: ignore

logger = logging.getLogger(__name__)


class Profiler:
    """Named-block interval timer with GL pipeline sync.

    Measures wall-clock time around named sections of work. On each end()
    call, glFinish() is issued once to drain the GL pipeline before recording
    the timestamp — giving serialized per-section timing without the redundant
    double-sync of calling glFinish in both begin() and end().

    Usage::

        profiler = Profiler(lambda: "MyClass 512x288", names=["stage_a", "stage_b"])
        profiler.enabled = True

        # in update():
        profiler.begin("stage_a")
        do_work_a()
        profiler.end("stage_a")

        profiler.begin("stage_b")
        do_work_b()
        profiler.end("stage_b")

        profiler.report()  # prints every `interval` frames, then resets

    Must be called from the GL thread.
    """

    def __init__(self, label_fn: Callable[[], str], names: list[str], interval: int = 120) -> None:
        self.enabled: bool = False
        self._label_fn = label_fn
        self._names = names
        self._interval = interval
        self._frame_count: int = 0
        self._accum: dict[str, float] = {}

    def begin(self, name: str) -> None:
        """Record start timestamp for a named section. No GL sync."""
        if not self.enabled:
            return
        self._accum[name + "_t0"] = time.perf_counter()

    def end(self, name: str) -> None:
        """Drain the GL pipeline, then accumulate elapsed ms for a named section."""
        if not self.enabled:
            return
        glFinish()
        t0 = self._accum.get(name + "_t0", 0.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._accum[name] = self._accum.get(name, 0.0) + elapsed_ms

    def report(self) -> None:
        """Increment frame counter; print averaged timings every interval frames."""
        if not self.enabled:
            return
        self._frame_count += 1
        if self._frame_count < self._interval:
            return

        n = self._frame_count
        total = 0.0
        lines = [f"{self._label_fn()} — avg over {n} frames:"]
        for name in self._names:
            avg_ms = self._accum.get(name, 0.0) / n
            total += avg_ms
            bar = "█" * int(avg_ms * 2)  # 1 block per 0.5 ms
            lines.append(f"  {name:<14s} {avg_ms:6.2f} ms  {bar}")
        lines.append(f"  {'TOTAL':<14s} {total:6.2f} ms  ({1000.0 / max(total, 0.001):.0f} fps budget)")
        logger.debug("\n".join(lines))

        self._frame_count = 0
        for key in list(self._accum.keys()):
            if not key.endswith("_t0"):
                self._accum[key] = 0.0
