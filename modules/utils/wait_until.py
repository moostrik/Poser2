"""wait_until: wait for a `perf_counter` deadline to the millisecond, waking early when an event is set or a condition
notified.

On Windows a timed `Event.wait` or `Condition.wait` returns on the system timer's ticks, every 15.6 ms: a 1 ms wait
takes 15.6 ms and an 18 ms one 31 ms, while `time.sleep` comes within half a millisecond (measured 2026-11-20). So
the wait that can be woken runs only while more than one tick is left, and `time.sleep` takes the rest."""

import time
from collections.abc import Callable
from typing import Protocol

WAKE_STEP: float = 0.0175   # s: the most a timed Event or Condition wait returns late (one 15.6 ms tick), and a margin


class Releasable(Protocol):
    """A lock or a condition, released around the last stretch's sleep."""
    def acquire(self, blocking: bool = ..., timeout: float = ...) -> bool: ...
    def release(self) -> None: ...


def wait_until(deadline: float, wait: Callable[[float], bool], lock: Releasable | None = None) -> bool:
    """Wait until `time.perf_counter()` reaches `deadline`. `wait(timeout)`, an `Event.wait` or a `Condition.wait`,
    waits while more than `WAKE_STEP` is left; when it returns True (the event set, the condition notified) this
    returns True at once. The last stretch is slept, under `WAKE_STEP`, with `lock` released if one is given: the
    condition's, whose wait the caller calls holding it. False when the deadline came."""
    remaining = deadline - time.perf_counter()
    if remaining > WAKE_STEP and wait(remaining - WAKE_STEP):
        return True
    remaining = deadline - time.perf_counter()
    if remaining <= 0.0:
        return False
    if lock is None:
        time.sleep(remaining)
        return False
    lock.release()
    try:
        time.sleep(remaining)
    finally:
        lock.acquire()
    return False
