"""Waker: on time to the millisecond at its deadline, at once on `set()`, and a `set()` before the wait kept."""

import sys
import threading
import time
import unittest

from modules.utils import Waker

TOLERANCE = 0.003   # s past the deadline, or after set(), a wait may return
TICK = 0.0156 if sys.platform == 'win32' else 0.0     # s, Windows' system tick


class WakerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.waker = Waker()

    def tearDown(self) -> None:
        self.waker.close()

    def test_it_returns_at_the_deadline_not_on_the_timer_tick(self) -> None:
        for delay in (0.001, 0.004, 0.018, 0.033):
            deadline = time.perf_counter() + delay
            self.assertFalse(self.waker.wait(deadline))
            late = time.perf_counter() - deadline
            self.assertGreaterEqual(late, 0.0)
            self.assertLess(late, TOLERANCE, f'{delay * 1000:.0f} ms wait came {late * 1000:.1f} ms late')

    def test_a_past_deadline_returns_at_once(self) -> None:
        started = time.perf_counter()
        self.assertFalse(self.waker.wait(started - 1.0))
        self.assertLess(time.perf_counter() - started, 0.001)

    def test_a_set_wakes_it_at_once(self) -> None:
        set_at: list[float] = []

        def wake() -> None:
            set_at.append(time.perf_counter())
            self.waker.set()
        threading.Timer(0.05, wake).start()
        self.assertTrue(self.waker.wait(time.perf_counter() + 2.0))
        self.assertLess(time.perf_counter() - set_at[0], TOLERANCE)

    def test_a_set_before_the_wait_is_kept_and_taken_once(self) -> None:
        self.waker.set()
        started = time.perf_counter()
        self.assertTrue(self.waker.wait(started + 2.0))
        self.assertLess(time.perf_counter() - started, TOLERANCE)
        self.assertFalse(self.waker.wait(time.perf_counter() + 0.01))

    def test_none_waits_for_a_set_only(self) -> None:
        threading.Timer(0.05, self.waker.set).start()
        started = time.perf_counter()
        self.assertTrue(self.waker.wait(None))
        self.assertGreaterEqual(time.perf_counter() - started, 0.04)

    def test_a_set_after_close_does_nothing(self) -> None:
        self.waker.close()
        self.waker.set()

    def test_a_busy_python_thread_runs_while_it_waits(self) -> None:
        # The woken thread asks the busy one for the GIL with a timed wait of the switch interval, which on Windows
        # ends on the system tick: beside a pure-Python loop a wake comes up to one tick late, as time.sleep's does.
        stop = threading.Event()
        counts = [0]

        def spin() -> None:
            while not stop.is_set():
                counts[0] += 1
        busy = threading.Thread(target=spin)
        busy.start()
        try:
            for delay in (0.001, 0.018, 0.050):
                before = counts[0]
                deadline = time.perf_counter() + delay
                self.assertFalse(self.waker.wait(deadline))
                late = time.perf_counter() - deadline
                self.assertGreater(counts[0], before)
                self.assertLess(late, TOLERANCE + max(TICK, sys.getswitchinterval()),
                                f'{delay * 1000:.0f} ms wait came {late * 1000:.1f} ms late')
        finally:
            stop.set()
            busy.join()


if __name__ == '__main__':
    unittest.main()
