"""wait_until: on time to the millisecond, early when an event is set or a condition notified, and a condition's lock
free while it sleeps."""

import threading
import time
import unittest

from modules.utils import wait_until

TOLERANCE = 0.003   # s past the deadline a wait may return


class WaitUntilTest(unittest.TestCase):
    def test_it_returns_at_the_deadline_not_on_the_timer_tick(self) -> None:
        event = threading.Event()
        for delay in (0.004, 0.020, 0.033):
            deadline = time.perf_counter() + delay
            self.assertFalse(wait_until(deadline, event.wait))
            late = time.perf_counter() - deadline
            self.assertGreaterEqual(late, 0.0)
            self.assertLess(late, TOLERANCE, f'{delay * 1000:.0f} ms wait came {late * 1000:.1f} ms late')

    def test_a_past_deadline_returns_at_once(self) -> None:
        started = time.perf_counter()
        self.assertFalse(wait_until(started - 1.0, threading.Event().wait))
        self.assertLess(time.perf_counter() - started, 0.001)

    def test_an_event_set_wakes_it_early(self) -> None:
        event = threading.Event()
        threading.Timer(0.05, event.set).start()
        started = time.perf_counter()
        self.assertTrue(wait_until(started + 2.0, event.wait))
        self.assertLess(time.perf_counter() - started, 0.5)

    def test_a_condition_notified_wakes_it_early(self) -> None:
        condition = threading.Condition()

        def notify() -> None:
            with condition:
                condition.notify_all()
        threading.Timer(0.05, notify).start()
        started = time.perf_counter()
        with condition:
            self.assertTrue(wait_until(started + 2.0, condition.wait, condition))
        self.assertLess(time.perf_counter() - started, 0.5)

    def test_the_condition_is_free_while_it_sleeps_and_held_after(self) -> None:
        condition = threading.Condition()
        taken: list[bool] = []

        def take() -> None:
            time.sleep(0.004)                   # inside the last stretch, which is slept
            taken.append(condition.acquire(timeout=0.5))
            if taken[-1]:
                condition.release()
        threading.Thread(target=take).start()
        with condition:
            self.assertFalse(wait_until(time.perf_counter() + 0.012, condition.wait, condition))
            self.assertTrue(condition._is_owned())      # pyright: ignore[reportAttributeAccessIssue]
        self.assertEqual(taken, [True])


if __name__ == '__main__':
    unittest.main()
