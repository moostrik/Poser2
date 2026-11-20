"""Tests for the light clock: its rate, one live setting the tick is paced at and carries for the layers, and its
wait for the deadline."""

import unittest
from time import perf_counter

from apps.white_space.light.clock import Clock, ClockSettings, Tick


class TickTest(unittest.TestCase):
    def test_interval_defaults_to_dt(self) -> None:
        self.assertAlmostEqual(Tick(0.0, 0.02).interval, 0.02, places=9)

    def test_interval_given_is_kept(self) -> None:
        self.assertAlmostEqual(Tick(0.0, 0.02, 0.05).interval, 0.05, places=9)


class ClockRateTest(unittest.TestCase):
    def test_interval_follows_the_setting_live(self) -> None:
        settings = ClockSettings()
        clock = Clock(settings)
        self.assertAlmostEqual(clock.interval, 1.0 / 32.0, places=9)
        settings.light_rate = 60.0
        self.assertAlmostEqual(clock.interval, 1.0 / 60.0, places=9)

    def test_the_tick_carries_the_interval_it_was_paced_at(self) -> None:
        settings = ClockSettings()
        settings.light_rate = 100.0
        clock = Clock(settings)
        first = clock.next_tick()                    # baseline tick, returns at once
        self.assertAlmostEqual(first.interval, 0.01, places=9)
        settings.light_rate = 200.0
        second = clock.next_tick()                   # paced one interval at the new rate
        self.assertAlmostEqual(second.interval, 0.005, places=9)
        self.assertGreater(second.dt, 0.0)

    def test_the_deadline_is_the_paced_time_not_the_late_moment(self) -> None:
        settings = ClockSettings()
        settings.light_rate = 100.0
        clock = Clock(settings)
        first = clock.next_tick()
        self.assertAlmostEqual(first.deadline, first.now, places=9)
        second = clock.next_tick()
        self.assertAlmostEqual(second.deadline, first.deadline + 0.01, places=9)
        self.assertGreaterEqual(second.now, second.deadline)


class ClockWaitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ClockSettings()
        self.settings.light_rate = 100.0
        self.clock = Clock(self.settings)

    def tearDown(self) -> None:
        self.clock.close()

    def test_a_tick_comes_well_within_the_system_tick_of_its_deadline(self) -> None:
        # Windows' timed waits end on its 15.6 ms tick; the Waker's on the deadline
        self.clock.next_tick()
        late = []
        for _ in range(50):
            tick = self.clock.next_tick()
            late.append(tick.now - tick.deadline)
        self.assertLess(sum(late) / len(late), 0.002)
        self.assertLess(max(late), 0.008)

    def test_wake_ends_the_wait_at_once(self) -> None:
        self.settings.light_rate = 1.0
        self.clock.next_tick()
        self.clock.wake()
        start = perf_counter()
        self.clock.next_tick()
        self.assertLess(perf_counter() - start, 0.1)


if __name__ == "__main__":
    unittest.main()
