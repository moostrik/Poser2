"""Tests for the light clock's rate: one live setting the tick is paced at and carries for the layers."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
