"""PerformanceTimer's timing session: samples kept between start and stop, pooled per name."""

import unittest

from modules.utils import PerformanceTimer


class TestTimingSession(unittest.TestCase):

    def test_samples_outside_the_session_are_left_out(self) -> None:
        timer = PerformanceTimer(name="session test")
        timer.add_time(100.0, report=False)
        PerformanceTimer.start_session()
        for ms in (1.0, 2.0, 3.0):
            timer.add_time(ms, report=False)
        stats = PerformanceTimer.stop_session()
        timer.add_time(100.0, report=False)

        self.assertEqual(stats["session test"]["count"], 3)
        self.assertAlmostEqual(stats["session test"]["mean"], 2.0)
        self.assertAlmostEqual(stats["session test"]["max"], 3.0)
        self.assertEqual(PerformanceTimer.stop_session(), {})

    def test_timers_that_share_a_name_are_pooled(self) -> None:
        a = PerformanceTimer(name="pooled ")
        b = PerformanceTimer(name="pooled")
        PerformanceTimer.start_session()
        a.add_time(1.0, report=False)
        b.add_time(3.0, report=False)
        stats = PerformanceTimer.stop_session()

        self.assertEqual(stats["pooled"]["count"], 2)
        self.assertAlmostEqual(stats["pooled"]["p50"], 2.0)

    def test_omitted_warmup_samples_still_count_in_the_session(self) -> None:
        timer = PerformanceTimer(name="omit test", omit_init=5)
        PerformanceTimer.start_session()
        timer.add_time(1.0, report=False)
        stats = PerformanceTimer.stop_session()

        self.assertEqual(stats["omit test"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
