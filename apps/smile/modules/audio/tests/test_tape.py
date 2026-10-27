"""The tape: the speed curve of the share clip's sound and the resampling that plays it."""

import unittest

import numpy as np

from modules.audio import tape


class TestTape(unittest.TestCase):
    def test_the_curve_spins_up_runs_and_winds_down(self):
        rate = 1000
        speed = tape.speed_curve(4.0, rate, spin_up=0.5, wind_down=1.0)
        self.assertEqual(len(speed), 4000)
        self.assertEqual(speed[0], 0.0)                     # standstill at the start
        self.assertLess(speed[50], tape.SPIN_UP_FLOOR + 0.5)
        self.assertAlmostEqual(float(speed.max()), 1.5, places=2)    # the peak of the wobble
        self.assertTrue(np.all(speed[600:2900] == 1.0))     # the middle at full speed
        self.assertAlmostEqual(speed[3500], 0.5, places=2)  # halfway down
        self.assertLess(speed[-1], 0.01)

    def test_the_spin_up_follows_the_patch(self):
        self.assertEqual(float(tape.spin_up_speed(0.0)), 0.0)
        self.assertAlmostEqual(float(tape.spin_up_speed(0.707)), 1.5, places=3)
        self.assertAlmostEqual(float(tape.spin_up_speed(1.0)), 0.773 * 1.5, places=3)

    def test_resampling_at_double_speed_doubles_the_pitch(self):
        rate = 8000
        t = np.arange(rate) / rate
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio = np.stack([sine, sine], axis=1)
        out = tape.resample(audio, np.full(rate // 2, 2.0), rate)
        self.assertEqual(out.shape, (rate // 2, 2))
        crossings = int(np.sum(np.diff(np.signbit(out[200:-200, 0])) != 0))
        # 880 Hz over the measured 0.45 s: about 792 zero crossings
        self.assertAlmostEqual(crossings, 2 * 880 * 0.45, delta=10)
        self.assertEqual(out[0, 0], 0.0)                    # the click fade

    def test_past_the_end_is_silence(self):
        audio = np.ones((100, 2), np.float32)
        out = tape.resample(audio, np.ones(300), 1000)
        self.assertTrue(np.all(out[150:] == 0.0))


if __name__ == '__main__':
    unittest.main()
