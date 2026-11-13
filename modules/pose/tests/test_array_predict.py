"""`Predict`: the fit's window holds distinct moments; a sample stamped as the newest replaces it."""

import unittest

import numpy as np

from ..nodes._utils.array_predict import Predict, PredictionMethod


class PredictRepeatTest(unittest.TestCase):
    def test_a_repeated_stamp_replaces_the_newest_sample(self) -> None:
        fit = Predict(1, PredictionMethod.LINEAR, fit_frames=3)
        fit.add_sample(np.array([0.0]), 0.0)
        fit.add_sample(np.array([1.0]), 1.0)
        fit.add_sample(np.array([1.0]), 1.0)                 # the same moment again
        self.assertEqual(fit.samples, 2)
        self.assertAlmostEqual(float(fit.at(2.0)[0]), 2.0)   # the motion of the two distinct moments: 1 a second

    def test_three_repeats_keep_the_motion(self) -> None:
        fit = Predict(1, PredictionMethod.QUADRATIC, fit_frames=3)
        fit.add_sample(np.array([0.0]), 0.0)
        fit.add_sample(np.array([1.0]), 1.0)
        for _ in range(3):
            fit.add_sample(np.array([1.0]), 1.0)
        self.assertNotAlmostEqual(float(fit.at(2.0)[0]), 1.0)     # before the fix: three alike, no motion

    def test_a_repeated_stamp_takes_the_new_value(self) -> None:
        fit = Predict(1, PredictionMethod.NONE, fit_frames=3)
        fit.add_sample(np.array([1.0]), 1.0)
        fit.add_sample(np.array([2.0]), 1.0)
        self.assertEqual(fit.samples, 1)
        self.assertAlmostEqual(float(fit.at(1.0)[0]), 2.0)


if __name__ == '__main__':
    unittest.main()
