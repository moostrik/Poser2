"""Tests for the Azimuth smoother, predictor and chase interpolator: angular handling across the ±π join."""

import math
import unittest

import numpy as np

from modules.pose.features import Azimuth
from modules.pose.frame import Frame
from modules.pose.nodes import (
    AzimuthChaseInterpolator, AzimuthEuroSmoother, AzimuthPredictor, ChaseInterpolatorSettings,
    EuroSmootherSettings, PredictionMethod, PredictorSettings,
)

from ._builders import FPS, frame, wrap


def _azimuth_frame(i: int, azimuth: float) -> Frame:
    return frame(t=i / FPS, features={Azimuth: Azimuth.from_value(azimuth)})


def _off_short_arc(value: float, target: float) -> float:
    """Angular distance from ``target``, wrapped so crossing ±π is not a jump."""
    return abs(wrap(value - target))


class AzimuthEuroSmootherTest(unittest.TestCase):
    def test_step_across_pi_goes_the_short_way(self) -> None:
        f = AzimuthEuroSmoother(EuroSmootherSettings())
        for i in range(10):
            f.process(_azimuth_frame(i, math.pi - 0.1))
        for i in range(10, 20):
            out = f.process(_azimuth_frame(i, -math.pi + 0.1))[Azimuth].value
            self.assertGreater(abs(out), math.pi - 0.15, f"frame {i} swung through 0: {out}")

    def test_reduces_noise_on_a_still_azimuth(self) -> None:
        f = AzimuthEuroSmoother(EuroSmootherSettings())
        rng = np.random.default_rng(1)
        noisy = 2.0 + rng.normal(0.0, 0.02, 120)
        smoothed = [f.process(_azimuth_frame(i, float(v)))[Azimuth].value for i, v in enumerate(noisy)]
        self.assertLess(float(np.std(smoothed[30:])), 0.5 * float(np.std(noisy[30:])))

    def test_output_is_a_valid_azimuth(self) -> None:
        f = AzimuthEuroSmoother(EuroSmootherSettings())
        for i in range(5):
            out = f.process(_azimuth_frame(i, math.pi - 0.01 * i))[Azimuth]
            ok, err = out.validate()
            self.assertTrue(ok, err)


class AzimuthPredictorTest(unittest.TestCase):
    def test_linear_prediction_wraps_across_pi(self) -> None:
        settings = PredictorSettings()
        settings.method = PredictionMethod.LINEAR
        p = AzimuthPredictor(settings)
        p.process(_azimuth_frame(0, math.pi - 0.2))
        out = p.process(_azimuth_frame(1, math.pi - 0.05))[Azimuth].value
        self.assertAlmostEqual(out, wrap(math.pi + 0.1), places=5)


class AzimuthChaseInterpolatorTest(unittest.TestCase):
    def test_chases_across_pi_the_short_way(self) -> None:
        interpolator = AzimuthChaseInterpolator(ChaseInterpolatorSettings())
        interpolator.set(_azimuth_frame(0, math.pi - 0.1))
        interpolator.update()
        target = -math.pi + 0.1
        interpolator.set(_azimuth_frame(1, target))
        value = math.pi - 0.1
        for _ in range(60):
            out = interpolator.update()
            assert out is not None
            value = out[Azimuth].value
            # The chase may overshoot a little; the long way round would pass through 0.
            self.assertGreater(abs(value), math.pi - 0.3, f"swung through 0: {value}")
        self.assertLess(_off_short_arc(value, target), 0.01)


if __name__ == "__main__":
    unittest.main()
