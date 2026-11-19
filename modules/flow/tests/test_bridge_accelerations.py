"""The 3D bridge's bandpass scales: the change as an acceleration, (fast − slow) × `change_acceleration_scale`, sums
× dt over a change of speed to that change, Δv, at any frame rate and for a sudden or a gradual change; as a
velocity, (fast − slow) / `step_peak`, it reads Δv at its height only for a sudden one."""

import unittest

import numpy as np

from modules.flow import band_width, change_acceleration_scale, decay_weight, step_peak

FAST, SLOW = 25.0, 150.0      # ms, the preset's bandpass
DV: np.ndarray = np.array([[0.6, -0.3, 0.2]])   # widths/s


def speeds(dt: float, ramp: float = 0.0, seconds: float = 3.0) -> list[np.ndarray]:
    """A limb from rest to DV, at once or over `ramp` s, then held."""
    t: np.ndarray = np.arange(int(seconds / dt)) * dt
    share: np.ndarray = np.clip((t - 0.5) / ramp, 0.0, 1.0) if ramp > 0.0 else (t >= 0.5).astype(float)
    return [DV * s for s in share]


def bandpass(velocities: list[np.ndarray], dt: float, acceleration: bool, start: np.ndarray | None = None) -> list[np.ndarray]:
    """(fast − slow) per frame of two Emas of `velocities`, times the scale the bandpass hands its shaders."""
    fast = slow = start if start is not None else np.zeros_like(DV)
    fast_weight, slow_weight = decay_weight(dt, FAST), decay_weight(dt, SLOW)
    peak: float = step_peak(FAST, SLOW)
    scale: float = change_acceleration_scale(FAST, SLOW, dt) if acceleration else 1.0 / peak
    out: list[np.ndarray] = []
    for v in velocities:
        fast = fast_weight * fast + (1.0 - fast_weight) * v
        slow = slow_weight * slow + (1.0 - slow_weight) * v
        out.append((fast - slow) * scale)
    return out


class BandpassAccelerationTest(unittest.TestCase):
    def test_the_change_as_an_acceleration_adds_up_to_its_dv_at_any_frame_rate(self) -> None:
        for dt in (1.0 / 60.0, 1.0 / 25.0, 1.0 / 120.0):
            with self.subTest(dt=dt):
                area: np.ndarray = sum(bandpass(speeds(dt), dt, True)) * dt
                np.testing.assert_allclose(area, DV, rtol=0.01, atol=1e-4)

    def test_a_gradual_change_adds_up_to_the_same_dv(self) -> None:
        dt: float = 1.0 / 60.0
        area: np.ndarray = sum(bandpass(speeds(dt, ramp=0.4), dt, True)) * dt
        np.testing.assert_allclose(area, DV, rtol=0.01, atol=1e-4)

    def test_a_stop_brakes_by_the_same_dv(self) -> None:
        dt: float = 1.0 / 60.0
        stop: list[np.ndarray] = [DV - v for v in speeds(dt)]
        area: np.ndarray = sum(bandpass(stop, dt, True, start=DV)) * dt
        np.testing.assert_allclose(area, -DV, rtol=0.01, atol=1e-4)

    def test_the_change_as_a_velocity_reads_a_sudden_dv_at_its_height_and_a_gradual_one_less(self) -> None:
        dt: float = 1.0 / 240.0
        sudden: float = max(float(np.linalg.norm(c)) for c in bandpass(speeds(dt), dt, False))
        gradual: float = max(float(np.linalg.norm(c)) for c in bandpass(speeds(dt, ramp=0.4), dt, False))
        self.assertAlmostEqual(sudden, float(np.linalg.norm(DV)), delta=0.02)
        self.assertLess(gradual, 0.5 * sudden)

    def test_the_bands_width_is_slow_minus_fast_for_a_short_frame(self) -> None:
        self.assertAlmostEqual(band_width(FAST, SLOW, 1e-4), (SLOW - FAST) / 1000.0, places=6)
        self.assertAlmostEqual(band_width(0.0, SLOW, 1e-4), SLOW / 1000.0, delta=1e-4)     # within half a frame

    def test_no_band_gives_no_change(self) -> None:
        self.assertEqual(step_peak(SLOW, SLOW), 0.0)
        self.assertEqual(change_acceleration_scale(SLOW, SLOW, 1.0 / 60.0), 0.0)

    def test_the_step_peak(self) -> None:
        self.assertEqual(step_peak(0.0, 150.0), 1.0)
        self.assertAlmostEqual(step_peak(25.0, 150.0), 0.582, places=3)
        self.assertEqual(step_peak(150.0, 150.0), 0.0)
        self.assertEqual(step_peak(200.0, 150.0), 0.0)


if __name__ == '__main__':
    unittest.main()
