"""The 3D bridge's readings as accelerations: each one's sum × dt over a change of speed is that change, Δv, at any
frame rate and for a sudden or a gradual change; the readings as velocities read Δv at their height only for a
sudden one."""

import unittest

import numpy as np

from modules.flow import band_width, decay_weight, step_peak, trail_acceleration, velocity_trail, whole_change

FAST, SLOW = 25.0, 150.0      # ms, the preset's bandpass
DV: np.ndarray = np.array([[0.6, -0.3, 0.2]])   # widths/s


def speeds(dt: float, ramp: float = 0.0, seconds: float = 3.0) -> list[np.ndarray]:
    """A limb from rest to DV, at once or over `ramp` s, then held."""
    t: np.ndarray = np.arange(int(seconds / dt)) * dt
    share: np.ndarray = np.clip((t - 0.5) / ramp, 0.0, 1.0) if ramp > 0.0 else (t >= 0.5).astype(float)
    return [DV * s for s in share]


def bandpass(velocities: list[np.ndarray], dt: float, acceleration: bool, start: np.ndarray | None = None) -> list[np.ndarray]:
    """The whole change per frame through the fast and the slow trail."""
    fast = slow = start if start is not None else np.zeros_like(DV)
    fast_weight, slow_weight = decay_weight(dt, FAST), decay_weight(dt, SLOW)
    out: list[np.ndarray] = []
    for v in velocities:
        fast = fast_weight * fast + (1.0 - fast_weight) * v
        slow = slow_weight * slow + (1.0 - slow_weight) * v
        out.append(whole_change(fast, slow, FAST, SLOW, dt, acceleration))
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
        np.testing.assert_array_equal(whole_change(DV, np.zeros_like(DV), SLOW, SLOW, 1.0 / 60.0, True), np.zeros_like(DV))
        np.testing.assert_array_equal(whole_change(DV, np.zeros_like(DV), SLOW, SLOW, 1.0 / 60.0, False), np.zeros_like(DV))


class LowpassAccelerationTest(unittest.TestCase):
    ATTACK, SMOOTH = 40.0, 300.0    # ms

    def run_trail(self, velocities: list[np.ndarray], dt: float) -> tuple[np.ndarray, list[np.ndarray]]:
        trail: np.ndarray = np.zeros_like(DV)
        rates: list[np.ndarray] = []
        attack, release = decay_weight(dt, self.ATTACK), decay_weight(dt, self.SMOOTH)
        for v in velocities:
            previous: np.ndarray = trail
            trail = velocity_trail(previous, v, attack, release)
            rates.append(trail_acceleration(previous, trail, dt))
        return trail, rates

    def test_the_trails_rate_adds_up_to_its_change_at_any_frame_rate(self) -> None:
        for dt in (1.0 / 60.0, 1.0 / 25.0):
            with self.subTest(dt=dt):
                velocities: list[np.ndarray] = speeds(dt) + [np.zeros_like(DV)] * int(1.0 / dt)    # a start, then a stop
                trail, rates = self.run_trail(velocities, dt)
                np.testing.assert_allclose(sum(rates) * dt, trail, atol=1e-9)

    def test_a_start_adds_its_dv_and_a_stop_brakes_it_back(self) -> None:
        dt: float = 1.0 / 60.0
        start: list[np.ndarray] = speeds(dt)
        _, rates = self.run_trail(start, dt)
        np.testing.assert_allclose(sum(rates) * dt, DV, rtol=1e-3)
        _, rates = self.run_trail(start + [np.zeros_like(DV)] * int(4.0 / dt), dt)
        np.testing.assert_allclose(sum(rates) * dt, np.zeros_like(DV), atol=1e-3)

    def test_the_rate_is_the_highpass_over_tau_for_a_short_frame(self) -> None:
        dt: float = 1.0 / 1000.0
        previous: np.ndarray = np.zeros_like(DV)
        attack, release = decay_weight(dt, self.ATTACK), decay_weight(dt, self.SMOOTH)
        rate: np.ndarray = trail_acceleration(previous, velocity_trail(previous, DV, attack, release), dt)
        np.testing.assert_allclose(rate, DV / (self.ATTACK / 1000.0), rtol=0.02)     # the speed rises: over the attack


if __name__ == '__main__':
    unittest.main()
