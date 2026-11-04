"""The Lucas–Kanade flow's numpy references: it finds how far an image moved and gives that as a
velocity, and with xy_smooth filters it as z is. Also the trail bridge's shaping of a velocity. Its depth is tested in test_depth_velocity."""

import unittest

import numpy as np

from modules.flow import displacement_velocity, filter_displacement, lucas_kanade, threshold_displacement, track_depth, track_weights, velocity_trail, one_euro_rate, one_euro_trail

SIZE: int = 64


def pattern(dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """A smooth luminance pattern (0 to 1), moved by (dx, dy) pixels: its value at x is the unmoved one at x − d."""
    rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    x, y = columns - dx, rows - dy
    return 0.5 + 0.2 * np.sin(x / 5.0) * np.cos(y / 7.0) + 0.15 * np.sin((x + y) / 9.0) + 0.1 * np.cos(x / 13.0 - y / 6.0)


def interior(field: np.ndarray, margin: int = 12) -> np.ndarray:
    return field[margin:-margin, margin:-margin]


class LucasKanadeTest(unittest.TestCase):
    def test_a_one_pixel_shift_is_found(self) -> None:
        d = lucas_kanade(pattern(), pattern(1.0, 0.0))
        self.assertAlmostEqual(float(np.median(interior(d[..., 0]))), 1.0, delta=0.1)
        self.assertAlmostEqual(float(np.median(interior(d[..., 1]))), 0.0, delta=0.1)

    def test_a_larger_shift_is_found_through_the_mip_levels(self) -> None:
        d = lucas_kanade(pattern(), pattern(4.0, -2.0), levels=3)
        self.assertAlmostEqual(float(np.median(interior(d[..., 0]))), 4.0, delta=0.5)
        self.assertAlmostEqual(float(np.median(interior(d[..., 1]))), -2.0, delta=0.5)

    def test_no_motion_is_no_displacement(self) -> None:
        d = lucas_kanade(pattern(), pattern())
        self.assertLess(float(np.abs(d).max()), 1e-6)

    def test_a_flat_image_solves_to_no_motion(self) -> None:
        flat = np.full((SIZE, SIZE), 0.5)
        self.assertLess(float(np.abs(lucas_kanade(flat, flat)).max()), 1e-6)

    def test_a_nearly_flat_image_never_moves_past_the_levels_reach(self) -> None:
        # a faint gradient and a faint change: the window is almost flat, the solve's step would blow up
        rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        before = 0.5 + 1e-4 * columns
        after = before + 1e-3 * np.sin(rows)
        d = lucas_kanade(before, after, levels=4, window=2, damping=0.0)
        reach = sum(3 * 2 ** level for level in range(4))     # (window + 1) texels a level, a level's texel 2^level px
        self.assertTrue(np.all(np.isfinite(d)))
        self.assertLessEqual(float(np.abs(d).max()), reach + 1e-9)

    def test_the_same_surface_keeps_the_background_still_beside_a_moving_strip(self) -> None:
        # a strip 12 px wide, 0.3 m before a still background, moving 2 px right: beside it the background is still
        still, moved = pattern(), pattern()
        moved[:, 26:38] = pattern(2.0, 0.0)[:, 26:38]
        depth = np.full((SIZE, SIZE), 1.5)
        depth[:, 26:38] = 1.2
        plain = lucas_kanade(still, moved, levels=2, window=2)
        surface = lucas_kanade(still, moved, levels=2, window=2, depth=depth, tolerance=0.2)
        beside = slice(40, 44)                                      # background 2–6 px right of the strip
        self.assertLess(float(np.abs(surface[16:48, beside, 0]).mean()), float(np.abs(plain[16:48, beside, 0]).mean()))


class ThresholdTest(unittest.TestCase):
    def test_jitter_below_the_threshold_is_no_motion(self) -> None:
        np.testing.assert_array_equal(threshold_displacement(np.array([[0.2, 0.1]]), 0.5), [[0.0, 0.0]])

    def test_the_threshold_comes_off_the_length_and_the_direction_stays(self) -> None:
        np.testing.assert_allclose(threshold_displacement(np.array([[3.0, 4.0]]), 1.0), [[2.4, 3.2]])

    def test_no_threshold_leaves_it(self) -> None:
        np.testing.assert_allclose(threshold_displacement(np.array([[0.3, -0.2]]), 0.0), [[0.3, -0.2]])


class XySmoothTest(unittest.TestCase):
    """x/y through the depth track's filter, with xy_smooth."""
    DT: float = 1.0 / 30.0

    def follow(self, steps: list[np.ndarray], carry: np.ndarray | None = None, smooth: float = 50.0) -> list[np.ndarray]:
        alpha = float(track_weights(self.DT, smooth, 1.0)[0])
        state = np.zeros(steps[0].shape[:2] + (4,))
        out: list[np.ndarray] = []
        for step in steps:
            state = filter_displacement(step, state, np.zeros_like(step) if carry is None else carry, alpha)
            out.append(state[..., :2])
        return out

    def test_a_change_of_speed_lags_as_z_does(self) -> None:
        # z: still for 10 frames, then 0.01 of disparity per frame; x/y: the same steps as a displacement
        frames: list[float] = [0.0] * 10 + [0.01] * 20
        disparity: float = 0.5
        state = np.zeros((4, 4, 4))
        z: list[float] = []
        for step in frames:
            disparity += step
            state, _ = track_depth(np.full((4, 4), 1.0 / disparity), state, np.zeros((4, 4, 2)), np.zeros((4, 4, 2)), self.DT, 0, (1.0, 1.0), 1.0, 50.0, 1.0, 0.0, 0.0, 0.5)
            z.append(float(state[1, 1, 2]) * self.DT)                        # the rate, per frame
        xy = [float(f[1, 1, 0]) for f in self.follow([np.full((4, 4, 2), step) for step in frames])]
        np.testing.assert_allclose(xy, z, atol=1e-9)

    def test_no_smoothing_is_the_measurement(self) -> None:
        rng = np.random.default_rng(2)
        steps = [rng.normal(0.0, 1.0, (4, 4, 2)) for _ in range(5)]
        for measured, followed in zip(steps, self.follow(steps, smooth=0.0)):
            np.testing.assert_allclose(followed, measured, atol=1e-12)

    def test_still_jitter_is_quieter(self) -> None:
        rng = np.random.default_rng(3)
        steps = [rng.normal(0.0, 0.2, (8, 8, 2)) for _ in range(120)]
        followed = np.stack(self.follow(steps)[60:])
        self.assertLess(float(np.std(followed)), 0.2 * 0.5)

    def test_a_steady_motion_settles_on_its_speed(self) -> None:
        self.assertAlmostEqual(float(self.follow([np.full((4, 4, 2), 1.5)] * 40)[-1][1, 1, 0]), 1.5, places=3)

    def test_carried_along_its_motion_a_moving_bar_keeps_its_speed(self) -> None:
        # a bar 4 pixels wide moving 1 pixel per frame: carried along, its filter goes with it and settles
        width, frames = 64, 40
        alpha = float(track_weights(self.DT, 50.0, 1.0)[0])
        state = np.zeros((1, width, 4))
        for frame in range(frames):
            step = np.zeros((1, width, 2))
            step[0, 10 + frame:14 + frame, 0] = 1.0
            state = filter_displacement(step, state, step, alpha)
        self.assertGreater(float(state[0, 10 + frames - 1:14 + frames - 1, 0].min()), 0.8)
        self.assertTrue(np.isfinite(state).all())


class DisplacementVelocityTest(unittest.TestCase):
    def test_a_width_per_second_is_one(self) -> None:
        velocity = displacement_velocity(np.array([0.1, 0.0]), 0.1, 200, 100, (1.0, 1.0))
        np.testing.assert_allclose(velocity, [1.0, 0.0])

    def test_vertical_motion_is_counted_in_widths(self) -> None:
        # 10 pixels down in a 200x100 image is 0.1 of the height, 0.05 of the width
        velocity = displacement_velocity(np.array([0.0, 0.1]), 1.0, 200, 100, (1.0, 1.0))
        np.testing.assert_allclose(velocity, [0.0, 0.05])

    def test_strength_multiplies_each_axis(self) -> None:
        np.testing.assert_allclose(displacement_velocity(np.array([0.1, 0.1]), 1.0, 10, 10, (3.0, -1.0)), [0.3, -0.1])


class VelocityTrailTest(unittest.TestCase):
    def test_a_rising_xy_follows_the_attack(self) -> None:
        stepped = velocity_trail(np.zeros((1, 3)), np.array([[1.0, 0.0, 1.0]]), 0.1, 0.9)
        np.testing.assert_allclose(stepped, [[0.9, 0.0, 0.1]])

    def test_a_falling_xy_follows_the_release(self) -> None:
        stepped = velocity_trail(np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)), 0.1, 0.9)
        np.testing.assert_allclose(stepped, [[0.9, 0.0, 0.0]])


class OneEuroTest(unittest.TestCase):
    def test_the_rate_is_the_change_over_dt_low_passed(self) -> None:
        rate = one_euro_rate(np.zeros((1, 3)), np.array([[0.3, 0.4, 9.0]]), np.array([[1.0]]), 0.1, 0.5)
        np.testing.assert_allclose(rate, [[0.5 * 1.0 + 0.5 * 5.0]])

    def test_at_rest_the_cutoff_is_the_minimum(self) -> None:
        stepped = one_euro_trail(np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)), np.zeros((1, 1)), 1.0, 5.0, 0.1, 0.5)
        np.testing.assert_allclose(stepped[0, 0], np.exp(-0.1 * 2.0 * np.pi))

    def test_a_fast_change_follows_sooner(self) -> None:
        still = one_euro_trail(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 1)), 1.0, 1.0, 0.016, 0.9)
        moving = one_euro_trail(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), np.array([[20.0]]), 1.0, 1.0, 0.016, 0.9)
        self.assertGreater(moving[0, 0], still[0, 0] * 5.0)
        np.testing.assert_allclose(moving[0, 2], still[0, 2])


if __name__ == '__main__':
    unittest.main()
