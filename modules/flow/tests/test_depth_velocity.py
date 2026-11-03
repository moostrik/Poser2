"""The depth velocity's numpy reference: z from a depth track in disparity, carried along the motion and
corrected each frame by the mean over a window of the depth samples within the tolerance, in view
widths per frame; and the noise floor taken off it."""

import unittest

import numpy as np

from modules.flow import depth_threshold, track_depth, track_weights

SIZE: int = 24
SPAN: float = 0.5       # metres a view width spans per metre of depth
TOLERANCE: float = 0.1
STEP: tuple[float, float] = (1.0, 1.0)
DT: float = 1.0 / 30.0


def still(shape: tuple[int, int] = (SIZE, SIZE)) -> np.ndarray:
    return np.zeros(shape + (2,))


def uniform(depth: float) -> np.ndarray:
    return np.full((SIZE, SIZE), depth)


def interior(field: np.ndarray, margin: int = 8) -> np.ndarray:
    return field[margin:-margin, margin:-margin]


def track(depths: list[np.ndarray], moved: np.ndarray | None = None, window: int = 2, step: tuple[float, float] = STEP,
          tolerance: float = TOLERANCE, time: float = 0.0, span: float = SPAN) -> list[np.ndarray]:
    """The track after each depth frame, from no track."""
    state: np.ndarray = np.zeros(depths[0].shape + (3,))
    states: list[np.ndarray] = []
    for depth in depths:
        state = track_depth(depth, state, still(depth.shape) if moved is None else moved, DT, window, step, tolerance, time, span)
        states.append(state)
    return states


def z(depths: list[np.ndarray], **kwargs) -> np.ndarray:
    """z after the last frame (view widths per frame)."""
    return track(depths, **kwargs)[-1][..., 0]


class TwoFrameTest(unittest.TestCase):
    """With no time and a full window, the track is the change between the last two frames."""

    def test_coming_closer_is_positive(self) -> None:
        # 2 m to 1.9 m: the disparity's change over the new disparity, (1 − 1.9 / 2), over the span
        np.testing.assert_allclose(interior(z([uniform(2.0), uniform(1.9)], tolerance=0.5)), 0.05 / SPAN)

    def test_going_away_is_negative(self) -> None:
        np.testing.assert_allclose(interior(z([uniform(2.0), uniform(2.1)], tolerance=0.5)), -0.05 / SPAN)

    def test_the_first_frame_reads_no_motion(self) -> None:
        self.assertEqual(float(np.abs(z([uniform(2.0)])).max()), 0.0)

    def test_the_same_step_further_away_is_smaller_like_a_sideways_one(self) -> None:
        near = z([uniform(2.0), uniform(1.99)])
        far = z([uniform(4.0), uniform(3.99)])
        np.testing.assert_allclose(interior(far) / interior(near), 0.5)

    def test_it_is_the_change_of_two_frames_per_pixel(self) -> None:
        rng = np.random.default_rng(1)
        before, after = 2.0 + rng.normal(0.0, 0.03, (SIZE, SIZE)), 2.0 + rng.normal(0.0, 0.03, (SIZE, SIZE))
        np.testing.assert_allclose(z([before, after], window=0, tolerance=1.0), (1.0 - after / before) / SPAN, atol=1e-12)


class TrackTest(unittest.TestCase):
    def test_something_passing_over_the_pixels_starts_the_track_over(self) -> None:
        states = track([uniform(3.0), uniform(1.2), uniform(1.2)])
        self.assertEqual(float(np.abs(states[1][..., 0]).max()), 0.0)          # no spike
        self.assertEqual(float(np.abs(states[2][..., 0]).max()), 0.0)          # and no rate carried over
        np.testing.assert_allclose(states[1][..., 1], 1.0 / 1.2)              # the new surface's track

    def test_no_depth_or_no_view_span_is_no_track(self) -> None:
        self.assertEqual(float(np.abs(track([uniform(2.0), uniform(0.0)])[-1]).max()), 0.0)
        self.assertEqual(float(np.abs(track([uniform(2.0), uniform(1.9)], tolerance=0.5, span=0.0)[-1]).max()), 0.0)

    def test_the_background_beside_the_player_is_not_averaged_in(self) -> None:
        # the player on the left, 1 cm closer; the wall on the right, 5 cm further
        before, after = uniform(1.01), uniform(1.0)
        before[:, SIZE // 2:], after[:, SIZE // 2:] = 2.95, 3.0
        field = z([before, after])
        self.assertGreater(float(field[SIZE // 2, SIZE // 2 - 1]), 0.0)        # beside the wall: only the player counts
        self.assertLess(float(field[SIZE // 2, SIZE // 2]), 0.0)               # and the other way round

    def test_a_thin_window_corrects_less(self) -> None:
        # beside the wall only 3 of the 5 columns are the player: it moves the track less than a full window
        before, after = uniform(1.01), uniform(1.0)
        before[:, SIZE // 2:], after[:, SIZE // 2:] = 3.0, 3.0
        field = z([before, after])
        edge, full = float(field[SIZE // 2, SIZE // 2 - 1]), float(field[SIZE // 2, 4])
        self.assertGreater(edge, 0.0)
        self.assertLess(edge, full * 0.7)

    def test_a_still_noisy_depth_is_quieter_tracked_than_differenced(self) -> None:
        rng = np.random.default_rng(0)
        depths = [2.0 + rng.normal(0.0, 0.03, (SIZE, SIZE)) for _ in range(120)]
        differenced = np.std(np.stack([s[..., 0] for s in track(depths, window=0, tolerance=1.0)[60:]]))
        tracked = np.std(np.stack([s[..., 0] for s in track(depths, window=0, tolerance=1.0, time=100.0)[60:]]))
        self.assertLess(float(tracked), float(differenced) * 0.3)

    def test_a_steady_move_toward_the_camera_settles_on_its_speed(self) -> None:
        # steady in disparity, as the track is: from 2.5 m to 1.5 m over two seconds. At a steady speed in
        # metres the disparity's rate grows as the player comes closer, and the track lags it a little
        rate: float = (1.0 / 1.5 - 1.0 / 2.5) / 2.0
        depths = [uniform(1.0 / (1.0 / 2.5 + rate * DT * frame)) for frame in range(61)]
        field = z(depths, tolerance=0.5, time=100.0)
        expected: float = rate * DT / ((1.0 / 1.5) * SPAN)
        np.testing.assert_allclose(interior(field), expected, rtol=0.01)

    def test_independent_noise_drops_with_the_root_of_the_samples(self) -> None:
        # depth samples three flow pixels apart, as a coarse depth stretched onto the view
        rng = np.random.default_rng(0)
        samples: int = 48
        before = np.repeat(np.repeat(2.0 + rng.normal(0.0, 0.03, (samples, samples)), 3, axis=0), 3, axis=1)
        after = np.repeat(np.repeat(2.0 + rng.normal(0.0, 0.03, (samples, samples)), 3, axis=0), 3, axis=1)
        moved = still(before.shape)
        one = z([before, after], moved=moved, window=0, step=(3.0, 3.0), tolerance=1.0)
        many = z([before, after], moved=moved, window=2, step=(3.0, 3.0), tolerance=1.0)
        ratio: float = float(np.std(many[12:-12, 12:-12]) / np.std(one[12:-12, 12:-12]))
        self.assertAlmostEqual(ratio, 1.0 / 5.0, delta=0.05)       # 25 samples


class CarriedTest(unittest.TestCase):
    def test_a_surface_sliding_sideways_over_a_slope_reads_no_motion_along_its_motion(self) -> None:
        rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        before = 2.0 + 0.01 * columns                   # a slope: 1 cm per pixel
        after = 2.0 + 0.01 * (columns - 3.0)            # slid 3 pixels to the right
        moved = still()
        moved[..., 0] = 3.0
        self.assertGreater(float(np.abs(interior(z([before, after]))).min()), 0.02)      # held in place, 3 cm per frame
        self.assertLess(float(np.abs(interior(z([before, after], moved=moved))).max()), 1e-9)

    def test_a_hand_sliding_over_the_chest_reads_no_z_at_its_edge(self) -> None:
        # the hand at 0.85 m over the chest at 1 m, its edge 1 pixel right per frame; the flow at the
        # edge smears to 1.3 pixels, so the track is carried from between hand and chest texels. Taken
        # only from the texels of its own surface, no pixel reads a motion toward the camera
        depths: list[np.ndarray] = []
        for frame in range(8):
            depth = uniform(1.0)
            depth[:, :8 + frame] = 0.85
            depths.append(depth)
        moved = still()
        moved[..., 0] = 1.3
        states = track(depths, moved=moved, window=0, time=50.0)
        self.assertLess(max(float(np.abs(s[..., 0]).max()) for s in states), 1e-9)

    def test_a_surface_moving_toward_the_camera_along_its_motion_still_reads_so(self) -> None:
        moved = still()
        moved[..., 0] = 2.0
        np.testing.assert_allclose(interior(z([uniform(2.0), uniform(1.99)], moved=moved)), 0.005 / SPAN)


class WeightsTest(unittest.TestCase):
    def test_no_time_takes_the_measurement_whole(self) -> None:
        self.assertEqual(track_weights(DT, 0.0, 1.0), (1.0, 1.0))

    def test_one_time_constant_leaves_one_over_e(self) -> None:
        alpha, beta = track_weights(0.1, 100.0, 1.0)
        self.assertAlmostEqual(float(alpha), 1.0 - np.exp(-1.0))
        self.assertAlmostEqual(float(beta), float(alpha) ** 2 / (2.0 - float(alpha)))

    def test_the_share_of_the_window_scales_the_correction(self) -> None:
        self.assertAlmostEqual(float(track_weights(DT, 0.0, 0.5)[0]), 0.5)


class DepthThresholdTest(unittest.TestCase):
    def test_the_threshold_clears_the_floor_and_stays_continuous(self) -> None:
        np.testing.assert_allclose(depth_threshold(np.array([0.05, -0.05, 0.3, -0.3]), 0.1), [0.0, 0.0, 0.2, -0.2])


if __name__ == '__main__':
    unittest.main()
