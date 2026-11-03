"""The depth velocity's numpy reference: z from a depth track in disparity, carried along the motion and
corrected each frame by the mean over a window of the depth samples within the tolerance, in view
widths per frame; and the noise floor taken off it."""

import unittest

import numpy as np

from modules.flow import catch_up_share, color_motion, depth_threshold, filter_depth_steps, track_depth, track_weights

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
          tolerance: float = TOLERANCE, time: float = 0.0, span: float = SPAN, dead_zone: float = 0.0,
          catch_up_time: float = 0.0, dt: float = DT) -> list[np.ndarray]:
    """The track after each depth frame, from no track."""
    state: np.ndarray = np.zeros(depths[0].shape + (4,))
    catch_up: np.ndarray = np.zeros(depths[0].shape + (2,))
    states: list[np.ndarray] = []
    for depth in depths:
        state, catch_up = track_depth(depth, state, catch_up, still(depth.shape) if moved is None else moved, dt, window, step, tolerance,
                                      time, span, dead_zone, catch_up_time, 0.5)
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

    def test_a_slow_slide_over_a_slope_reads_no_z_carried_by_its_displacement(self) -> None:
        # 0.4 pixels per frame, below the x/y threshold: carried by the displacement as solved it reads
        # (almost) nothing; carried by the thresholded one, which is 0 here, the slope reads as z
        rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        depths = [2.0 + 0.01 * (columns - 0.4 * frame) for frame in range(10)]
        moved = still()
        moved[..., 0] = 0.4
        carried = interior(z(depths, moved=moved, window=0, time=50.0))
        held = interior(z(depths, window=0, time=50.0))
        self.assertLess(float(np.abs(carried).max()), 1e-4)
        self.assertGreater(float(np.abs(held).min()), 1e-3)

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


class DeadZoneTest(unittest.TestCase):
    """Within the dead zone of its anchor the track reads no z; past it, full speed."""
    ZONE: float = 0.005

    def test_a_wander_within_the_zone_reads_nothing(self) -> None:
        # ±3 mm over 400 ms, as a stereo surface wanders; without the zone it reads as z
        depths = [uniform(0.7 + 0.003 * np.sin(2 * np.pi * frame * DT / 0.4)) for frame in range(60)]
        self.assertEqual(max(float(np.abs(s[..., 0]).max()) for s in track(depths, window=0, time=50.0, dead_zone=self.ZONE)), 0.0)
        self.assertGreater(max(float(np.abs(s[..., 0]).max()) for s in track(depths, window=0, time=50.0)), 0.0)

    def test_depth_steps_read_nothing(self) -> None:
        rng = np.random.default_rng(4)
        depths = [uniform(0.7 + 0.001 * float(rng.integers(0, 2))) for _ in range(60)]
        self.assertEqual(max(float(np.abs(s[..., 0]).max()) for s in track(depths, window=0, time=50.0, dead_zone=self.ZONE)), 0.0)

    def test_a_real_motion_comes_through_at_full_speed_past_the_zone(self) -> None:
        # still, then 1 m/s toward the camera: the first 5 mm read nothing, then z as without the zone
        depths = [uniform(0.9 - (max(frame - 5, 0) * 1.0 * DT)) for frame in range(20)]
        zoned = track(depths, window=0, tolerance=0.5, time=50.0, dead_zone=self.ZONE)
        free = track(depths, window=0, tolerance=0.5, time=50.0)
        start: float = 1.0 / float(free[4][1, 1, 1])
        past = [i for i, s in enumerate(free) if start - 1.0 / float(s[1, 1, 1]) > self.ZONE + 1e-6]
        self.assertTrue(past and past[0] < 12)
        before = [i for i in range(5, past[0])]
        self.assertTrue(all(float(zoned[i][1, 1, 0]) == 0.0 for i in before))
        for i in past:
            self.assertAlmostEqual(float(zoned[i][1, 1, 0]), float(free[i][1, 1, 0]), places=12)

    def test_no_zone_is_the_track_as_it_was(self) -> None:
        rng = np.random.default_rng(5)
        depths = [2.0 + rng.normal(0.0, 0.03, (SIZE, SIZE)) for _ in range(10)]
        for zoned, free in zip(track(depths, time=50.0, tolerance=1.0, dead_zone=0.0), track(depths, time=50.0, tolerance=1.0)):
            np.testing.assert_array_equal(zoned[..., :3], free[..., :3])
            np.testing.assert_allclose(zoned[..., 3], np.where(free[..., 1] > 0, 1.0 / np.where(free[..., 1] > 0, free[..., 1], 1.0), 0.0))

    def test_a_track_that_starts_over_is_anchored_anew(self) -> None:
        # the chest at 1 m, then a hand at 0.7 m over it: the track starts over, anchored at the hand,
        # and the hand's own wander reads nothing
        depths = [uniform(1.0)] * 5 + [uniform(0.7 + 0.003 * (frame % 2)) for frame in range(10)]
        states = track(depths, window=0, time=50.0, dead_zone=self.ZONE)
        np.testing.assert_allclose(states[5][..., 3], 0.7)
        self.assertEqual(max(float(np.abs(s[..., 0]).max()) for s in states), 0.0)


class WeightsTest(unittest.TestCase):
    def test_no_time_takes_the_measurement_whole(self) -> None:
        self.assertEqual(track_weights(DT, 0.0, 1.0), (1.0, 1.0))

    def test_one_time_constant_leaves_one_over_e(self) -> None:
        alpha, beta = track_weights(0.1, 100.0, 1.0)
        self.assertAlmostEqual(float(alpha), 1.0 - np.exp(-1.0))
        self.assertAlmostEqual(float(beta), float(alpha) ** 2 / (2.0 - float(alpha)))

    def test_the_share_of_the_window_scales_the_correction(self) -> None:
        self.assertAlmostEqual(float(track_weights(DT, 0.0, 0.5)[0]), 0.5)


class CatchUpTest(unittest.TestCase):
    """With z_catch_up_time, a track that keeps missing one way where the image moves follows faster,
    as much as it keeps missing that way."""
    MOVING: np.ndarray = np.tile([1.0, 0.0], (SIZE, SIZE, 1))     # the image moving a pixel per frame, as a hand's edges do

    @staticmethod
    def approach(speed: float, frames: int = 50) -> list[np.ndarray]:
        # still at 2 m, then toward the camera at 40 m/s² up to `speed`, 50 fps
        v, d, depths = 0.0, 2.0, []
        for frame in range(frames):
            v = min(v + 40.0 * 0.02, speed) if frame >= 5 else 0.0
            d -= v * 0.02
            depths.append(uniform(max(d, 0.6)))
        return depths

    def follow(self, speed: float, catch_up_time: float, moving: bool = True) -> list[np.ndarray]:
        return track(self.approach(speed), moved=self.MOVING if moving else None, window=0, time=50.0, dead_zone=0.02,
                     catch_up_time=catch_up_time, dt=0.02)

    @staticmethod
    def restarts(states: list[np.ndarray]) -> int:
        return sum(1 for before, now in zip(states[6:], states[7:]) if float(before[1, 1, 1]) > 0 and float(now[1, 1, 2]) == 0.0
                   and float(now[1, 1, 1]) < 1.0 / 0.6)

    def test_a_fast_approach_to_6_m_s_is_followed_without_starting_over(self) -> None:
        for speed in (3.0, 4.0, 6.0):
            self.assertGreater(self.restarts(self.follow(speed, 0.0)), 0)       # today: it starts over and over
            self.assertEqual(self.restarts(self.follow(speed, 12.5)), 0)

    def test_without_color_motion_it_does_not_catch_up(self) -> None:
        np.testing.assert_array_equal(np.stack(self.follow(4.0, 12.5, moving=False)), np.stack(self.follow(4.0, 0.0, moving=False)))

    def test_color_motion_is_the_largest_of_the_3x3_past_the_floor(self) -> None:
        moved = still()
        moved[10, 10] = (0.6, 0.0)
        motion = color_motion(moved, 0.5)
        self.assertTrue(motion[9:12, 9:12].all())
        self.assertEqual(int(motion.sum()), 9)

    def test_off_is_the_track_as_it_was(self) -> None:
        rng = np.random.default_rng(9)
        depths = [2.0 + rng.normal(0.0, 0.03, (SIZE, SIZE)) for _ in range(10)] + self.approach(3.0)[:20]
        state, catch_up, plain = np.zeros((SIZE, SIZE, 4)), np.zeros((SIZE, SIZE, 2)), []
        for depth in depths:
            state, catch_up = track_depth(depth, state, catch_up, still(), DT, 1, STEP, 0.1, 50.0, SPAN, 0.02, 0.0, 0.5)
            plain.append(state)
        for a, b in zip(plain, track(depths, window=1, tolerance=0.1, time=50.0, dead_zone=0.02)):
            np.testing.assert_array_equal(a, b)

    def test_a_flip_jitter_at_a_still_edge_reads_as_today(self) -> None:
        # one-frame jumps of 3–9 cm, back the next frame, where the image does not move
        rng = np.random.default_rng(10)
        depths, off = [], False
        for _ in range(200):
            off = (not off) and rng.random() < 0.3
            depths.append(uniform(0.7 + (rng.uniform(0.03, 0.09) * rng.choice((-1.0, 1.0)) if off else 0.0)))
        np.testing.assert_array_equal(np.stack(track(depths, window=0, time=50.0, dead_zone=0.02, catch_up_time=12.5)),
                                      np.stack(track(depths, window=0, time=50.0, dead_zone=0.02)))

    def test_a_flip_jitter_cancels_out_in_the_catch_up_share(self) -> None:
        rng = np.random.default_rng(11)
        signed, size, shares = np.zeros(1), np.zeros(1), []
        for _ in range(300):
            miss = rng.choice((-1.0, 1.0)) * rng.uniform(0.03, 0.09) if rng.random() < 0.3 else 0.0
            signed, size = signed + 0.33 * (miss - signed), size + 0.33 * (abs(miss) - size)
            shares.append(float(catch_up_share(signed, size, 0.02)[0]))
        self.assertLess(float(np.mean(shares)), 0.1)

    def test_a_push_raises_the_catch_up_share(self) -> None:
        state, catch_up, shares = np.zeros((4, 4, 4)), np.zeros((4, 4, 2)), []
        for depth in self.approach(4.0, 25):
            state, catch_up = track_depth(depth[:4, :4], state, catch_up, self.MOVING[:4, :4], 0.02, 0, STEP, 0.1, 50.0, SPAN, 0.02, 12.5, 0.5)
            shares.append(float(catch_up_share(catch_up[..., 0], catch_up[..., 1], 0.02)[1, 1]))
        self.assertGreater(max(shares), 0.5)                                  # it opens, and closes as it catches up

    def test_a_new_surface_still_starts_over(self) -> None:
        states = track([uniform(1.0)] * 6 + [uniform(0.7)], moved=self.MOVING, window=0, time=50.0, dead_zone=0.02, catch_up_time=12.5, dt=0.02)
        self.assertEqual(float(states[-1][1, 1, 2]), 0.0)
        np.testing.assert_allclose(states[-1][1, 1, 1], 1.0 / 0.7)


class StepFilterTest(unittest.TestCase):
    """Before the track, a sample whose 3×3 depth samples span two surfaces has no depth."""

    def test_a_flat_or_sloping_surface_is_unchanged(self) -> None:
        rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        flat, slope = uniform(1.0), 1.0 + 0.02 * columns               # 2 cm per sample, under the tolerance
        np.testing.assert_array_equal(filter_depth_steps(flat, STEP, TOLERANCE), flat)
        np.testing.assert_array_equal(filter_depth_steps(slope, STEP, TOLERANCE), slope)

    def test_a_silhouette_loses_a_band_a_sample_wide_on_each_side(self) -> None:
        depth = uniform(2.0)
        depth[:, :SIZE // 2] = 0.7                                       # the player on the left, the wall on the right
        kept = filter_depth_steps(depth, STEP, TOLERANCE)
        self.assertTrue(np.all(kept[:, SIZE // 2 - 1:SIZE // 2 + 1] == 0.0))
        np.testing.assert_array_equal(kept[:, :SIZE // 2 - 1], 0.7)
        np.testing.assert_array_equal(kept[:, SIZE // 2 + 1:], 2.0)

    def test_the_band_is_one_depth_sample_wide_at_its_spacing(self) -> None:
        depth = uniform(2.0)
        depth[:, :12] = 0.7
        kept = filter_depth_steps(depth, (3.0, 3.0), TOLERANCE)        # depth samples three pixels apart
        self.assertTrue(np.all(kept[:, 9:15] == 0.0))
        self.assertTrue(np.all(kept[:, :9] == 0.7) and np.all(kept[:, 15:] == 2.0))

    def test_no_depth_beside_it_is_no_step(self) -> None:
        depth = uniform(1.0)
        depth[:, :SIZE // 2] = 0.0
        np.testing.assert_array_equal(filter_depth_steps(depth, STEP, TOLERANCE), depth)

    def test_a_silhouette_jittering_between_its_two_surfaces_reads_no_z(self) -> None:
        # the edge's depth flips between the player and the wall from frame to frame; the rest is still
        rng = np.random.default_rng(8)
        depths = []
        for _ in range(60):
            depth = uniform(2.0)
            depth[:, :SIZE // 2] = 0.7
            flip = rng.random(SIZE) < 0.5
            depth[flip, SIZE // 2] = 0.7
            depth[:, SIZE // 2 - 1] = np.where(rng.random(SIZE) < 0.3, 0.7 + rng.uniform(0.02, 0.09, SIZE), 0.7)
            depths.append(filter_depth_steps(depth, STEP, TOLERANCE))
        states = track(depths, window=1, time=50.0, dead_zone=0.02)
        self.assertEqual(max(float(np.abs(s[..., 0]).max()) for s in states), 0.0)


class DepthThresholdTest(unittest.TestCase):
    def test_the_threshold_clears_the_floor_and_stays_continuous(self) -> None:
        np.testing.assert_allclose(depth_threshold(np.array([0.05, -0.05, 0.3, -0.3]), 0.1), [0.0, 0.0, 0.2, -0.2])


if __name__ == '__main__':
    unittest.main()
