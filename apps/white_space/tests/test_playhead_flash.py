"""Tests for the INTRO flash layer (the ticks closest to each crossing) and ``beam_haunted``'s
closest-approach kernel (``_closest_pass``).

``_closest_pass`` guarantees one flash on the frame where the playhead is *nearest* the pose — the
local minimum of |offset| — firing in real time on that sample whether it sits just before or just
after the zero-crossing.
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from apps.white_space.light import Tick, MotorCommand
from apps.white_space.light.frame import Frame, BeamLightId
from apps.white_space.light.layers.beam.flash import BeamFlash, BeamFlashSettings
from apps.white_space.light.layers.beam.haunted import _closest_pass
from apps.white_space.pose import PlayheadOffset

R = math.radians


def _fires(degrees: list[float]) -> list[bool]:
    """Run a per-tick offset sequence (degrees) through the detector; prev starts NaN (no history)."""
    prev = float("nan")
    out: list[bool] = []
    for deg in degrees:
        cur = R(deg)
        out.append(_closest_pass(prev, cur))
        prev = cur
    return out


class ClosestPassTest(unittest.TestCase):
    def test_fires_on_departing_sample_when_nearest(self) -> None:
        # 13° → 6° → −1°: the sample nearest zero is −1° (just past) → single fire there.
        self.assertEqual(_fires([13, 6, -1]), [False, False, True])

    def test_fires_on_approaching_sample_when_nearest(self) -> None:
        # 8° → 1° → −6°: the nearest sample is +1° (just before) → fires there, not "after".
        self.assertEqual(_fires([8, 1, -6]), [False, True, False])

    def test_exactly_one_fire_per_sweep(self) -> None:
        # A steady sweep fires once, on the minimum-|offset| sample (2°).
        self.assertEqual(_fires([30, 23, 16, 9, 2, -5, -12]),
                         [False, False, False, False, True, False, False])

    def test_far_side_never_fires(self) -> None:
        # |offset| stays past the near-half gate (π/2) → no guarantee flash for the opposite side.
        self.assertEqual(_fires([100, 95, 92, 95, 100]), [False] * 5)

    def test_nan_never_fires(self) -> None:
        self.assertFalse(_closest_pass(float("nan"), 0.0))     # no prev yet
        self.assertFalse(_closest_pass(0.1, float("nan")))     # offset absent (motor stopped)


# -- the layer ---------------------------------------------------------------------

RES = 200


class FlashPose:
    """A pose frame carrying only a PlayheadOffset — no GhostFeature anywhere, which is the
    point: the flash must be full brightness without the (opt-in, usually off) Ghoster."""

    def __init__(self, track_id: int, offset_deg: float) -> None:
        self.track_id = track_id
        self._offset = R(offset_deg)

    def __getitem__(self, feature_type):
        assert feature_type is PlayheadOffset
        return SimpleNamespace(value=self._offset)


class FlashBoard(SimpleNamespace):
    """Poses only: the flash never reads tracklets. Records the flashes it is handed."""

    def get_frames(self, stage: int):
        return self.frames

    def add_flash(self, azimuth: float, white: float, blue: float) -> None:
        self.flashes.append((azimuth, white, blue))


class FlashTest(unittest.TestCase):
    """At 36 rpm and 30 Hz the playhead steps 7.2° per tick."""

    INTERVAL = 1.0 / 30.0

    def setUp(self) -> None:
        self.cfg = BeamFlashSettings()
        self.board = FlashBoard(frames={}, flashes=[])
        self.layer = BeamFlash(RES, self.cfg, self.board, pose_stage=4)

    def _render(self, pose, playhead: float = 1.0) -> Frame:
        self.board.frames = {} if pose is None else {pose.track_id: pose}
        f = Frame(RES, Tick(0.0, self.INTERVAL), motor_command=MotorCommand(beam_rpm=36.0), playhead=playhead)
        self.layer.render(f)
        return f

    def _sweep(self, *offsets_deg: float, id: int = 0) -> list[float]:
        """Run one pose through a sequence of offsets; return the front white lamp per tick."""
        return [float(self._render(FlashPose(id, deg)).beam_lights[BeamLightId.FRONT_WHITE])
                for deg in offsets_deg]

    STEADY = (23.4, 16.2, 9.0, 1.8, -5.4, -12.6)     # a steady sweep, closest tick at +1.8°

    def test_one_frame_is_the_closest_tick(self) -> None:
        self.assertEqual(self._sweep(*self.STEADY), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_two_frames_straddle_the_crossing(self) -> None:
        self.cfg.frames = 2
        self.assertEqual(self._sweep(*self.STEADY), [0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

    def test_three_frames_are_the_closest_and_both_neighbours(self) -> None:
        self.cfg.frames = 3
        self.assertEqual(self._sweep(*self.STEADY), [0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    def test_a_phase_the_old_window_lit_twice_is_one_frame(self) -> None:
        # +5.0° and −2.2° both sat inside an 11.5° window; only −2.2° is the closest tick.
        self.assertEqual(self._sweep(12.2, 5.0, -2.2, -9.4), [0.0, 0.0, 1.0, 0.0])

    def test_a_shrinking_step_never_lights_an_extra_tick(self) -> None:
        # +3.0° then −2.9° (a tracking correction): both are within half a step, one frame lights.
        self.assertEqual(self._sweep(10.2, 3.0, -2.9, -10.1), [0.0, 1.0, 0.0, 0.0])

    def test_a_jittered_pass_that_misses_every_close_tick_still_flashes_once(self) -> None:
        # +4.0° and −4.0° are each more than half a 7.2° step out: the sign flip fires.
        self.assertEqual(self._sweep(12.0, 4.0, -4.0, -12.0), [0.0, 0.0, 1.0, 0.0])

    def test_the_next_pass_flashes_again(self) -> None:
        self._sweep(*self.STEADY)
        self.assertEqual(self._sweep(-100.0, 100.0, 9.0, 1.8), [0.0, 0.0, 0.0, 1.0])

    def test_no_history_needed_on_the_first_drawn_tick(self) -> None:
        # INTRO resets this layer on the very tick of the hit; that tick must still flash.
        self.assertEqual(self._sweep(1.8), [1.0])

    def test_flash_is_full_brightness_without_ghost_data(self) -> None:
        self.cfg.white = 0.7
        self.assertAlmostEqual(self._sweep(0.0)[0], 0.7, places=6)

    def test_away_from_the_crossing_stays_at_base(self) -> None:
        self.cfg.base_white = 0.2
        self.assertAlmostEqual(self._sweep(90.0)[0], 0.2, places=6)

    def test_nan_offset_never_flashes(self) -> None:
        pose = FlashPose(0, 0.0)
        pose._offset = float("nan")
        self.assertEqual(float(self._render(pose).beam_lights[BeamLightId.FRONT_WHITE]), 0.0)

    def test_nobody_with_a_pose_no_flash(self) -> None:
        self.assertEqual(float(self._render(None).beam_lights[BeamLightId.FRONT_WHITE]), 0.0)

    def test_each_lit_tick_posts_a_flash_at_the_playhead(self) -> None:
        self.cfg.frames = 2
        for deg in self.STEADY:
            self._render(FlashPose(0, deg), playhead=1.25)
        self.assertEqual(self.board.flashes, [(1.25, 1.0, 0.0), (1.25, 1.0, 0.0)])

    def test_no_playhead_posts_nothing(self) -> None:
        self._render(FlashPose(0, 0.0), playhead=float("nan"))
        self.assertEqual(self.board.flashes, [])

    def test_reset_starts_a_fresh_pass(self) -> None:
        self._sweep(1.8)
        self.layer.reset()
        self.assertEqual(self._sweep(0.5), [1.0])


if __name__ == "__main__":
    unittest.main()
