"""Tests for the INTRO flash layer and its closest-approach guarantee (``_closest_pass``).

The width window can be stepped clean over on a fast crossing; ``_closest_pass`` guarantees one flash
on the frame where the playhead is *nearest* the pose — the local minimum of |offset| — firing in real
time on that sample whether it sits just before or just after the zero-crossing. ``PlayheadHaunted``
imports the same kernel.
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from apps.white_space.light import Tick
from apps.white_space.light.frame import Frame, BeamLightId
from apps.white_space.light.layers.beam.playhead_flash import (
    PlayheadFlash, PlayheadFlashSettings, _closest_pass)
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
    def get_tracklets(self):
        return self.tracklets

    def get_frames(self, stage: int):
        return self.frames


class PlayheadFlashTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PlayheadFlashSettings()
        self.cfg.width = 20.0                       # ±10°
        self.board = FlashBoard(frames={}, tracklets={})
        self.layer = PlayheadFlash(RES, self.cfg, self.board, pose_stage=4)

    def _sweep(self, *offsets_deg: float, id: int = 0) -> list[float]:
        """Run one pose through a sequence of offsets; return the front white lamp per tick."""
        out: list[float] = []
        for deg in offsets_deg:
            self.board.frames = {id: FlashPose(id, deg)}
            self.board.tracklets = {id: SimpleNamespace(is_active=True)}
            f = Frame(RES, Tick(0.0, 1 / 30))
            self.layer.render(f)
            out.append(float(f.beam_lights[BeamLightId.FRONT_WHITE]))
        return out

    def test_flash_is_full_brightness_without_ghost_data(self) -> None:
        self.assertEqual(self._sweep(5.0), [1.0])           # inside the window → full `white`

    def test_outside_the_window_stays_at_base(self) -> None:
        self.cfg.base_white = 0.2
        self.assertAlmostEqual(self._sweep(90.0)[0], 0.2, places=6)   # far side → base only

    def test_nan_offset_never_flashes(self) -> None:
        pose = FlashPose(0, 0.0)
        pose._offset = float("nan")
        self.board.frames = {0: pose}
        self.board.tracklets = {0: SimpleNamespace(is_active=True)}
        f = Frame(RES, Tick(0.0, 1 / 30))
        self.layer.render(f)
        self.assertEqual(float(f.beam_lights[BeamLightId.FRONT_WHITE]), 0.0)

    def test_a_pass_that_steps_over_the_window_still_flashes(self) -> None:
        # 40° → −38°: no sample lands inside ±10°, but the crossing must not be silently skipped.
        levels = self._sweep(40.0, -38.0)
        self.assertEqual(levels[-1], 1.0)

    def test_gap_notch_disables_the_guarantee(self) -> None:
        # A notch asks for darkness at exactly the crossing the guarantee would fire on.
        self.cfg.gap = 0.5
        self.assertEqual(self._sweep(40.0, -38.0), [0.0, 0.0])

    def test_inactive_poses_are_ignored(self) -> None:
        self.board.frames = {0: FlashPose(0, 0.0)}
        self.board.tracklets = {0: SimpleNamespace(is_active=False)}
        f = Frame(RES, Tick(0.0, 1 / 30))
        self.layer.render(f)
        self.assertEqual(float(f.beam_lights[BeamLightId.FRONT_WHITE]), 0.0)

    def test_reset_clears_the_pass_history(self) -> None:
        self._sweep(40.0)
        self.layer.reset()
        self.assertEqual(self._sweep(-38.0), [0.0])   # no prev → no guarantee fire


if __name__ == "__main__":
    unittest.main()
