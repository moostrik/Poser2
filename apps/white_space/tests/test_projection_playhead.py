"""Tests for the projection playhead: the marker, and its dimming inside a person's mask."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from modules.pose import features

from apps.white_space.light import Tick
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers import ProjectionPlayhead, ProjectionPlayheadSettings, PoseInstrumentSettings

RES = 3600
DEG = RES // 360


class FakePose:
    def __init__(self, azimuth_pos: float, length: float = 1.0) -> None:
        self._by_type = {
            features.Azimuth: SimpleNamespace(value=azimuth_pos * math.tau),
            features.BBox: {features.BBoxElement.height: length},
        }

    def __getitem__(self, feature_type):
        return self._by_type[feature_type]


class PlayheadBoard(SimpleNamespace):
    def get_frames(self, stage: int):
        return self.frames


class ProjectionPlayheadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ProjectionPlayheadSettings()
        self.cfg.width = 1.0                                   # 10 px
        self.instrument = PoseInstrumentSettings()
        self.board = PlayheadBoard(frames={})
        self.layer = ProjectionPlayhead(RES, self.cfg, self.instrument.mask, self.board, pose_stage=4)

    def _render(self, playhead_pos: float) -> Frame:
        f = Frame(RES, Tick(0.0, 1 / 30), playhead=playhead_pos * math.tau)
        self.layer.render(f)
        return f

    def test_the_marker_is_at_full_level_away_from_people(self) -> None:
        self.board.frames = {0: FakePose(0.5)}
        f = self._render(0.25)
        centre = RES // 4
        np.testing.assert_array_equal(f.white[centre - 5:centre + 5], self.cfg.level)
        self.assertEqual(float(f.white.sum()), 10 * self.cfg.level)
        self.assertEqual(float(f.blue.sum()), 0.0)

    def test_the_marker_dims_inside_a_mask(self) -> None:
        self.board.frames = {0: FakePose(0.5)}
        f = self._render(0.5)
        centre = RES // 2
        np.testing.assert_allclose(f.white[centre - 5:centre + 5], self.cfg.level * self.instrument.mask.playhead_at_mask)

    def test_the_marker_dims_pixel_by_pixel_at_the_mask_edge(self) -> None:
        # The default mask is 3° × 1 at length 1: 15 px each side of the person.
        self.board.frames = {0: FakePose(0.5)}
        f = self._render(0.5 + 15 / RES)                       # the marker straddles the mask's edge
        centre = RES // 2 + 15
        dim = self.cfg.level * self.instrument.mask.playhead_at_mask
        np.testing.assert_allclose(f.white[centre - 5:centre + 1], dim)
        np.testing.assert_allclose(f.white[centre + 1:centre + 5], self.cfg.level)

    def test_the_mask_scales_with_pose_length(self) -> None:
        self.board.frames = {0: FakePose(0.5, length=3.0)}     # 3° × 2: 30 px each side
        f = self._render(0.5 + 25 / RES)
        np.testing.assert_allclose(f.white[RES // 2 + 20:RES // 2 + 30], self.cfg.level * self.instrument.mask.playhead_at_mask)

    def test_a_pose_without_an_azimuth_has_no_mask(self) -> None:
        self.board.frames = {0: FakePose(float("nan"))}
        f = self._render(0.5)
        np.testing.assert_array_equal(f.white[RES // 2 - 5:RES // 2 + 5], self.cfg.level)

    def test_no_playhead_draws_nothing(self) -> None:
        f = self._render(float("nan"))
        self.assertEqual(float(f.light_img.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
