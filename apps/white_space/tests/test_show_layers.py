"""Tests for the show layers: sound_light (levels → blue lamps, jitter window, stale
fallback), wind_down (the dying wall), and the pose_instrument (symmetric people-anchored
line patterns, the seamless join, sync growth, line motion, presence envelope)."""

import math
import unittest
from time import monotonic
from types import SimpleNamespace

import numpy as np

from modules.board import SoundLevels
from modules.pose import features

from apps.white_space.light import Tick, BeamLightId
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers.beam.sound_light import SoundLight, SoundLightSettings, SoundFallback
from apps.white_space.light.layers.projection.pose_instrument import (
    PoseInstrument, PoseInstrumentSettings, LineMotion, LineFlow,
    _signed_offset, _segment_counts, _line_distance, _between_distance)

RES = 200
HALF = RES // 2


def frame(time: float = 0.0) -> Frame:
    return Frame(RES, Tick(time, 1 / 30))


# -- sound_light -----------------------------------------------------------------

class SoundBoard(SimpleNamespace):
    def get_sound_levels(self) -> SoundLevels:
        return self.levels


class SoundLightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SoundLightSettings()
        self.board = SoundBoard(levels=SoundLevels())
        self.layer = SoundLight(RES, self.cfg, self.board)

    def _fresh(self, left: float, right: float) -> None:
        self.board.levels = SoundLevels(left=left, right=right, timestamp=monotonic())

    def test_levels_land_on_the_blue_lamps(self) -> None:
        self.cfg.smoothing_frames = 0
        self._fresh(0.8, 0.3)
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE],  0.8, places=5)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.RIGHT_BLUE], 0.3, places=5)
        self.assertEqual(float(f.beam_lights[BeamLightId.FRONT_WHITE]), 0.0)   # whites untouched
        self.assertEqual(float(f.beam_lights[BeamLightId.BACK_WHITE]),  0.0)
        self.assertEqual(float(f.light_img.sum()), 0.0)                       # no pixels written

    def test_gain_scales(self) -> None:
        self.cfg.smoothing_frames = 0
        self.cfg.gain = 0.5
        self._fresh(0.8, 0.4)
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE], 0.4, places=5)

    def test_smoothing_window_bridges_jitter(self) -> None:
        self.cfg.smoothing_frames = 2
        self._fresh(1.0, 1.0)
        self.layer.render(frame())
        self._fresh(0.0, 0.0)
        f = frame()
        self.layer.render(f)                                  # average of the last 2 frames
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE], 0.5, places=5)

    def test_stale_input_off_fallback(self) -> None:
        self.board.levels = SoundLevels(left=1.0, right=1.0, timestamp=monotonic() - 60.0)
        f = frame()
        self.layer.render(f)
        self.assertEqual(float(f.beam_lights.sum()), 0.0)      # never freezes at a stuck level

    def test_stale_input_pulse_fallback(self) -> None:
        self.cfg.fallback = SoundFallback.PULSE
        self.board.levels = SoundLevels()                      # never received
        f = frame(time=1.25)                                   # quarter period of the 0.2 Hz pulse
        self.layer.render(f)
        left, right = f.beam_lights[BeamLightId.LEFT_BLUE], f.beam_lights[BeamLightId.RIGHT_BLUE]
        self.assertGreater(left, 0.0)
        self.assertAlmostEqual(left, right, places=5)
        self.assertLessEqual(left, self.cfg.fallback_level + 1e-6)


# -- wind_down ---------------------------------------------------------------------

class WindDownTest(unittest.TestCase):
    """The dying wall: both white lamps fading over spin_down_seconds, monotonic, no
    pixels; progress is the readout."""

    def setUp(self) -> None:
        from apps.white_space.light.layers.beam.wind_down import WindDown, WindDownSettings
        self.cfg = WindDownSettings()
        self.layer = WindDown(RES, self.cfg, board=None)

    def _wall(self) -> float:
        f = frame()
        self.layer.render(f)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.FRONT_WHITE],
                               f.beam_lights[BeamLightId.BACK_WHITE], places=6)   # both whites alike
        self.assertEqual(float(f.beam_lights[BeamLightId.LEFT_BLUE]), 0.0)
        self.assertEqual(float(f.light_img.sum()), 0.0)                        # no pixels written
        return float(f.beam_lights[BeamLightId.FRONT_WHITE])

    def test_timed_fade_reaches_zero(self) -> None:
        self.cfg.spin_down_seconds = 1.0
        levels = [self._wall() for _ in range(45)]         # 1.5 s of 1/30 ticks
        self.assertGreater(levels[0], 0.9)                 # starts at the full wall
        self.assertEqual(levels[-1], 0.0)                  # gone by time alone
        self.assertEqual(levels, sorted(levels, reverse=True))   # monotonic, no snap
        self.assertEqual(self.cfg.progress, 1.0)

    def test_level_scales_the_wall(self) -> None:
        self.cfg.level = 0.5
        self.assertAlmostEqual(self._wall(), 0.5, places=3)

    def test_reset_restarts_at_the_full_wall(self) -> None:
        self.cfg.spin_down_seconds = 1.0
        for _ in range(60):
            self._wall()
        self.assertEqual(self.cfg.progress, 1.0)
        self.layer.reset()
        self.assertEqual(self.cfg.progress, 0.0)
        self.assertGreater(self._wall(), 0.9)              # the wall is back at full


# -- pose_instrument ----------------------------------------------------------------

IRES = 3600                 # one pixel per 0.1° — fine enough to measure line geometry
DEG = IRES // 360           # pixels per degree


class FakePose:
    """A fake pose frame: ``pose[FeatureType]`` over a prepared feature dict."""

    def __init__(self, by_type: dict) -> None:
        self._by_type = by_type

    def __getitem__(self, feature_type):
        return self._by_type[feature_type]


def _pose(azimuth_pos: float, sims: dict[int, float] | None = None, shoulders: float = 0.0,
          elbows: float = 0.0, legs: float = 0.0, tilt: float = 0.0) -> FakePose:
    """A fake pose at strip position ``azimuth_pos`` (0..1) with the given arm angles
    (radians, applied to both sides), leg deviation, torso tilt and pairwise sims."""
    angles = np.full(len(features.AngleLandmark), np.nan)
    angles[features.AngleLandmark.left_shoulder]  = shoulders
    angles[features.AngleLandmark.right_shoulder] = shoulders
    angles[features.AngleLandmark.left_elbow]     = elbows
    angles[features.AngleLandmark.right_elbow]    = elbows
    sim_values = np.full(16, np.nan)
    for j, v in (sims or {}).items():
        sim_values[j] = v
    return FakePose({
        features.Azimuth: SimpleNamespace(value=azimuth_pos * math.tau),
        features.BBox: {features.BBoxElement.height: 1.0},
        features.Angles: SimpleNamespace(values=angles),
        features.Similarity: SimpleNamespace(values=sim_values),
        features.LegDeviation: SimpleNamespace(value=legs),
        features.TorsoTilt: SimpleNamespace(value=tilt),
    })


class InstrumentBoard(SimpleNamespace):
    def get_tracklets(self):
        return self.tracklets

    def get_frames(self, stage: int):
        return self.frames

    def get_playhead_signals(self) -> SimpleNamespace:
        return SimpleNamespace(bars=self.bars)


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of lit pixels (no wrap handling — keep runs clear of 0)."""
    lit = values > 1e-6
    edges = np.flatnonzero(np.diff(np.concatenate(([False], lit, [False])).astype(int)))
    return [(int(s), int(e - s)) for s, e in zip(edges[::2], edges[1::2])]


class PoseInstrumentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PoseInstrumentSettings()
        self.cfg.attack_seconds = 0.0            # present at once — geometry tests read one frame
        self.board = InstrumentBoard(tracklets={}, frames={}, bars=0.0)
        self.layer = PoseInstrument(IRES, self.cfg, self.board, pose_stage=4)

    def _people(self, poses: dict[int, FakePose]) -> None:
        self.board.frames = poses
        self.board.tracklets = {id: SimpleNamespace(is_active=True) for id in poses}

    def _render(self, time: float = 0.0) -> Frame:
        f = Frame(IRES, Tick(time, 1 / 30))
        self.layer.render(f)
        return f

    # -- helper geometry --

    def test_signed_offset_takes_the_shortest_way(self) -> None:
        self.assertAlmostEqual(_signed_offset(0.1, 0.3), 0.2)
        self.assertAlmostEqual(_signed_offset(0.95, 0.05), 0.1)       # through the wrap
        self.assertAlmostEqual(_signed_offset(0.05, 0.95), -0.1)

    def test_segment_counts_round_and_crossfade(self) -> None:
        spacing = 1 / 36
        self.assertEqual(_segment_counts(7.2 * spacing, spacing, 0.1), (7, 7, 0.0))
        n_low, n_high, blend = _segment_counts(7.5 * spacing, spacing, 0.1)
        self.assertEqual((n_low, n_high), (7, 8))
        self.assertAlmostEqual(blend, 0.5, places=5)
        self.assertEqual(_segment_counts(0.3 * spacing, spacing, 0.1), (1, 1, 0.0))   # never below one line

    def test_line_distances(self) -> None:
        u = np.array([0.0, 0.5, 1.2, 0.25])
        np.testing.assert_allclose(_line_distance(u, 1), [0.0, 0.5, 0.2, 0.25])
        np.testing.assert_allclose(_line_distance(u, 2), [0.0, 0.0, 0.4, 0.5])   # harmonic 2 lines at halves
        np.testing.assert_allclose(_between_distance(u, 1), [0.5, 0.0, 0.3, 0.25])

    # -- the pattern per person --

    def test_neutral_shows_one_line_each_side_and_the_anchor(self) -> None:
        self._people({0: _pose(0.5)})
        f = self._render()
        runs = _runs(f.white)
        self.assertEqual(len(runs), 2)
        centre = int(0.5 * IRES)
        (s0, l0), (s1, l1) = runs
        self.assertAlmostEqual((s0 + l0 / 2) - centre, -(s1 + l1 / 2 - centre), delta=1.5)   # mirror-symmetric
        self.assertAlmostEqual(abs(s0 + l0 / 2 - centre), self.cfg.line_spacing * DEG, delta=2)  # one spacing out
        self.assertGreater(f.blue[centre], 0.0)                                        # the blue anchor
        self.assertEqual(f.white[centre], 0.0)                                         # no white at the person

    def test_arms_up_shows_many_thicker_lines_within_the_reach(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi)})
        f = self._render()
        runs = _runs(f.white)
        centre = int(0.5 * IRES)
        expected = int(self.cfg.extent_max / self.cfg.line_spacing)                    # lines per side
        self.assertEqual(len(runs), 2 * expected)
        self.assertGreater(min(l for _, l in runs), self.cfg.line_min * self.cfg.line_spacing * DEG * 2)
        reach = int((self.cfg.extent_max + self.cfg.line_edge) * DEG)
        self.assertEqual(float(f.white[:centre - reach - 1].sum()), 0.0)
        self.assertEqual(float(f.white[centre + reach + 1:].sum()), 0.0)
        for (s0, l0), (s1, l1) in zip(runs[:expected], reversed(runs[expected:])):
            self.assertAlmostEqual((s0 + l0 / 2) - centre, centre - (s1 + l1 / 2), delta=1.5)

    def test_bent_elbows_subdivide_the_spacing(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi)})
        straight = len(_runs(self._render().white))
        self.cfg.harmonics = 1                                   # subdivision disabled
        self.layer.reset()
        self._people({0: _pose(0.5, shoulders=math.pi, elbows=math.pi)})
        self.assertEqual(len(_runs(self._render().white)), straight)
        self.cfg.harmonics = 2
        self.layer.reset()
        self._people({0: _pose(0.5, shoulders=math.pi, elbows=math.pi)})
        self.assertGreater(len(_runs(self._render().white)), straight)

    def test_bent_legs_raise_the_blue_lines(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi, legs=1.0)})
        f = self._render()
        centre = int(0.5 * IRES)
        between = centre + int(1.5 * self.cfg.line_spacing * DEG)   # half-way between line 1 and 2
        self.assertGreater(f.blue[between], 0.0)
        self.assertLess(f.white[centre + int(self.cfg.line_spacing * DEG)], self.cfg.level)   # dimmed

    # -- the line world between people --

    def test_the_lines_between_two_people_are_shared(self) -> None:
        a, b = 0.1, 0.3
        self._people({0: _pose(a, shoulders=math.pi), 1: _pose(b, shoulders=math.pi)})
        f = self._render()
        mid = int(0.2 * IRES)
        span = int(0.05 * IRES)                                 # the overlap region, both windows full
        left, right = f.white[mid - span:mid], f.white[mid + 1:mid + span + 1][::-1]
        np.testing.assert_allclose(left, right, atol=1e-4)      # same lines counted from either side

    def test_a_third_person_only_changes_the_segments_it_bounds(self) -> None:
        self._people({0: _pose(0.1, shoulders=math.pi), 1: _pose(0.3, shoulders=math.pi)})
        before = self._render().white.copy()
        self.layer.reset()
        self._people({0: _pose(0.1, shoulders=math.pi), 1: _pose(0.3, shoulders=math.pi), 2: _pose(0.7)})
        after = self._render().white
        lo, hi = int(0.1 * IRES) + 1, int(0.3 * IRES)
        np.testing.assert_allclose(after[lo:hi], before[lo:hi], atol=1e-6)

    def test_thickness_blends_along_the_gap(self) -> None:
        # A thick pattern next to a neutral one, fully synced: the runs thin toward B.
        self._people({0: _pose(0.1, shoulders=math.pi, sims={1: 1.0}), 1: _pose(0.3, sims={0: 1.0})})
        f = self._render()
        lo, hi = int(0.1 * IRES) + 1, int(0.3 * IRES)
        lengths = [l for _, l in _runs(f.white[lo:hi])]
        self.assertGreater(len(lengths), 4)
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_line_count_crossfades_across_the_half_spacing(self) -> None:
        spacing = self.cfg.line_spacing / 360.0
        gap = 7.5 * spacing                                     # exactly on the boundary
        for n_blend, expect_partial in ((0.1, True), (0.0, False)):
            self.cfg.n_blend = n_blend
            self.layer.reset()
            self._people({0: _pose(0.1, shoulders=math.pi, sims={1: 1.0}), 1: _pose(0.1 + gap, sims={0: 1.0}, shoulders=math.pi)})
            f = self._render()
            mid = int(round((0.1 + gap / 2) * IRES))            # a line for n = 8, dark for n = 7
            first = f.white[int(round((0.1 + gap / 8) * IRES))]
            if expect_partial:
                self.assertGreater(f.white[mid], 0.05)
                self.assertLess(f.white[mid], first * 0.9)
            else:
                self.assertTrue(f.white[mid] < 1e-6 or abs(f.white[mid] - first) < 1e-3)

    # -- sync growth --

    def _mid_lit(self, f: Frame, at: float, span_deg: float = 6.0) -> float:
        c = int(at * IRES)
        idx = np.arange(c - int(span_deg * DEG), c + int(span_deg * DEG)) % IRES
        return float(f.white[idx].max())

    def test_sync_lines_the_gap_above_threshold_only(self) -> None:
        self._people({0: _pose(0.1, sims={1: 0.5}), 1: _pose(0.3, sims={0: 0.5})})
        self.assertEqual(self._mid_lit(self._render(), 0.2), 0.0)
        self.layer.reset()
        self._people({0: _pose(0.1, sims={1: 1.0}), 1: _pose(0.3, sims={0: 1.0})})
        self.assertGreater(self._mid_lit(self._render(), 0.2), 0.0)

    def test_sync_takes_the_shorter_arc_across_the_wrap(self) -> None:
        self._people({0: _pose(0.95, sims={1: 1.0}), 1: _pose(0.05, sims={0: 1.0})})
        f = self._render()
        self.assertGreater(self._mid_lit(f, 0.0), 0.0)                       # through the wrap
        self.assertEqual(float(f.white[int(0.3 * IRES):int(0.7 * IRES)].sum()), 0.0)   # never the long way

    def test_sync_reaches_through_an_intermediate_person(self) -> None:
        self._people({0: _pose(0.1, sims={2: 1.0}), 1: _pose(0.2), 2: _pose(0.3, sims={0: 1.0})})
        f = self._render()
        self.assertGreater(self._mid_lit(f, 0.15), 0.0)
        self.assertGreater(self._mid_lit(f, 0.25), 0.0)

    # -- line motion --

    def _first_line_px(self, f: Frame, side: int) -> int:
        centre = int(0.5 * IRES)
        lo, hi = (centre + 5 * DEG, centre + 15 * DEG) if side > 0 else (centre - 15 * DEG, centre - 5 * DEG)
        return lo + int(np.argmax(f.white[lo:hi]))

    def _motion(self, motion: LineMotion, flow: LineFlow, ticks: int = 6) -> tuple[int, int, int, int]:
        """First-line pixel per side before and after ``ticks`` at one spacing/s (φ stays
        well under ½, so the same line is still the first one in the search window)."""
        self.cfg.extent_min = 20.0                     # the first line stays well inside the window
        self.cfg.line_motion, self.cfg.line_flow, self.cfg.line_speed = motion, flow, 1.0
        self.cfg.lines_per_bar = 1.0
        self._people({0: _pose(0.5)})
        before = self._render()
        self.board.bars = 0.25
        for i in range(ticks):
            after = self._render(time=(i + 1) / 30)
        return (self._first_line_px(before, +1), self._first_line_px(after, +1),
                self._first_line_px(before, -1), self._first_line_px(after, -1))

    def test_static_lines_hold(self) -> None:
        r0, r1, l0, l1 = self._motion(LineMotion.STATIC, LineFlow.SYMMETRIC)
        self.assertEqual((r0, l0), (r1, l1))

    def test_symmetric_flow_moves_both_sides_outward(self) -> None:
        r0, r1, l0, l1 = self._motion(LineMotion.CONSTANT, LineFlow.SYMMETRIC)
        self.assertGreater(r1, r0 + 5)
        self.assertLess(l1, l0 - 5)

    def test_global_flow_moves_everything_one_way(self) -> None:
        r0, r1, l0, l1 = self._motion(LineMotion.CONSTANT, LineFlow.GLOBAL)
        self.assertGreater(r1, r0 + 5)
        self.assertGreater(l1, l0 + 5)

    def test_playhead_motion_follows_the_bars(self) -> None:
        r0, r1, l0, l1 = self._motion(LineMotion.PLAYHEAD, LineFlow.SYMMETRIC, ticks=1)
        self.assertAlmostEqual(r1 - r0, 0.25 * self.cfg.line_spacing * DEG, delta=2)

    # -- presence --

    def test_release_holds_then_fades_and_reset_clears(self) -> None:
        self.cfg.release_seconds = 1.0
        self._people({0: _pose(0.5, shoulders=math.pi)})
        lit = float(self._render().white.max())
        self._people({})                                        # gone
        held = float(self._render().white.max())
        self.assertGreater(held, 0.0)
        self.assertLess(held, lit)
        for _ in range(40):
            last = self._render()
        self.assertEqual(float(last.white.max()) + float(last.blue.max()), 0.0)
        self._people({0: _pose(0.5, shoulders=math.pi)})
        self._render()
        self._people({})
        self.layer.reset()
        f = self._render()
        self.assertEqual(float(f.white.max()) + float(f.blue.max()), 0.0)

    def test_inactive_participants_are_ignored(self) -> None:
        self.board.frames = {0: _pose(0.5)}
        self.board.tracklets = {0: SimpleNamespace(is_active=False)}
        f = self._render()
        self.assertAlmostEqual(float(f.white.sum()) + float(f.blue.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
