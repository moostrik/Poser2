"""Tests for the show layers: beam_blue_sound (levels → blue lamps, jitter window, stale
fallback), wind_down (the dying wall), and the pose_instrument (mirror-symmetric on/off line
patterns patched from the pose, the band, legibility, the hit, sync reach, presence envelope)."""

import math
import unittest
from time import monotonic
from types import SimpleNamespace

import numpy as np

from modules.board import SoundLevels
from modules.pose import features

from apps.white_space.light import Tick, BeamLightId, MotorCommand, MotorMode
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers.beam.blue_sound import BeamBlueSound, BeamBlueSoundSettings, SoundFallback
from apps.white_space.light.layers import PoseInstrument, PoseInstrumentSettings, PatchSettings, PoseControl
from apps.white_space.pose import PlayheadOffset

RES = 200
HALF = RES // 2


def frame(time: float = 0.0) -> Frame:
    return Frame(RES, Tick(time, 1 / 30))


# -- beam_blue_sound -----------------------------------------------------------------

class SoundBoard(SimpleNamespace):
    def get_sound_levels(self) -> SoundLevels:
        return self.levels


class BeamBlueSoundTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = BeamBlueSoundSettings()
        self.board = SoundBoard(levels=SoundLevels())
        self.layer = BeamBlueSound(RES, self.cfg, self.board)

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

class BeamWindDownTest(unittest.TestCase):
    """The dying wall: both white lamps fading over spin_down_seconds, monotonic, no
    pixels; progress is the readout."""

    def setUp(self) -> None:
        from apps.white_space.light.layers.beam.wind_down import BeamWindDown, BeamWindDownSettings
        self.cfg = BeamWindDownSettings()
        self.layer = BeamWindDown(RES, self.cfg, board=None)

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

IRES = 3600                 # one pixel per 0.1°
DEG = IRES // 360           # pixels per degree
C = IRES // 2               # the pixel of a person at normalized azimuth 0.5
TICK = 1 / 30
BAND = 15                   # band half width (px) at the default 3° and pose length 1
REACH = 450                 # default 45° reach (px)


class FakePose:
    """A fake pose frame: ``pose[FeatureType]`` over a prepared feature dict."""

    def __init__(self, by_type: dict) -> None:
        self._by_type = by_type

    def __getitem__(self, feature_type):
        return self._by_type[feature_type]


def _pose(azimuth_pos: float, sims: dict[int, float] | None = None, shoulders: float = 0.0,
          elbows: float = 0.0, legs: float = 0.0, tilt: float = 0.0,
          offset_deg: float = float("nan")) -> FakePose:
    """A fake pose at normalized azimuth ``azimuth_pos`` (0..1) with the given arm angles
    (radians, both sides), leg deviation, torso tilt, pairwise sims and playhead offset."""
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
        PlayheadOffset: SimpleNamespace(value=math.radians(offset_deg)),
    })


class InstrumentBoard(SimpleNamespace):
    """Poses only: the instrument never reads tracklets."""

    def get_frames(self, stage: int):
        return self.frames


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of lit pixels (no wrap handling — keep runs clear of 0)."""
    lit = values > 0.5
    edges = np.flatnonzero(np.diff(np.concatenate(([False], lit, [False])).astype(int)))
    return [(int(s), int(e - s)) for s, e in zip(edges[::2], edges[1::2])]


def _set(patch: PatchSettings, source: PoseControl, low: float, high: float) -> None:
    patch.source, patch.low, patch.high = source, low, high


class PoseInstrumentTest(unittest.TestCase):
    # Line centres a quarter pixel off the grid, so no line edge falls exactly on a pixel.
    QUARTER = 0.25 / 140.0

    def setUp(self) -> None:
        self.cfg = PoseInstrumentSettings()
        self.cfg.attack_seconds = 0.0            # present at once — geometry tests read one frame
        W, B = self.cfg.white, self.cfg.blue
        _set(W.duty, PoseControl.LIFT, 0.0, 1.0)               # arms down: no white
        _set(B.duty, PoseControl.LIFT, 1.0, 0.0)               # arms down: solid blue
        for channel in (W, B):
            _set(channel.interval, PoseControl.CONSTANT, 0.0, 0.5)    # 4 + 0.5 × 20 = 14° = 140 px
        _set(W.phase, PoseControl.CONSTANT, 0.0, self.QUARTER)
        _set(B.phase, PoseControl.CONSTANT, 0.0, 0.5 + self.QUARTER)  # blue between the white
        self.board = InstrumentBoard(frames={})
        self.layer = PoseInstrument(IRES, self.cfg, self.board, pose_stage=4, tick_interval=TICK)

    def _people(self, poses: dict[int, FakePose]) -> None:
        self.board.frames = poses

    def _render(self) -> Frame:
        f = Frame(IRES, Tick(0.0, TICK), motor_command=MotorCommand(mode=MotorMode.PROJECTION, beam_rpm=36.0))
        self.layer.render(f)
        return f

    def _outside_band(self, centre: int = C) -> np.ndarray:
        keep = np.ones(IRES, dtype=bool)
        keep[centre - BAND:centre + BAND + 1] = False
        return keep

    # -- one person --

    def test_arms_down_is_solid_blue_with_the_dim_band(self) -> None:
        self._people({0: _pose(0.5)})
        f = self._render()
        self.assertEqual(float(f.white.sum()), 0.0)
        np.testing.assert_array_equal(f.blue[C + BAND + 1:C + REACH + 1], 1.0)
        np.testing.assert_array_equal(f.blue[C - REACH:C - BAND], 1.0)
        np.testing.assert_allclose(f.blue[C - BAND:C + BAND + 1], self.cfg.band_level, atol=1e-6)
        self.assertEqual(float(f.blue[C + REACH + 1:].sum() + f.blue[:C - REACH].sum()), 0.0)

    def test_arms_up_is_solid_white_and_blue_only_in_the_band(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi)})
        f = self._render()
        np.testing.assert_array_equal(f.white[C + BAND + 1:C + REACH + 1], 1.0)
        np.testing.assert_array_equal(f.white[C - REACH:C - BAND], 1.0)
        np.testing.assert_array_equal(f.white[C - BAND:C + BAND + 1], 0.0)    # the band masks white
        self.assertEqual(float(f.blue[self._outside_band()].sum()), 0.0)

    def test_in_between_the_lines_are_full_and_mirror_symmetric(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi / 2)})
        f = self._render()
        outside = self._outside_band()
        self.assertTrue(np.isin(f.white, (0.0, 1.0)).all())
        self.assertTrue(np.isin(f.blue[outside], (0.0, 1.0)).all())
        self.assertGreater(len(_runs(f.white)), 4)
        self.assertGreater(len(_runs(f.blue)), 4)
        span = REACH + 10
        for channel in (f.white, f.blue):
            np.testing.assert_array_equal(channel[C + 1:C + span], channel[C - 1:C - span:-1])

    def test_half_lift_lines_have_half_the_interval(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi / 2)})
        f = self._render()
        inner = [length for start, length in _runs(f.white) if C + 50 < start and start + length < C + REACH - 50]
        self.assertTrue(inner)
        self.assertEqual(set(inner), {70})

    def test_no_line_or_gap_is_narrower_than_the_minimum_feature(self) -> None:
        for channel, duty in ((self.cfg.white, 0.4), (self.cfg.blue, 0.6)):
            _set(channel.interval, PoseControl.LEGS, 0.2, 0.35)     # 8° straight legs, 11° bent
            _set(channel.duty, PoseControl.CONSTANT, 0.0, duty)
            _set(channel.harmonic, PoseControl.CONSTANT, 0.0, 0.3)
            _set(channel.harmonic_phase, PoseControl.CONSTANT, 0.0, 0.2)
        # Two overlapping synced patterns of different intervals: the union is a moiré.
        b = round(0.537 * IRES)
        self.cfg.reach = 90.0                                   # a long overlap: plenty of interior
        reach = 900
        self._people({0: _pose(0.5, sims={1: 1.0}), 1: _pose(0.537, sims={0: 1.0}, legs=1.0)})
        f = self._render()
        min_px = int(self.cfg.min_feature * DEG)
        # Bands and reach edges may cut a line (it slides out): features touching them are exempt.
        cuts = [(C - BAND, C + BAND), (b - BAND, b + BAND)] + [(e, e) for e in (C - reach, b - reach, C + reach, b + reach)]

        def clear(start: int, stop: int) -> bool:
            return all(stop + min_px < lo or start - min_px > hi for lo, hi in cuts)

        for channel in (f.white, f.blue):
            runs = _runs(channel)
            self.assertGreater(len(runs), 5)
            lines = [length for start, length in runs if clear(start, start + length - 1)]
            gaps = [s1 - (s0 + l0) for (s0, l0), (s1, _) in zip(runs, runs[1:]) if clear(s0 + l0, s1 - 1)]
            self.assertGreater(len(lines), 3)
            self.assertTrue(all(length >= min_px for length in lines), lines)
            self.assertTrue(all(gap >= min_px for gap in gaps), gaps)

    def test_a_still_pose_is_a_still_frame_and_a_small_move_a_small_change(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi / 2)})
        f = self._render()
        first = f.light_img.copy()
        edges = 2 * (len(_runs(f.white)) + len(_runs(f.blue)))
        np.testing.assert_array_equal(self._render().light_img, first)
        self._people({0: _pose(0.5, shoulders=math.pi / 2 + 0.03)})      # lines ~1.3 px wider
        moved = self._render().light_img
        changed = int(np.count_nonzero(moved != first))
        self.assertGreater(changed, 0)
        self.assertLessEqual(changed, edges)                    # each line edge shifts a pixel at most: nothing pops

    def test_every_band_masks_every_pattern(self) -> None:
        b = round(0.52 * IRES)
        self._people({0: _pose(0.5, shoulders=math.pi), 1: _pose(0.52)})    # B inside A's white reach
        f = self._render()
        np.testing.assert_array_equal(f.white[b - BAND:b + BAND + 1], 0.0)
        np.testing.assert_allclose(f.blue[b - BAND:b + BAND + 1], self.cfg.band_level, atol=1e-6)
        self.assertEqual(float(f.white[b + BAND + 5]), 1.0)                 # A's white continues past B

    # -- the hit --

    def _inner_white_runs(self, f: Frame) -> list[int]:
        return [length for start, length in _runs(f.white) if C + 50 < start and start + length < C + REACH - 50]

    def test_the_crossing_tick_widens_every_line_once(self) -> None:
        widen = int(self.cfg.hit_widen * DEG)
        lengths = []
        for offset in (23.4, 1.8, -2.0, -9.2):                 # 36 rpm at 30 Hz: 7.2° a tick, closest at +1.8°
            self._people({0: _pose(0.5, shoulders=math.pi / 2, offset_deg=offset)})
            lengths.append(set(self._inner_white_runs(self._render())))
        self.assertEqual(lengths, [{70}, {70 + 2 * widen}, {70}, {70}])

    def test_reset_starts_a_new_pass(self) -> None:
        self._people({0: _pose(0.5, shoulders=math.pi / 2, offset_deg=1.8)})
        self._render()
        self.layer.reset()
        self.assertEqual(set(self._inner_white_runs(self._render())), {70 + 2 * int(self.cfg.hit_widen * DEG)})

    # -- sync --

    def test_sync_fills_the_arc_between_the_pair_above_threshold_only(self) -> None:
        a, b = round(0.3 * IRES), round(0.7 * IRES)
        self._people({0: _pose(0.3, sims={1: 0.5}), 1: _pose(0.7, sims={0: 0.5})})
        self.assertEqual(float(self._render().blue[C]), 0.0)
        self.layer.reset()
        self._people({0: _pose(0.3, sims={1: 1.0}), 1: _pose(0.7, sims={0: 1.0})})
        f = self._render()
        np.testing.assert_array_equal(f.blue[a + BAND + 1:b - BAND], 1.0)
        self.assertEqual(float(f.blue[b + REACH + 1:].sum()), 0.0)             # never past the partner's reach

    def test_sync_reaches_over_an_intermediate_person(self) -> None:
        probe = round(0.35 * IRES) + 50         # beyond both base reaches, inside the arms-up person's
        self._people({0: _pose(0.2), 1: _pose(0.35, shoulders=math.pi), 2: _pose(0.5)})
        self.assertEqual(float(self._render().blue[probe]), 0.0)
        self.layer.reset()
        self._people({0: _pose(0.2, sims={2: 1.0}), 1: _pose(0.35, shoulders=math.pi), 2: _pose(0.5, sims={0: 1.0})})
        f = self._render()
        self.assertEqual(float(f.blue[probe]), 1.0)
        self.assertEqual(float(f.white[probe]), 1.0)             # the intermediate pattern unions in

    def test_sync_takes_the_shorter_arc_across_the_wrap(self) -> None:
        self._people({0: _pose(0.95, sims={1: 1.0}), 1: _pose(0.05, sims={0: 1.0})})
        f = self._render()
        self.assertEqual(float(f.blue[0]), 1.0)
        self.assertEqual(float(f.blue[round(0.2 * IRES):round(0.8 * IRES)].sum()), 0.0)

    # -- presence --

    def test_attack_grows_the_reach_from_the_band(self) -> None:
        self.cfg.attack_seconds = 1.0
        self._people({0: _pose(0.5)})
        first = self._render()
        self.assertEqual(float(first.blue[C + BAND + 1:].sum()), 0.0)     # the reach is still inside the band
        for _ in range(30):
            last = self._render()
        self.assertEqual(float(last.blue[C + 300]), 1.0)

    def test_release_shrinks_the_reach_then_everything_goes_and_reset_clears(self) -> None:
        self.cfg.release_seconds = 1.0
        self._people({0: _pose(0.5)})
        self._render()
        self._people({})                                        # gone
        held = self._render()
        self.assertEqual(float(held.blue[C + 300]), 1.0)
        self.assertEqual(float(held.blue[C + REACH]), 0.0)      # the reach shrinks
        self.assertLess(float(held.blue[C]), self.cfg.band_level)
        for _ in range(40):
            last = self._render()
        self.assertEqual(float(last.light_img.sum()), 0.0)
        self._people({0: _pose(0.5)})
        self._render()
        self._people({})
        self.layer.reset()
        self.assertEqual(float(self._render().light_img.sum()), 0.0)

    def test_nobody_with_a_pose_draws_nothing(self) -> None:
        f = self._render()
        self.assertEqual(float(f.light_img.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
