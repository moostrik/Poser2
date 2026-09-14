"""Tests for the show layers: beam_blue_sound (levels → blue lamps, jitter window, stale
fallback), wind_down (the dying wall), and the pose_instrument (the connections, mirror-symmetric
on/off line patterns, the mask, the visual limit, the hit, sync, presence)."""

import math
import unittest
from time import monotonic
from types import SimpleNamespace

import numpy as np

from modules.board import SoundLevels
from modules.pose import features

from apps.white_space.light import Tick, BeamLightId, MotorCommand, MotorMode, LayerSettings
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers.beam.blue_sound import BeamBlueSound, BeamBlueSoundSettings, SoundFallback
from apps.white_space.light.layers import PoseInstrument, PoseInstrumentSettings
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
MASK = 15                   # mask half width (px) at the default 3° and pose length 1
WINDOW = 450                # default 45° window (px)
INTERVAL = 140              # the default 14° interval (px)

# The joints' rests and reaches as the settings default them; a measure is a fraction of the reach.
SHOULDER_REST, SHOULDER_REACH = 0.18 * math.pi, math.pi
ELBOW_REST, ELBOW_REACH = -0.10 * math.pi, math.pi


def shoulder(fraction: float) -> float:
    return SHOULDER_REST + fraction * SHOULDER_REACH


def elbow(fraction: float) -> float:
    return ELBOW_REST + fraction * ELBOW_REACH


class FakePose:
    """A fake pose frame: ``pose[FeatureType]`` over a prepared feature dict."""

    def __init__(self, by_type: dict) -> None:
        self._by_type = by_type

    def __getitem__(self, feature_type):
        return self._by_type[feature_type]


def _pose(azimuth_pos: float, sims: dict[int, float] | None = None, shoulders: float = shoulder(0.0),
          elbows: float = elbow(0.0), legs: float = 0.0, tilt: float = 0.0,
          offset_deg: float = float("nan"), left_shoulder: float | None = None,
          right_shoulder: float | None = None, left_elbow: float | None = None,
          right_elbow: float | None = None) -> FakePose:
    """A fake pose at normalized azimuth ``azimuth_pos`` (0..1) with the given arm angles
    (radians; ``shoulders`` / ``elbows`` set both sides, a side overrides), leg deviation, torso
    tilt, pairwise sims and playhead offset."""
    angles = np.full(len(features.AngleLandmark), np.nan)
    angles[features.AngleLandmark.left_shoulder]  = shoulders if left_shoulder is None else left_shoulder
    angles[features.AngleLandmark.right_shoulder] = shoulders if right_shoulder is None else right_shoulder
    angles[features.AngleLandmark.left_elbow]     = elbows if left_elbow is None else left_elbow
    angles[features.AngleLandmark.right_elbow]    = elbows if right_elbow is None else right_elbow
    sim_values = np.full(16, np.nan)
    for j, v in (sims or {}).items():
        sim_values[j] = v
    return FakePose({
        features.Azimuth: SimpleNamespace(value=azimuth_pos * math.tau),
        features.BBox: {features.BBoxElement.height: 1.0},
        features.Angles: SimpleNamespace(values=angles),
        features.AngleSymmetry: SimpleNamespace(values=np.full(len(features.SymmetryElement), np.nan)),
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


class PoseInstrumentTest(unittest.TestCase):
    # Line centres a quarter pixel off the grid, so no line edge falls exactly on a pixel: the
    # blue's rest phase carries it, the white's comes from a barely folded left elbow.
    QUARTER = 0.25 / INTERVAL
    ELBOW_QUARTER = elbow(2.0 * QUARTER)      # phase = measure × phase_range (½) → QUARTER intervals
    # The fundamental alone, fully out: plain white lines half the interval wide.
    FUNDAMENTAL_OUT = dict(left_shoulder=shoulder(1.0), right_shoulder=shoulder(0.0),
                           left_elbow=ELBOW_QUARTER, right_elbow=elbow(0.0))
    # A T: both drawbars half out, the elbows straight.
    T = dict(shoulders=shoulder(0.5), left_elbow=ELBOW_QUARTER, right_elbow=elbow(0.0))

    def setUp(self) -> None:
        self.cfg = PoseInstrumentSettings()
        self.cfg.presence.attack_seconds = 0.0            # present at once — geometry tests read one frame
        self.cfg.pattern.blue_phase = 0.5 + self.QUARTER
        self.board = InstrumentBoard(frames={})
        self.layer = PoseInstrument(IRES, LayerSettings(), self.cfg, self.board, pose_stage=4, tick_interval=TICK)

    def _people(self, poses: dict[int, FakePose]) -> None:
        self.board.frames = poses

    def _render(self) -> Frame:
        f = Frame(IRES, Tick(0.0, TICK), motor_command=MotorCommand(mode=MotorMode.PROJECTION, beam_rpm=36.0))
        self.layer.render(f)
        return f

    def _outside_mask(self, centre: int = C) -> np.ndarray:
        keep = np.ones(IRES, dtype=bool)
        keep[centre - MASK:centre + MASK + 1] = False
        return keep

    def _connect(self, pose: FakePose):
        self._people({0: pose})
        self._render()
        return self.layer.connect(self.layer._participants[0])

    # -- the connections --

    def test_the_shoulders_are_the_drawbars_white_out_and_blue_in(self) -> None:
        p = self._connect(_pose(0.5, left_shoulder=shoulder(1.0), right_shoulder=shoulder(0.0)))
        self.assertAlmostEqual(p.white.fundamental, 1.0, places=5)
        self.assertAlmostEqual(p.white.harmonic, 0.0, places=5)
        self.assertAlmostEqual(p.blue.fundamental, 0.0, places=5)
        self.assertAlmostEqual(p.blue.harmonic, 1.0, places=5)
        p = self._connect(_pose(0.5, left_shoulder=shoulder(0.0), right_shoulder=shoulder(0.5)))
        self.assertAlmostEqual(p.white.fundamental, 0.0, places=5)
        self.assertAlmostEqual(p.white.harmonic, 0.5, places=5)

    def test_an_arm_past_the_vertical_stays_fully_out(self) -> None:
        # The shoulder wraps at π: straight up reads as −0.82π, not +1.18π; a little further is still out.
        for angle in (shoulder(1.0), shoulder(1.0) - math.tau, shoulder(1.1) - math.tau):
            self.assertAlmostEqual(self._connect(_pose(0.5, left_shoulder=angle)).white.fundamental, 1.0, places=5)
        self.assertEqual(self._connect(_pose(0.5, left_shoulder=shoulder(-0.05))).white.fundamental, 0.0)   # a little behind

    def test_the_elbows_place_the_lines_and_the_overtone(self) -> None:
        p = self._connect(_pose(0.5, shoulders=shoulder(0.5), left_elbow=elbow(1.0), right_elbow=elbow(0.5)))
        self.assertAlmostEqual(p.white.phase, self.cfg.pattern.phase_range, places=5)
        self.assertAlmostEqual(p.white.overtone_phase, 0.5 * self.cfg.pattern.overtone_phase_range, places=5)
        self.assertAlmostEqual(p.blue.phase, self.cfg.pattern.blue_phase, places=5)
        self.assertEqual(p.blue.overtone_phase, 0.0)

    def test_the_legs_detune_and_the_body_bends_the_interval(self) -> None:
        rest = self.cfg.pattern.interval
        self.assertAlmostEqual(self._connect(_pose(0.5)).interval, rest, places=5)
        self.assertAlmostEqual(self._connect(_pose(0.5, tilt=1.0)).interval, rest * 2.0 ** self.cfg.pattern.octaves, places=5)
        self.assertAlmostEqual(self._connect(_pose(0.5, tilt=-1.0)).interval, rest / 2.0 ** self.cfg.pattern.octaves, places=5)
        self.assertEqual(self._connect(_pose(0.5)).detune, 0.0)
        self.assertAlmostEqual(self._connect(_pose(0.5, legs=1.0)).detune, self.cfg.pattern.detune, places=5)

    # -- one person --

    def test_arms_hanging_is_full_blue_with_the_dim_mask(self) -> None:
        self._people({0: _pose(0.5)})
        f = self._render()
        self.assertEqual(float(f.white.sum()), 0.0)
        np.testing.assert_array_equal(f.blue[C + MASK + 1:C + WINDOW + 1], 1.0)
        np.testing.assert_array_equal(f.blue[C - WINDOW:C - MASK], 1.0)
        np.testing.assert_allclose(f.blue[C - MASK:C + MASK + 1], self.cfg.mask.brightness, atol=1e-6)
        self.assertEqual(float(f.blue[C + WINDOW + 1:].sum() + f.blue[:C - WINDOW].sum()), 0.0)

    def test_arms_up_is_full_white_and_blue_only_in_the_mask(self) -> None:
        self._people({0: _pose(0.5, shoulders=shoulder(1.0))})
        f = self._render()
        np.testing.assert_array_equal(f.white[C + MASK + 1:C + WINDOW + 1], 1.0)
        np.testing.assert_array_equal(f.white[C - WINDOW:C - MASK], 1.0)
        np.testing.assert_array_equal(f.white[C - MASK:C + MASK + 1], 0.0)    # the mask goes over white
        self.assertEqual(float(f.blue[self._outside_mask()].sum()), 0.0)

    def test_in_between_the_lines_are_full_and_mirror_symmetric(self) -> None:
        self._people({0: _pose(0.5, shoulders=shoulder(0.5), elbows=self.ELBOW_QUARTER)})     # both drawbars half out
        f = self._render()
        outside = self._outside_mask()
        self.assertTrue(np.isin(f.white, (0.0, 1.0)).all())
        self.assertTrue(np.isin(f.blue[outside], (0.0, 1.0)).all())
        self.assertGreater(len(_runs(f.white)), 4)
        self.assertGreater(len(_runs(f.blue)), 4)
        span = WINDOW + 10
        for channel in (f.white, f.blue):
            np.testing.assert_array_equal(channel[C + 1:C + span], channel[C - 1:C - span:-1])

    @staticmethod
    def _inner(channel: np.ndarray) -> list[tuple[int, int]]:
        """(start, length) of the right side's runs clear of the mask and the window edge, which cut lines."""
        return [(s, l) for s, l in _runs(channel) if C + MASK + 1 < s and s + l < C + WINDOW]

    def _inner_white_runs(self, f: Frame) -> list[int]:
        return [length for _, length in self._inner(f.white)]

    @staticmethod
    def _centres(inner: list[tuple[int, int]]) -> list[float]:
        return [s + (l - 1) / 2 - C for s, l in inner]

    def _on_grid(self, inner: list[tuple[int, int]], spacing: float, offset: float = 0.0) -> None:
        """Every run's centre sits at ``offset`` modulo ``spacing`` from the person, to a pixel."""
        for c in self._centres(inner):
            d = (c - offset) % spacing
            self.assertLessEqual(min(d, spacing - d), 1.0, f"centre {c} off the grid of {spacing} at {offset}")

    def _rows(self, pose: FakePose) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """The inner white and blue runs of one person at the centre."""
        self._people({0: pose})
        f = self._render()
        return self._inner(f.white), self._inner(f.blue)

    def test_the_fundamental_half_out_lights_a_third_of_the_interval(self) -> None:
        self._people({0: _pose(0.5, **dict(self.FUNDAMENTAL_OUT, left_shoulder=shoulder(0.5)))})
        inner = self._inner_white_runs(self._render())
        self.assertTrue(inner)
        self.assertEqual(set(inner), {round(INTERVAL / 3)})

    # -- the pose results (POSE_INSTRUMENT.md, Part 3): one test per row --

    def test_row_arms_hanging_is_the_blue_ping(self) -> None:
        self._people({0: _pose(0.5)})
        f = self._render()
        self.assertEqual(float(f.white.sum()), 0.0)
        np.testing.assert_array_equal(f.blue[C + MASK + 1:C + WINDOW + 1], 1.0)

    def test_row_both_arms_up_is_the_bass(self) -> None:
        self._people({0: _pose(0.5, shoulders=shoulder(1.0))})
        f = self._render()
        np.testing.assert_array_equal(f.white[C + MASK + 1:C + WINDOW + 1], 1.0)
        self.assertEqual(float(f.blue[self._outside_mask()].sum()), 0.0)

    def test_row_a_t_is_half_registration(self) -> None:
        # Both drawbars half out: white lines a third of the interval wide, the blue's between them
        # at the blue's rest phase; the sub-line at half registration is a point that shows nothing.
        white, blue = self._rows(_pose(0.5, **self.T))
        self.assertGreater(len(white), 2)
        self.assertEqual({l for _, l in white}, {47})
        self.assertEqual({l for _, l in blue}, {47})
        self._on_grid(white, INTERVAL)
        self._on_grid(blue, INTERVAL, INTERVAL / 2)

    def test_row_left_arm_up_is_the_fundamental_alone(self) -> None:
        white, blue = self._rows(_pose(0.5, **self.FUNDAMENTAL_OUT))
        self.assertGreaterEqual(len(white), 2)
        self.assertEqual({l for _, l in white}, {INTERVAL // 2})                     # thick lines
        self._on_grid(white, INTERVAL)                                               # one per interval
        self.assertEqual({l for _, l in blue}, {INTERVAL // 4})                      # the blue's harmonic alone

    def test_row_right_arm_up_is_the_harmonic_alone(self) -> None:
        pose = _pose(0.5, left_shoulder=shoulder(0.0), right_shoulder=shoulder(1.0),
                     left_elbow=self.ELBOW_QUARTER, right_elbow=self.ELBOW_QUARTER)
        white, blue = self._rows(pose)
        self.assertGreater(len(white), 4)
        self.assertEqual({l for _, l in white}, {INTERVAL // 4})                      # thin sub-lines
        self._on_grid(white, INTERVAL / 2)                                            # two per interval
        self.assertEqual({l for _, l in blue}, {INTERVAL // 2})                       # the blue's fundamental alone

    def test_row_a_t_with_both_elbows_folded(self) -> None:
        # The lines move out half an interval; the overtone against the fundamental thickens them.
        white, _ = self._rows(_pose(0.5, shoulders=shoulder(0.5), elbows=elbow(1.0)))
        self.assertGreater(len(white), 1)
        self.assertEqual({l for _, l in white}, {93})
        self._on_grid(white, INTERVAL, INTERVAL / 2)

    def test_row_a_t_with_the_left_elbow_folded(self) -> None:
        # The lines move out half an interval, the shape unchanged.
        white, _ = self._rows(_pose(0.5, shoulders=shoulder(0.5), left_elbow=elbow(1.0), right_elbow=elbow(0.0)))
        self.assertGreater(len(white), 1)
        self.assertEqual({l for _, l in white}, {47})
        self._on_grid(white, INTERVAL, INTERVAL / 2)

    def test_row_a_t_with_the_right_elbow_folded(self) -> None:
        # The overtone moves against the fundamental: the lines stay in place and thicken.
        white, _ = self._rows(_pose(0.5, shoulders=shoulder(0.5), left_elbow=self.ELBOW_QUARTER, right_elbow=elbow(1.0)))
        self.assertGreater(len(white), 1)
        self.assertEqual({l for _, l in white}, {93})
        self._on_grid(white, INTERVAL)

    def test_row_a_t_leaning_bends_the_interval_and_back(self) -> None:
        white, _ = self._rows(_pose(0.5, tilt=1.0, **self.T))
        self.assertGreater(len(white), 0)
        self.assertLessEqual({l for _, l in white}, {93, 94})                          # a third of a doubled interval
        self._on_grid(white, 2 * INTERVAL)
        white, _ = self._rows(_pose(0.5, tilt=0.0, **self.T))
        self.assertEqual({l for _, l in white}, {47})

    def test_row_a_t_in_a_crouch_detunes_the_blue(self) -> None:
        white, blue = self._rows(_pose(0.5, legs=1.0, **self.T))
        detuned = round(INTERVAL * (1.0 + self.cfg.pattern.detune))
        self.assertGreater(len(blue), 1)
        self.assertEqual({round(b - a) for a, b in zip(self._centres(white), self._centres(white)[1:])}, {INTERVAL})
        self.assertEqual({round(b - a) for a, b in zip(self._centres(blue), self._centres(blue)[1:])}, {detuned})

    def test_no_line_or_gap_is_narrower_than_the_visual_limit(self) -> None:
        # Two overlapping synced patterns of different intervals (one leans): the union is a moiré.
        b = round(0.537 * IRES)
        self.cfg.window.width = 150.0                           # a long overlap: plenty of interior
        window = 1500
        arms = dict(left_shoulder=shoulder(0.4), right_shoulder=shoulder(0.3), right_elbow=elbow(0.4))
        self._people({0: _pose(0.5, sims={1: 1.0}, **arms), 1: _pose(0.537, sims={0: 1.0}, tilt=0.4, **arms)})
        f = self._render()
        min_px = round(IRES / (2 * self.cfg.max_lines))
        # Masks and window edges may cut a line (it slides out): features touching them are exempt.
        cuts = [(C - MASK, C + MASK), (b - MASK, b + MASK)] + [(e, e) for e in (C - window, b - window, C + window, b + window)]

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
        nearly_out = dict(self.FUNDAMENTAL_OUT, left_shoulder=shoulder(0.9))
        self._people({0: _pose(0.5, **nearly_out)})
        f = self._render()
        first = f.light_img.copy()
        edges = 2 * (len(_runs(f.white)) + len(_runs(f.blue)))
        np.testing.assert_array_equal(self._render().light_img, first)
        self._people({0: _pose(0.5, **dict(nearly_out, left_shoulder=shoulder(0.9) + 0.1))})   # lines ~1 px wider
        moved = self._render().light_img
        changed = int(np.count_nonzero(moved != first))
        self.assertGreater(changed, 0)
        self.assertLessEqual(changed, 2 * edges)                # each line edge shifts a pixel or two: nothing pops

    def test_every_mask_goes_over_every_pattern(self) -> None:
        b = round(0.52 * IRES)
        self._people({0: _pose(0.5, shoulders=shoulder(1.0)), 1: _pose(0.52)})    # B inside A's white window
        f = self._render()
        np.testing.assert_array_equal(f.white[b - MASK:b + MASK + 1], 0.0)
        np.testing.assert_allclose(f.blue[b - MASK:b + MASK + 1], self.cfg.mask.brightness, atol=1e-6)
        self.assertEqual(float(f.white[b + MASK + 5]), 1.0)                 # A's white continues past B

    # -- the hit --

    def test_the_crossing_tick_widens_every_line_once(self) -> None:
        widen = int(self.cfg.events.hit_widen * DEG)
        lengths = []
        for offset in (23.4, 1.8, -2.0, -9.2):                 # 36 rpm at 30 Hz: 7.2° a tick, closest at +1.8°
            self._people({0: _pose(0.5, offset_deg=offset, **self.FUNDAMENTAL_OUT)})
            lengths.append(set(self._inner_white_runs(self._render())))
        self.assertEqual(lengths, [{70}, {70 + 2 * widen}, {70}, {70}])

    def test_reset_starts_a_new_pass(self) -> None:
        self._people({0: _pose(0.5, offset_deg=1.8, **self.FUNDAMENTAL_OUT)})
        self._render()
        self.layer.reset()
        self.assertEqual(set(self._inner_white_runs(self._render())), {70 + 2 * int(self.cfg.events.hit_widen * DEG)})

    # -- sync --

    def test_sync_fills_the_arc_between_the_pair_above_threshold_only(self) -> None:
        a, b = round(0.3 * IRES), round(0.7 * IRES)
        self._people({0: _pose(0.3, sims={1: 0.5}), 1: _pose(0.7, sims={0: 0.5})})
        self.assertEqual(float(self._render().blue[C]), 0.0)
        self.layer.reset()
        self._people({0: _pose(0.3, sims={1: 1.0}), 1: _pose(0.7, sims={0: 1.0})})
        f = self._render()
        np.testing.assert_array_equal(f.blue[a + MASK + 1:b - MASK], 1.0)
        self.assertEqual(float(f.blue[b + WINDOW + 1:].sum()), 0.0)             # never past the partner's window

    def test_sync_reaches_over_an_intermediate_person(self) -> None:
        probe = round(0.35 * IRES) + 50         # beyond both base windows, inside the arms-up person's
        self._people({0: _pose(0.2), 1: _pose(0.35, shoulders=shoulder(1.0)), 2: _pose(0.5)})
        self.assertEqual(float(self._render().blue[probe]), 0.0)
        self.layer.reset()
        self._people({0: _pose(0.2, sims={2: 1.0}), 1: _pose(0.35, shoulders=shoulder(1.0)), 2: _pose(0.5, sims={0: 1.0})})
        f = self._render()
        self.assertEqual(float(f.blue[probe]), 1.0)
        self.assertEqual(float(f.white[probe]), 1.0)             # the intermediate pattern unions in

    def test_sync_takes_the_shorter_arc_across_the_wrap(self) -> None:
        self._people({0: _pose(0.95, sims={1: 1.0}), 1: _pose(0.05, sims={0: 1.0})})
        f = self._render()
        self.assertEqual(float(f.blue[0]), 1.0)
        self.assertEqual(float(f.blue[round(0.2 * IRES):round(0.8 * IRES)].sum()), 0.0)

    # -- presence --

    def test_attack_opens_the_window_from_the_mask(self) -> None:
        self.cfg.presence.attack_seconds = 1.0
        self._people({0: _pose(0.5)})
        first = self._render()
        self.assertEqual(float(first.blue[C + MASK + 1:].sum()), 0.0)     # the window is still inside the mask
        for _ in range(30):
            last = self._render()
        self.assertEqual(float(last.blue[C + 300]), 1.0)

    def test_release_closes_the_window_then_everything_goes_and_reset_clears(self) -> None:
        self.cfg.presence.release_seconds = 1.0
        self._people({0: _pose(0.5)})
        self._render()
        self._people({})                                        # gone
        held = self._render()
        self.assertEqual(float(held.blue[C + 300]), 1.0)
        self.assertEqual(float(held.blue[C + WINDOW]), 0.0)     # the window closes
        self.assertLess(float(held.blue[C]), self.cfg.mask.brightness)
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
