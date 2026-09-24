"""Tests for the pose instrument, the bridge between the pose data and the light synth: the
connections, the two fixed points, the window's taper, the mask, the hit (the push and the flash),
playing by hand, sync and presence. The synth itself is tested in test_synth_*.py."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from modules.pose import features

from apps.white_space.light import Tick, MotorCommand, MotorMode, LayerSettings
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers import PoseInstrument, PoseInstrumentSettings
from apps.white_space.light.synth import Parameter, Curve
from apps.white_space.pose import PlayheadOffset

IRES = 3600                 # one pixel per 0.1°
C = IRES // 2               # the pixel of a person at normalized azimuth 0.5
TICK = 1 / 30
MASK = 15                   # mask half width (px) at the default 3°
REACH = 450                 # the default 45° reach (px)
FULL = 360                  # the window is full up to the taper, the last fifth of the reach
INTERVAL = 140              # the default 14° interval (px)
SOLID = FULL - INTERVAL // 2   # a solid output is lines that touch: whole ones end within half an interval of the taper


def shoulder(fraction: float) -> float:
    """A shoulder angle as a fraction of the way from neutral (0) to raised (π)."""
    return fraction * math.pi


class FakePose:
    """A fake pose frame: ``pose[FeatureType]`` over a prepared feature dict."""

    def __init__(self, by_type: dict) -> None:
        self._by_type = by_type

    def __getitem__(self, feature_type):
        return self._by_type[feature_type]


def _pose(azimuth_pos: float, sims: dict[int, float] | None = None, left_shoulder: float = 0.0,
          right_shoulder: float | None = None, left_elbow: float = 0.0, right_elbow: float = 0.0,
          legs: float = 0.0, tilt: float = 0.0, distance: float = 0.0,
          offset_deg: float = float("nan")) -> FakePose:
    """A fake pose at normalized azimuth ``azimuth_pos`` (0..1) with the arm angles (radians; the
    right shoulder follows the left unless given), leg deviation, body bend, distance, pairwise
    sims and playhead offset. The arm travel is the angles over π, no dead zones, as the
    pipeline's extractor gives it with zero zones."""
    angles = np.full(len(features.AngleLandmark), np.nan)
    angles[features.AngleLandmark.left_shoulder] = left_shoulder
    angles[features.AngleLandmark.right_shoulder] = left_shoulder if right_shoulder is None else right_shoulder
    angles[features.AngleLandmark.left_elbow] = left_elbow
    angles[features.AngleLandmark.right_elbow] = right_elbow
    travel = angles[:len(features.TravelElement)] / math.pi
    sim_values = np.full(16, np.nan)
    for j, v in (sims or {}).items():
        sim_values[j] = v
    return FakePose({
        features.Azimuth: SimpleNamespace(value=azimuth_pos * math.tau),
        features.Angles: SimpleNamespace(values=angles),
        features.ArmTravel: SimpleNamespace(values=travel),
        features.AngleSymmetry: SimpleNamespace(values=np.full(len(features.SymmetryElement), np.nan)),
        features.Similarity: SimpleNamespace(values=sim_values),
        features.LegDeviation: SimpleNamespace(value=legs),
        features.TorsoTilt: SimpleNamespace(value=tilt),
        features.Distance: SimpleNamespace(value=distance),
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
    # Line centres a quarter pixel off the grid, so no line edge falls exactly on a pixel.
    QUARTER = 0.25 / INTERVAL
    HALFWAY = shoulder(0.5)             # both pulse widths a half: white and blue lines alternate

    def setUp(self) -> None:
        self.cfg = PoseInstrumentSettings()
        self.cfg.window.attack_seconds = 0.0              # present at once — geometry tests read one frame
        W, B = self.cfg.white_lines, self.cfg.blue_lines              # the placeholder's patch: white out, blue in
        W.pulse_width, W.pulse_width_amount, W.phase = 0.0, 1.0, self.QUARTER
        B.pulse_width, B.pulse_width_amount, B.phase = 1.0, -1.0, -0.5 + self.QUARTER
        self.board = InstrumentBoard(frames={})
        self.layer = PoseInstrument(IRES, LayerSettings(), self.cfg, self.board, pose_stage=4)
        self.dt = TICK                                    # the tick's interval; the strobe tests run at 32 fps
        self.tick = 0                                     # the clock's tick index, counted by _render

    def _people(self, poses: dict[int, FakePose]) -> None:
        self.board.frames = poses

    def _render(self, playhead: float = float("nan")) -> Frame:
        """One tick; ``playhead`` is the content playhead as a normalized azimuth, none by
        default so the marker stays out of the picture."""
        f = Frame(IRES, Tick(0.0, self.dt, index=self.tick), motor_command=MotorCommand(mode=MotorMode.PROJECTION, beam_rpm=36.0),
                  playhead=playhead * math.tau)
        self.tick += 1
        self.layer.render(f)
        return f

    def _outside_mask(self, centre: int = C) -> np.ndarray:
        keep = np.ones(IRES, dtype=bool)
        keep[centre - MASK:centre + MASK + 1] = False
        return keep

    @staticmethod
    def _right(channel: np.ndarray) -> list[tuple[int, int]]:
        """(start, length) of the runs on the person's right, clear of the mask."""
        return [(s, l) for s, l in _runs(channel) if C + MASK + 1 < s]

    @staticmethod
    def _left(channel: np.ndarray) -> list[tuple[int, int]]:
        """(start, length) of the runs on the person's left, clear of the mask."""
        return [(s, l) for s, l in _runs(channel) if s + l < C - MASK]

    @classmethod
    def _inner(cls, channel: np.ndarray) -> list[tuple[int, int]]:
        """The right side's runs in the window's full part: clear of the mask and the taper."""
        return [(s, l) for s, l in cls._right(channel) if s + l < C + FULL]

    @staticmethod
    def _centres(runs: list[tuple[int, int]]) -> list[float]:
        return [s + (l - 1) / 2 - C for s, l in runs]

    def _on_grid(self, runs: list[tuple[int, int]], spacing: float, offset: float = 0.0) -> None:
        """Every run's centre sits at ``offset`` modulo ``spacing`` from the person, to a pixel."""
        for c in self._centres(runs):
            d = (c - offset) % spacing
            self.assertLessEqual(min(d, spacing - d), 1.0, f"centre {c} off the grid of {spacing} at {offset}")

    # -- the connections --

    def _connect(self, pose: FakePose):
        self._people({0: pose})
        self._render()
        return self.layer.connect(self.layer._players[0])

    def test_the_shoulders_play_both_colours_and_each_elbow_its_own(self) -> None:
        self.cfg.breath.rate, self.cfg.breath.depth = 0.0, 0.4                          # the breath held at its crest, 1
        white, blue = self._connect(_pose(0.5, left_shoulder=shoulder(0.25), right_shoulder=shoulder(0.75),
                                          left_elbow=shoulder(0.5), right_elbow=shoulder(1.0)))
        gap = 0.4 * 0.5                                                              # the depth times the shoulders' gap
        self.assertAlmostEqual(white[Parameter.PULSE_WIDTH], 0.5 + gap, places=5)    # the mean, swung by the breath
        self.assertAlmostEqual(blue[Parameter.PULSE_WIDTH], 0.5 + gap, places=5)     # the right higher: complementary
        self.assertAlmostEqual(white[Parameter.PITCH], 0.5, places=5)                # the left elbow: the white
        self.assertAlmostEqual(blue[Parameter.PITCH], 1.0, places=5)                 # the right elbow: the blue

    def test_each_elbows_turn_is_its_colours_phase(self) -> None:
        for degrees, turn in ((90.0, 1.0), (-90.0, -1.0), (0.0, 0.0), (180.0, 0.0), (-180.0, 0.0), (30.0, 0.5)):
            white, blue = self._connect(_pose(0.5, left_elbow=math.radians(degrees), right_elbow=-math.radians(degrees)))
            self.assertAlmostEqual(white[Parameter.PHASE], turn, places=5, msg=f"{degrees}°")    # the left elbow: the white
            self.assertAlmostEqual(blue[Parameter.PHASE], -turn, places=5, msg=f"{degrees}°")    # the right elbow: the blue

    def test_the_turn_is_continuous_where_the_elbow_folds_through_180(self) -> None:
        a = self._connect(_pose(0.5, left_elbow=math.radians(179.0)))[0][Parameter.PHASE]
        b = self._connect(_pose(0.5, left_elbow=math.radians(-179.0)))[0][Parameter.PHASE]
        self.assertLess(abs(a - b), 0.04)

    def test_the_body_bend_alone_is_both_speeds(self) -> None:
        for tilt in (-1.0, 0.0, 0.4):
            white, blue = self._connect(_pose(0.5, tilt=tilt, left_elbow=math.radians(30.0)))
            self.assertAlmostEqual(white[Parameter.SPEED], tilt, places=5, msg=f"tilt {tilt}")   # the turn plays no speed
            self.assertAlmostEqual(blue[Parameter.SPEED], tilt, places=5, msg=f"tilt {tilt}")

    def test_a_straight_body_stands_still_and_a_lean_carries_the_lines(self) -> None:
        W, B = self.cfg.white_lines, self.cfg.blue_lines
        W.speed = B.speed = 0.0                                        # the preset's base: nothing travels by itself
        W.speed_amount = B.speed_amount = 3.5                          # a full lean: a quarter interval a second
        for tilt, moved in ((0.0, 0), (1.0, 35), (-1.0, -35)):         # px after a second
            self.layer.reset()
            self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, tilt=tilt)})
            for _ in range(30):
                f = self._render()
            self._on_grid(self._inner(f.white), INTERVAL, moved % INTERVAL)
            self._on_grid(self._inner(f.blue), INTERVAL, (INTERVAL / 2 + moved) % INTERVAL)

    def test_the_sign_of_a_travel_is_not_a_measure(self) -> None:
        # The sign is the side of the body the arm passes; straight up is π from either side.
        for angle in (shoulder(0.5), -shoulder(0.5)):
            self.assertAlmostEqual(self._connect(_pose(0.5, left_shoulder=angle))[0][Parameter.PULSE_WIDTH], 0.5, places=5)

    def test_the_legs_are_the_lfos_level_as_they_come(self) -> None:
        for legs in (0.0, 0.525, 1.0):
            self._connect(_pose(0.5, legs=legs))
            self.assertAlmostEqual(self.layer.connect_lfo(self.layer._players[0]), legs, places=5, msg=f"legs {legs}")

    def test_the_distance_is_a_source_that_plays_nothing_yet(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, distance=0.0)})
        near = self._render().light_img.copy()
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, distance=1.0)})
        far = self._render().light_img
        self.assertEqual(self.layer._players[0].distance, 1.0)              # there for `connect` to use
        np.testing.assert_array_equal(far, near)                            # unconnected: the picture is the same
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, distance=float("nan"))})
        self._render()
        self.assertEqual(self.layer._players[0].distance, 1.0)              # no reading: the last one holds

    # -- the two fixed points --

    def test_arms_hanging_is_full_blue_with_the_dim_mask(self) -> None:
        self._people({0: _pose(0.5)})
        f = self._render()
        self.assertEqual(float(f.white.sum()), 0.0)
        np.testing.assert_array_equal(f.blue[C + MASK + 1:C + SOLID + 1], 1.0)
        np.testing.assert_array_equal(f.blue[C - SOLID:C - MASK], 1.0)
        np.testing.assert_allclose(f.blue[C - MASK:C + MASK + 1], self.cfg.mask.blue, atol=1e-6)
        self.assertEqual(float(f.blue[C + REACH + 1:].sum() + f.blue[:C - REACH].sum()), 0.0)

    def test_arms_up_is_full_white_and_blue_only_in_the_mask(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0))})
        f = self._render()
        np.testing.assert_array_equal(f.white[C + MASK + 1:C + SOLID + 1], 1.0)
        np.testing.assert_array_equal(f.white[C - SOLID:C - MASK], 1.0)
        np.testing.assert_array_equal(f.white[C - MASK:C + MASK + 1], 0.0)    # the mask goes over white
        self.assertEqual(float(f.blue[self._outside_mask()].sum()), 0.0)

    # -- the breath: the shoulders' gap, in step or complementary --

    BREATH = 60                                          # ticks in one breath at 0.5 Hz

    def _widths(self, left: float, right: float) -> tuple[list[float], list[float]]:
        """Both colours' width sources over one breath, for the shoulders at ``left`` and ``right``."""
        self.layer.reset()
        white, blue = [], []
        for _ in range(self.BREATH):
            w, b = self._connect(_pose(0.5, left_shoulder=shoulder(left), right_shoulder=shoulder(right)))
            white.append(w[Parameter.PULSE_WIDTH])
            blue.append(b[Parameter.PULSE_WIDTH])
        return white, blue

    def test_a_t_is_still_over_a_whole_breath(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        first = self._render().light_img.copy()
        for _ in range(self.BREATH):
            np.testing.assert_array_equal(self._render().light_img, first)

    def test_both_colours_breathe_whichever_shoulder_is_higher(self) -> None:
        depth = self.cfg.breath.depth
        for left, right in ((1.0, 0.0), (0.0, 1.0)):
            white, blue = self._widths(left, right)
            with self.subTest(left=left, right=right):
                for widths in (white, blue):                                     # neither colour is left out
                    self.assertAlmostEqual(max(widths) - min(widths), 2 * depth, delta=0.02)
                    self.assertAlmostEqual(sum(widths) / len(widths), 0.5, delta=0.02)

    def test_the_left_higher_swings_the_colours_together_and_the_right_higher_apart(self) -> None:
        white, blue = self._widths(1.0, 0.0)                                     # the left up alone: in step
        for w, b in zip(white, blue):                                            # blue's slot inverts its source,
            self.assertAlmostEqual(w, 1.0 - b, places=5)                         # so its width is 1 − b: one breath
        white, blue = self._widths(0.0, 1.0)                                     # the right up alone: complementary
        for w, b in zip(white, blue):
            self.assertAlmostEqual(w, b, places=5)                               # width and 1 − b swing opposite ways

    def test_left_up_and_right_up_draw_differently(self) -> None:
        drawn = []
        for left, right in ((1.0, 0.0), (0.0, 1.0)):
            self.layer.reset()
            self._people({0: _pose(0.5, left_shoulder=shoulder(left), right_shoulder=shoulder(right))})
            drawn.append(self._render().light_img.copy())                        # the breath starts at its crest

        self.assertFalse(np.array_equal(drawn[0], drawn[1]))

    def test_at_full_depth_a_breath_never_passes_full_or_none_and_the_fixed_points_are_exact(self) -> None:
        self.cfg.breath.depth = 0.5
        for left in np.linspace(0.0, 1.0, 6):
            for right in np.linspace(0.0, 1.0, 6):
                white, blue = self._widths(left, right)
                with self.subTest(left=left, right=right):
                    self.assertGreaterEqual(min(white + blue), -1e-9)
                    self.assertLessEqual(max(white + blue), 1.0 + 1e-9)
        white, blue = self._widths(0.0, 0.0)
        self.assertEqual(set(white) | set(blue), {0.0})                          # neutral: no white, full blue
        white, blue = self._widths(1.0, 1.0)
        self.assertEqual(set(white) | set(blue), {1.0})                          # raised: full white, no blue

    def test_level_shoulders_tile_the_colours(self) -> None:
        inner = slice(C + MASK + 1, C + FULL)
        self._people({0: _pose(0.5, left_shoulder=shoulder(0.3))})                   # level: complementary widths
        f = self._render()
        white, blue = f.white[inner] > 0.5, f.blue[inner] > 0.5
        self.assertLessEqual(int(np.count_nonzero(white == blue)), 4 * len(_runs(f.white[inner])))   # every pixel one colour

    def test_one_shoulder_moves_both_colours(self) -> None:
        self.cfg.breath.depth = 0.0                                                     # the rest alone
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, right_shoulder=0.0)})   # the mean: a quarter
        f = self._render()
        self.assertEqual({l for _, l in self._inner(f.white)}, {INTERVAL // 4})
        self.assertEqual({l for _, l in self._inner(f.blue)}, {3 * INTERVAL // 4})

    # -- the elbows: the pitch of their colour --

    def _spacings(self, channel: np.ndarray) -> set[int]:
        centres = self._centres(self._inner(channel))
        return {round(b - a) for a, b in zip(centres, centres[1:])}

    def test_an_elbow_makes_its_own_colour_finer_and_leaves_the_other(self) -> None:
        self.cfg.white_lines.pitch_amount = self.cfg.blue_lines.pitch_amount = 25.7                  # folded: twice the lines
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, left_elbow=shoulder(1.0))})
        f = self._render()
        self.assertEqual(self._spacings(f.white), {INTERVAL // 2})
        self.assertEqual(self._spacings(f.blue), {INTERVAL})

    def test_equal_elbows_keep_the_colours_tuned(self) -> None:
        self.cfg.white_lines.pitch_amount = self.cfg.blue_lines.pitch_amount = 25.7
        for fold in (0.0, 0.5, 1.0):
            self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, left_elbow=shoulder(fold), right_elbow=shoulder(fold))})
            f = self._render()
            self.assertEqual(self._spacings(f.white), self._spacings(f.blue))

    # -- the elbows' turn: the shift --

    def test_an_elbows_turn_shifts_its_own_colour_either_way(self) -> None:
        self.cfg.white_lines.phase_amount = self.cfg.blue_lines.phase_amount = 0.25      # a quarter interval at a 90° turn
        self.cfg.white_lines.pitch_amount = self.cfg.blue_lines.pitch_amount = 0.0       # the fold leaves the pitch
        for degrees, moved in ((90.0, 35), (-90.0, -35), (0.0, 0)):                      # px: a quarter interval out
            self.layer.reset()
            self._people({0: _pose(0.5, left_shoulder=self.HALFWAY, left_elbow=math.radians(degrees))})
            f = self._render()
            self._on_grid(self._inner(f.white), INTERVAL, moved % INTERVAL)
            self._on_grid(self._inner(f.blue), INTERVAL, INTERVAL / 2)                   # the right elbow straight: still

    # -- no jumps, through the whole bridge --

    def test_a_small_move_of_any_measure_is_a_small_change(self) -> None:
        W, B = self.cfg.white_lines, self.cfg.blue_lines
        W.pitch_amount = B.pitch_amount = 46.3
        self.cfg.breath.depth = 0.0                          # the breath moves the lines in time, not a pose's picture
        base = dict(left_shoulder=shoulder(0.4), right_shoulder=shoulder(0.6), left_elbow=shoulder(0.3),
                    right_elbow=shoulder(0.5), tilt=0.0)
        # A step that moves an edge by about a pixel: a shoulder's moves a width, an elbow's the
        # whole accordion, so the elbow's is the smaller.
        for name, step in (("left_shoulder", 0.05), ("right_shoulder", 0.05), ("left_elbow", 0.01), ("right_elbow", 0.01)):
            self.layer.reset()
            self._people({0: _pose(0.5, **base)})
            first = self._render()
            edges = 2 * (len(_runs(first.white)) + len(_runs(first.blue)))
            self._people({0: _pose(0.5, **dict(base, **{name: base[name] + step}))})
            changed = int(np.count_nonzero(self._render().light_img != first.light_img))
            self.assertGreater(changed, 0, name)
            self.assertLessEqual(changed, 3 * edges, name)                               # a pixel or so per edge: nothing pops

    # -- the pattern --

    def test_in_between_the_lines_are_full_and_mirror_symmetric(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        f = self._render()
        outside = self._outside_mask()
        self.assertTrue(np.isin(f.white, (0.0, 1.0)).all())
        self.assertTrue(np.isin(f.blue[outside], (0.0, 1.0)).all())
        self.assertGreater(len(_runs(f.white)), 4)
        self.assertGreater(len(_runs(f.blue)), 4)
        span = REACH + 10
        for channel in (f.white, f.blue):
            np.testing.assert_array_equal(channel[C + 1:C + span], channel[C - 1:C - span:-1])

    def test_a_line_is_the_pulse_width_of_the_interval_and_the_colours_alternate(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        f = self._render()
        white, blue = self._inner(f.white), self._inner(f.blue)
        self.assertGreaterEqual(len(white), 2)
        self.assertEqual({l for _, l in white}, {INTERVAL // 2})
        self.assertEqual({l for _, l in blue}, {INTERVAL // 2})
        self._on_grid(white, INTERVAL)
        self._on_grid(blue, INTERVAL, INTERVAL / 2)                       # half an interval from the white

    def test_a_thin_line_is_drawn(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=shoulder(0.02))})       # under the old 2° limit: 0.28° wide
        white = self._inner(self._render().white)
        self.assertGreaterEqual(len(white), 2)
        self.assertLessEqual({l for _, l in white}, {2, 3})

    def test_the_lines_thin_out_in_the_taper_and_none_is_cut(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        white = self._right(self._render().white)
        self._on_grid(white, INTERVAL)                                    # every line where it belongs,
        last_start, last_width = white[-1]
        self.assertGreater(last_start, C + FULL)                          # the last one in the taper,
        self.assertLess(last_width, INTERVAL // 4)                        # thinned, not cut
        self.assertLessEqual(last_start + last_width, C + REACH)

    def test_the_pitch_never_goes_above_the_visual_limit(self) -> None:
        self.cfg.white_lines.pitch = 180.0                                      # the limit is 90 lines: 4°
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        white = self._inner(self._render().white)
        self.assertEqual({round(b - a) for a, b in zip(self._centres(white), self._centres(white)[1:])}, {40})

    def test_a_still_pose_is_a_still_frame_and_a_small_move_a_small_change(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=shoulder(0.9))})
        f = self._render()
        first = f.light_img.copy()
        edges = 2 * (len(_runs(f.white)) + len(_runs(f.blue)))
        np.testing.assert_array_equal(self._render().light_img, first)
        self._people({0: _pose(0.5, left_shoulder=shoulder(0.9) + 0.03)})   # lines ~1 px wider
        moved = self._render().light_img
        changed = int(np.count_nonzero(moved != first))
        self.assertGreater(changed, 0)
        self.assertLessEqual(changed, 2 * edges)                # each line edge shifts a pixel or two: nothing pops

    def test_a_walking_person_carries_the_pattern_smoothly(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        first = self._render()
        edges = 2 * (len(_runs(first.white)) + len(_runs(first.blue)))
        self._people({0: _pose(0.5 + 0.4 / IRES, left_shoulder=self.HALFWAY)})    # under half a pixel
        changed = int(np.count_nonzero(self._render().light_img != first.light_img))
        self.assertGreater(changed, 0)                          # the lines follow the person, not the pixel
        self.assertLessEqual(changed, 2 * edges)

    # -- the mask and the union --

    def test_every_mask_goes_over_every_pattern(self) -> None:
        b = round(0.52 * IRES)
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0)), 1: _pose(0.52)})    # B inside A's white window
        f = self._render()
        np.testing.assert_array_equal(f.white[b - MASK:b + MASK + 1], 0.0)
        np.testing.assert_allclose(f.blue[b - MASK:b + MASK + 1], self.cfg.mask.blue, atol=1e-6)
        self.assertEqual(float(f.white[b + MASK + 5]), 1.0)                 # A's white continues past B

    def test_opposite_draws_the_lines_half_a_turn_from_the_person(self) -> None:
        self.cfg.opposite = True
        quarter, three_quarters = IRES // 4, 3 * IRES // 4
        self._people({0: _pose(0.25, left_shoulder=shoulder(1.0))})            # full white
        f = self._render()
        np.testing.assert_array_equal(f.white[three_quarters + MASK + 1:three_quarters + SOLID], 1.0)
        self.assertEqual(float(f.white[quarter - FULL:quarter + FULL].sum()), 0.0)    # nothing at the person
        np.testing.assert_allclose(f.blue[quarter - MASK:quarter + MASK + 1], self.cfg.mask.blue, atol=1e-6)
        self.assertEqual(float(f.blue[three_quarters]), 0.0)                  # the mask stays behind

    def test_opposite_lets_the_masks_cut_the_patterns_they_fall_on(self) -> None:
        self.cfg.opposite = True
        b = round(0.75 * IRES)
        self._people({0: _pose(0.25, left_shoulder=shoulder(1.0)), 1: _pose(0.75)})   # B stands in A's opposite white
        f = self._render()
        np.testing.assert_array_equal(f.white[b - MASK:b + MASK + 1], 0.0)
        self.assertEqual(float(f.white[b + MASK + 5]), 1.0)                   # A's white continues past B

    def test_overlapping_voices_show_the_fuller_one(self) -> None:
        self._people({0: _pose(0.5), 1: _pose(0.52)})                       # two full blues over each other
        f = self._render()
        self.assertEqual(float(f.blue.max()), 1.0)
        self.assertEqual(float(f.blue[C + 100]), 1.0)

    # -- the hit --

    STEADY = (23.4, 16.2, 9.0, 1.8, -5.4, -12.6)     # 36 rpm at 30 Hz: 7.2° a tick, closest at +1.8°

    def _sweep(self, **pose) -> list[Frame]:
        frames = []
        for offset in self.STEADY:
            self._people({0: _pose(0.5, offset_deg=offset, **pose)})
            frames.append(self._render())
        return frames

    def test_the_mask_flashes_on_the_hit_tick(self) -> None:
        M = self.cfg.mask
        M.flash_release_seconds = 0.0                                     # the hit tick only
        frames = self._sweep()
        np.testing.assert_allclose([float(f.blue[C]) for f in frames],
                                   [M.flash_blue if e else M.blue for e in (0, 0, 0, 1, 0, 0)], atol=1e-6)
        self.assertEqual(sum(float(f.white[C]) for f in frames), 0.0)     # both white levels are 0

    def test_the_flash_falls_back_over_its_release(self) -> None:
        M = self.cfg.mask
        M.flash_release_seconds = 0.1                                     # three ticks
        levels = [float(f.blue[C]) for f in self._sweep()]
        self.assertAlmostEqual(levels[3], M.flash_blue, places=6)
        self.assertGreater(levels[4], M.blue)                             # still falling
        self.assertGreater(levels[3], levels[4])
        self.assertGreater(levels[4], levels[5])
        self.assertAlmostEqual(float(self._render().blue[C]), M.blue, places=6)   # 4 ticks on: the mask again

    def test_a_flash_at_the_masks_levels_marks_nothing(self) -> None:
        M = self.cfg.mask
        M.flash_blue = M.blue
        for f in self._sweep():
            self.assertAlmostEqual(float(f.blue[C]), M.blue, places=6)

    def test_the_mask_has_a_white_level_too(self) -> None:
        M = self.cfg.mask
        M.white = 0.2
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0))})        # full white: the mask still cuts it
        f = self._render()
        np.testing.assert_allclose(f.white[C - MASK:C + MASK + 1], 0.2, atol=1e-6)
        np.testing.assert_array_equal(f.white[C + MASK + 1:C + SOLID + 1], 1.0)

    def test_the_hit_leaves_the_lines_colours_alone(self) -> None:
        outside = self._outside_mask()
        frames = self._sweep(left_shoulder=self.HALFWAY)
        np.testing.assert_array_equal(frames[3].white[outside], frames[2].white[outside])
        np.testing.assert_array_equal(frames[3].blue[outside], frames[2].blue[outside])

    def test_the_push_moves_standing_lines_and_they_keep_the_gain(self) -> None:
        self.cfg.white_lines.push = 14.0                              # an interval per second, outward
        self.cfg.blue_lines.push = -14.0
        self.cfg.white_lines.push_release_seconds = self.cfg.blue_lines.push_release_seconds = 0.1
        frames = self._sweep(left_shoulder=self.HALFWAY)
        before_white = self._centres(self._inner(frames[2].white))
        before_blue = self._centres(self._inner(frames[2].blue))
        on_hit = self._centres(self._inner(frames[3].white))                # the hit tick moves by push × dt
        self.assertAlmostEqual(on_hit[0] - before_white[0], INTERVAL * TICK, delta=1.0)
        for _ in range(30):
            settled = self._render()
        expected = INTERVAL * TICK * (1.0 + 0.75 + 0.25)                    # the release's eased levels, summed
        self.assertAlmostEqual(self._centres(self._inner(settled.white))[0] - before_white[0], expected, delta=1.0)
        self.assertAlmostEqual(before_blue[0] - self._centres(self._inner(settled.blue))[0], expected, delta=1.0)
        np.testing.assert_array_equal(self._render().light_img, settled.light_img)     # kept, nothing comes back

    def test_speed_moves_white_out_and_blue_in(self) -> None:
        self.cfg.white_lines.speed = 7.0                              # half an interval per second
        self.cfg.blue_lines.speed = -3.5
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        for _ in range(30):                                     # a second
            f = self._render()
        self._on_grid(self._inner(f.white), INTERVAL, INTERVAL / 2)         # out by half an interval
        self._on_grid(self._inner(f.blue), INTERVAL, INTERVAL / 4)          # in by a quarter, from the half
        for _ in range(30):
            f = self._render()
        self._on_grid(self._inner(f.white), INTERVAL)                       # a full interval on: the grid again
        self._on_grid(self._inner(f.blue), INTERVAL)

    def test_an_unmirrored_oscillator_passes_behind_the_person(self) -> None:
        self.cfg.white_lines.mirror = False
        self.cfg.white_lines.speed = 7.0                        # half an interval per second
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        first = self._render()
        for _ in range(29):                                     # a second in all
            f = self._render()
        self._on_grid(self._inner(f.white), INTERVAL, INTERVAL / 2)            # right: away from the person
        before_left = self._centres(self._left(first.white))[-1]
        after_left = self._centres(self._left(f.white))[-1]
        self.assertAlmostEqual(after_left - before_left, 29 * 7.0 * TICK * 10, delta=1.0)   # left: toward the person
        self._on_grid(self._inner(f.blue), INTERVAL, INTERVAL / 2)             # blue, mirrored: as before

    def test_reset_starts_a_new_pass(self) -> None:
        M = self.cfg.mask
        M.flash_release_seconds = 0.0
        self._people({0: _pose(0.5, offset_deg=1.8)})
        self.assertAlmostEqual(float(self._render().blue[C]), M.flash_blue, places=6)
        self.assertAlmostEqual(float(self._render().blue[C]), M.blue, places=6)   # one hit per pass
        self.layer.reset()
        self.assertAlmostEqual(float(self._render().blue[C]), M.flash_blue, places=6)

    # -- the playhead's marker --

    def test_the_marker_is_drawn_at_the_playhead_with_nobody_there(self) -> None:
        P = self.cfg.playhead
        P.width = 1.0                                                     # 10 px
        f = self._render(playhead=0.25)
        centre = IRES // 4
        np.testing.assert_array_equal(f.white[centre - 5:centre + 5], P.white)
        self.assertEqual(float(f.white.sum()), 10 * P.white)
        self.assertEqual(float(f.blue.sum()), 0.0)                        # blue at 0: untouched

    def test_the_marker_has_a_level_per_channel_and_dims_inside_a_mask(self) -> None:
        P = self.cfg.playhead
        P.width, P.white, P.blue = 1.0, 0.8, 0.4
        self._people({0: _pose(0.5)})                                     # full blue, the mask at C
        f = self._render(playhead=0.25)                                   # beyond the reach: dark there
        centre = IRES // 4
        np.testing.assert_allclose(f.white[centre - 5:centre + 5], 0.8)
        np.testing.assert_allclose(f.blue[centre - 5:centre + 5], 0.4)
        f = self._render(playhead=0.5)                                    # in the mask
        np.testing.assert_allclose(f.white[C - 5:C + 5], 0.8 * P.at_mask)
        np.testing.assert_allclose(f.blue[C - 5:C + 5], self.cfg.mask.blue + 0.4 * P.at_mask)

    def test_no_playhead_draws_no_marker(self) -> None:
        f = self._render()
        self.assertEqual(float(f.light_img.sum()), 0.0)

    # -- playing by hand --

    BYPASSES = ("pitch_bypass", "pulse_width_bypass", "phase_bypass", "speed_bypass", "hardness_bypass")

    def test_bypass_all_sets_every_tick_of_the_oscillator_and_again_clears_them(self) -> None:
        W, B = self.cfg.white_lines, self.cfg.blue_lines
        W.pulse_width = 0.25
        type(W).bypass_all.fire(W)
        self.assertTrue(all(getattr(W, name) for name in self.BYPASSES))
        self.assertFalse(any(getattr(B, name) for name in self.BYPASSES))     # the other oscillator untouched
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0), right_shoulder=shoulder(1.0))})   # the pose: full white, no blue
        f = self._render()
        white = self._inner(f.white)
        self.assertGreaterEqual(len(white), 2)
        self.assertEqual({l for _, l in white}, {INTERVAL // 4})            # white from the panel,
        self.assertEqual(float(f.blue[self._outside_mask()].sum()), 0.0)    # blue still following the body
        type(W).bypass_all.fire(W)                                          # all set: pressed again, cleared
        self.assertFalse(any(getattr(W, name) for name in self.BYPASSES))
        W.phase_bypass = True                                               # one set: pressed, all set
        type(W).bypass_all.fire(W)
        self.assertTrue(all(getattr(W, name) for name in self.BYPASSES))
        type(W).bypass_all.fire(W)
        self.assertEqual(self._inner(self._render().white), [])             # the pose again: solid, no lines
        self.assertEqual(float(self._render().white[C + 100]), 1.0)

    def test_a_bypass_takes_one_input_from_the_panel_and_leaves_the_rest_to_the_body(self) -> None:
        self.cfg.white_lines.pulse_width_bypass = True
        self.cfg.white_lines.pulse_width = 0.25
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0))})          # the body says full white, no blue
        f = self._render()
        white = self._inner(f.white)
        self.assertGreaterEqual(len(white), 2)
        self.assertEqual({l for _, l in white}, {INTERVAL // 4})            # white from the panel,
        self.assertEqual(float(f.blue[self._outside_mask()].sum()), 0.0)    # blue still following the body

    def test_a_curve_eases_a_source_and_keeps_its_ends(self) -> None:
        self.cfg.white_lines.pulse_width_curve = Curve.EASE_IN_QUAD               # little at first
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        eased = {l for _, l in self._inner(self._render().white)}
        self.assertEqual(eased, {INTERVAL // 4})                            # 0.5² of the interval
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0))})
        self.assertEqual(float(self._render().white[C + 100]), 1.0)         # the end unchanged: solid

    def test_the_window_bypass_holds_the_reach_without_a_partner(self) -> None:
        self.cfg.window.width_bypass = True
        self.cfg.window.width = 90.0
        self._people({0: _pose(0.5)})
        f = self._render()
        np.testing.assert_array_equal(f.blue[C + MASK + 1:C + 2 * FULL - INTERVAL // 2 + 1], 1.0)
        np.testing.assert_array_equal(f.blue[C - 2 * FULL + INTERVAL // 2:C - MASK], 1.0)
        self.assertEqual(float(f.blue[C + 2 * REACH + 1:].sum() + f.blue[:C - 2 * REACH].sum()), 0.0)

    # -- the strobe --

    def _strobing(self, rate: float, width: float, shift: float = 0.0, **pose) -> None:
        """The white strobe set by hand, at 32 fps, one person at C with the given pose (arms
        halfway by default: alternating lines)."""
        self.dt = 1 / 32
        S = self.cfg.white_strobe
        S.rate, S.width, S.shift = rate, width, shift
        self._people({0: _pose(0.5, **(pose or dict(left_shoulder=self.HALFWAY)))})

    def test_a_strobe_darkens_the_whole_output_on_its_dark_ticks_and_leaves_the_rest(self) -> None:
        self._strobing(rate=4, width=7 / 8)                           # an 8 tick cycle, its last tick dark
        outside = self._outside_mask()
        for tick in range(16):
            f = self._render()
            with self.subTest(tick=tick):
                if tick % 8 == 7:
                    self.assertEqual(float(f.white.sum()), 0.0)                        # every white line off
                else:
                    self.assertGreaterEqual(len(self._inner(f.white)), 2)              # the lines as drawn
                    self.assertTrue(np.isin(f.white, (0.0, 1.0)).all())                # off or full, nothing between
                self.assertGreaterEqual(len(self._inner(f.blue)), 2)                   # the blue strobe is off
                np.testing.assert_allclose(f.blue[C - MASK:C + MASK + 1], self.cfg.mask.blue, atol=1e-6)   # the mask stays
        self.assertTrue(np.isin(f.blue[outside], (0.0, 1.0)).all())

    def test_two_people_at_one_rate_go_dark_on_the_same_tick(self) -> None:
        self._strobing(rate=4, width=7 / 8)
        b = round(0.7 * IRES)
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0)), 1: _pose(0.7, left_shoulder=shoulder(1.0))})   # both full white
        for tick in range(8):
            f = self._render()
            a_lit, b_lit = float(f.white[C + 100]), float(f.white[b + 100])
            with self.subTest(tick=tick):
                self.assertEqual(a_lit, b_lit)
                self.assertEqual(a_lit, 0.0 if tick == 7 else 1.0)

    def test_the_shift_runs_the_dark_outward_one_line_at_a_time(self) -> None:
        self._strobing(rate=4, width=7 / 8, shift=0.25)               # a quarter of 8 ticks, a tick per line: line k dark at tick 7 + k
        lines = [C + 20, C + INTERVAL, C + 2 * INTERVAL, C + 3 * INTERVAL]    # a pixel in lines 0..3 (0 seen past the mask)
        for tick in range(12):
            f = self._render()
            with self.subTest(tick=tick):
                self.assertEqual([float(f.white[px]) for px in lines],
                                 [0.0 if (tick - k) % 8 == 7 else 1.0 for k in range(4)])

    def test_a_strobe_at_zero_is_no_strobe(self) -> None:
        self._strobing(rate=0, width=0.0)                             # width 0 would be all dark, were it on
        for _ in range(4):
            self.assertGreaterEqual(len(self._inner(self._render().white)), 2)

    STROBE_BYPASSES = ("rate_bypass", "width_bypass", "phase_bypass", "shift_bypass")

    def test_a_strobes_bypass_all_sets_its_four_and_again_clears_them(self) -> None:
        S, T = self.cfg.white_strobe, self.cfg.blue_strobe
        type(S).bypass_all.fire(S)
        self.assertTrue(all(getattr(S, name) for name in self.STROBE_BYPASSES))
        self.assertFalse(any(getattr(T, name) for name in self.STROBE_BYPASSES))
        type(S).bypass_all.fire(S)
        self.assertFalse(any(getattr(S, name) for name in self.STROBE_BYPASSES))

    # -- sync --

    def test_sync_fills_the_arc_between_the_pair_above_threshold_only(self) -> None:
        a, b = round(0.3 * IRES), round(0.7 * IRES)
        self._people({0: _pose(0.3, sims={1: 0.5}), 1: _pose(0.7, sims={0: 0.5})})
        self.assertEqual(float(self._render().blue[C]), 0.0)
        self.layer.reset()
        self._people({0: _pose(0.3, sims={1: 1.0}), 1: _pose(0.7, sims={0: 1.0})})
        f = self._render()
        np.testing.assert_array_equal(f.blue[a + MASK + 1:b - MASK], 1.0)

    def test_sync_opens_the_partners_side_only(self) -> None:
        a, b = round(0.3 * IRES), round(0.7 * IRES)
        self._people({0: _pose(0.3, sims={1: 1.0}), 1: _pose(0.7, sims={0: 1.0})})
        f = self._render()
        self.assertEqual(float(f.blue[:a - REACH].sum()), 0.0)               # the far sides keep their rest reach
        self.assertEqual(float(f.blue[b + REACH + 1:].sum()), 0.0)
        np.testing.assert_array_equal(f.blue[a - FULL:a - MASK], 1.0)

    def test_sync_reaches_over_an_intermediate_person(self) -> None:
        probe = round(0.35 * IRES) + 50         # beyond both rest reaches, inside the arms-up person's
        raised = shoulder(1.0)
        self._people({0: _pose(0.2), 1: _pose(0.35, left_shoulder=raised), 2: _pose(0.5)})
        self.assertEqual(float(self._render().blue[probe]), 0.0)
        self.layer.reset()
        self._people({0: _pose(0.2, sims={2: 1.0}), 1: _pose(0.35, left_shoulder=raised), 2: _pose(0.5, sims={0: 1.0})})
        f = self._render()
        self.assertEqual(float(f.blue[probe]), 1.0)
        self.assertEqual(float(f.white[probe]), 1.0)             # the intermediate pattern shows too

    def test_sync_takes_the_shorter_arc_across_the_wrap(self) -> None:
        self._people({0: _pose(0.95, sims={1: 1.0}), 1: _pose(0.05, sims={0: 1.0})})
        f = self._render()
        self.assertEqual(float(f.blue[0]), 1.0)
        self.assertEqual(float(f.blue[round(0.2 * IRES):round(0.8 * IRES)].sum()), 0.0)

    def test_a_leaving_partner_lets_the_reach_go_smoothly(self) -> None:
        self.cfg.window.release_seconds = 1.0
        self._people({0: _pose(0.3, sims={1: 1.0}), 1: _pose(0.7, sims={0: 1.0})})
        self._render()
        self._people({0: _pose(0.3, sims={1: 1.0})})             # the partner is gone
        lit = [int(np.count_nonzero(self._render().blue)) for _ in range(40)]
        self.assertLess(max(abs(b - a) for a, b in zip(lit, lit[1:])), 250)    # closing, never snapping shut
        self.assertLess(lit[-1], 2 * REACH + 10)                 # back at the rest reach

    # -- presence --

    def test_attack_opens_the_window_from_the_mask(self) -> None:
        self.cfg.window.attack_seconds = 1.0
        self._people({0: _pose(0.5)})
        first = self._render()
        self.assertEqual(float(first.blue[C + MASK + 1:].sum()), 0.0)     # the window is still inside the mask
        for _ in range(30):
            last = self._render()
        self.assertEqual(float(last.blue[C + 300]), 1.0)

    def test_release_closes_the_window_then_everything_goes_and_reset_clears(self) -> None:
        self.cfg.window.release_seconds = 1.0
        self._people({0: _pose(0.5)})
        self._render()
        self._people({})                                        # gone
        held = self._render()
        self.assertEqual(float(held.blue[C + 300]), 1.0)
        self.assertEqual(float(held.blue[C + REACH]), 0.0)      # the window closes
        self.assertLess(float(held.blue[C]), self.cfg.mask.blue)
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
