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
from apps.white_space.light.synth import Input
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
          offset_deg: float = float("nan")) -> FakePose:
    """A fake pose at normalized azimuth ``azimuth_pos`` (0..1) with the left shoulder's angle
    (radians), pairwise sims and playhead offset."""
    angles = np.full(len(features.AngleLandmark), np.nan)
    angles[features.AngleLandmark.left_shoulder] = left_shoulder
    sim_values = np.full(16, np.nan)
    for j, v in (sims or {}).items():
        sim_values[j] = v
    return FakePose({
        features.Azimuth: SimpleNamespace(value=azimuth_pos * math.tau),
        features.Angles: SimpleNamespace(values=angles),
        features.AngleSymmetry: SimpleNamespace(values=np.full(len(features.SymmetryElement), np.nan)),
        features.Similarity: SimpleNamespace(values=sim_values),
        features.LegDeviation: SimpleNamespace(value=0.0),
        features.TorsoTilt: SimpleNamespace(value=0.0),
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
        self.cfg.presence.attack_seconds = 0.0            # present at once — geometry tests read one frame
        W, B = self.cfg.white, self.cfg.blue              # the placeholder's patch: white out, blue in
        W.pulse_width, W.pulse_width_amount, W.phase = 0.0, 1.0, self.QUARTER
        B.pulse_width, B.pulse_width_amount, B.phase = 1.0, -1.0, -0.5 + self.QUARTER
        self.board = InstrumentBoard(frames={})
        self.layer = PoseInstrument(IRES, LayerSettings(), self.cfg, self.board, pose_stage=4)

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

    @staticmethod
    def _right(channel: np.ndarray) -> list[tuple[int, int]]:
        """(start, length) of the runs on the person's right, clear of the mask."""
        return [(s, l) for s, l in _runs(channel) if C + MASK + 1 < s]

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

    def test_the_left_shoulder_plays_both_pulse_widths(self) -> None:
        for fraction in (0.0, 0.5, 1.0):
            white, blue = self._connect(_pose(0.5, left_shoulder=shoulder(fraction)))
            self.assertAlmostEqual(white[Input.PULSE_WIDTH], fraction, places=5)
            self.assertAlmostEqual(blue[Input.PULSE_WIDTH], fraction, places=5)

    def test_the_sign_of_an_angle_is_not_a_measure(self) -> None:
        # The sign is the side of the body the arm passes; straight up is π from either side.
        for angle in (shoulder(0.5), -shoulder(0.5)):
            self.assertAlmostEqual(self._connect(_pose(0.5, left_shoulder=angle))[0][Input.PULSE_WIDTH], 0.5, places=5)

    # -- the two fixed points --

    def test_arms_hanging_is_full_blue_with_the_dim_mask(self) -> None:
        self._people({0: _pose(0.5)})
        f = self._render()
        self.assertEqual(float(f.white.sum()), 0.0)
        np.testing.assert_array_equal(f.blue[C + MASK + 1:C + SOLID + 1], 1.0)
        np.testing.assert_array_equal(f.blue[C - SOLID:C - MASK], 1.0)
        np.testing.assert_allclose(f.blue[C - MASK:C + MASK + 1], self.cfg.mask.brightness, atol=1e-6)
        self.assertEqual(float(f.blue[C + REACH + 1:].sum() + f.blue[:C - REACH].sum()), 0.0)

    def test_arms_up_is_full_white_and_blue_only_in_the_mask(self) -> None:
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0))})
        f = self._render()
        np.testing.assert_array_equal(f.white[C + MASK + 1:C + SOLID + 1], 1.0)
        np.testing.assert_array_equal(f.white[C - SOLID:C - MASK], 1.0)
        np.testing.assert_array_equal(f.white[C - MASK:C + MASK + 1], 0.0)    # the mask goes over white
        self.assertEqual(float(f.blue[self._outside_mask()].sum()), 0.0)

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

    def test_the_interval_never_goes_below_the_visual_limit(self) -> None:
        self.cfg.white.interval = 1.0                                     # the limit is 4° at 90 lines
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
        np.testing.assert_allclose(f.blue[b - MASK:b + MASK + 1], self.cfg.mask.brightness, atol=1e-6)
        self.assertEqual(float(f.white[b + MASK + 5]), 1.0)                 # A's white continues past B

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

    def test_the_mask_flashes_on_the_hit_frames(self) -> None:
        M = self.cfg.mask
        for frames, expected in ((1, [0, 0, 0, 1, 0, 0]), (3, [0, 0, 1, 1, 1, 0])):
            self.cfg.events.hit_frames = frames
            self.layer.reset()
            levels = [float(f.blue[C]) for f in self._sweep()]
            np.testing.assert_allclose(levels, [M.flash_brightness if e else M.brightness for e in expected], atol=1e-6)

    def test_the_hit_leaves_the_lines_colours_alone(self) -> None:
        outside = self._outside_mask()
        frames = self._sweep(left_shoulder=self.HALFWAY)
        np.testing.assert_array_equal(frames[3].white[outside], frames[2].white[outside])
        np.testing.assert_array_equal(frames[3].blue[outside], frames[2].blue[outside])

    def test_the_push_moves_standing_lines_and_they_keep_the_gain(self) -> None:
        self.cfg.white.push = 14.0                              # an interval per second, outward
        self.cfg.blue.push = -14.0
        self.cfg.push.settle_seconds = 0.1
        frames = self._sweep(left_shoulder=self.HALFWAY)
        before_white = self._centres(self._inner(frames[2].white))
        before_blue = self._centres(self._inner(frames[2].blue))
        on_hit = self._centres(self._inner(frames[3].white))                # the hit tick moves by push × dt
        self.assertAlmostEqual(on_hit[0] - before_white[0], INTERVAL * TICK, delta=1.0)
        for _ in range(30):
            settled = self._render()
        expected = INTERVAL * TICK * (1.0 + 0.75 + 0.25)                    # the settle's eased levels, summed
        self.assertAlmostEqual(self._centres(self._inner(settled.white))[0] - before_white[0], expected, delta=1.0)
        self.assertAlmostEqual(before_blue[0] - self._centres(self._inner(settled.blue))[0], expected, delta=1.0)
        np.testing.assert_array_equal(self._render().light_img, settled.light_img)     # kept, nothing comes back

    def test_speed_moves_white_out_and_blue_in(self) -> None:
        self.cfg.white.speed = 7.0                              # half an interval per second
        self.cfg.blue.speed = -3.5
        self._people({0: _pose(0.5, left_shoulder=self.HALFWAY)})
        for _ in range(30):                                     # a second
            f = self._render()
        self._on_grid(self._inner(f.white), INTERVAL, INTERVAL / 2)         # out by half an interval
        self._on_grid(self._inner(f.blue), INTERVAL, INTERVAL / 4)          # in by a quarter, from the half
        for _ in range(30):
            f = self._render()
        self._on_grid(self._inner(f.white), INTERVAL)                       # a full interval on: the grid again
        self._on_grid(self._inner(f.blue), INTERVAL)

    def test_reset_starts_a_new_pass(self) -> None:
        M = self.cfg.mask
        self._people({0: _pose(0.5, offset_deg=1.8)})
        self.assertAlmostEqual(float(self._render().blue[C]), M.flash_brightness, places=6)
        self.assertAlmostEqual(float(self._render().blue[C]), M.brightness, places=6)   # one hit per pass
        self.layer.reset()
        self.assertAlmostEqual(float(self._render().blue[C]), M.flash_brightness, places=6)

    # -- playing by hand --

    def test_the_override_draws_the_panel_not_the_pose(self) -> None:
        self.cfg.override.on = True
        self.cfg.white.pulse_width = 0.25
        self._people({0: _pose(0.5, left_shoulder=shoulder(1.0))})          # the pose says full white
        white = self._inner(self._render().white)
        self.assertGreaterEqual(len(white), 2)
        self.assertEqual({l for _, l in white}, {INTERVAL // 4})            # the base, from the panel
        self.cfg.override.on = False
        self.assertEqual(self._inner(self._render().white), [])             # the pose again: solid, no lines
        self.assertEqual(float(self._render().white[C + 100]), 1.0)

    def test_the_override_holds_the_reach_without_a_partner(self) -> None:
        self.cfg.override.on = True
        self.cfg.override.reach = 90.0
        self._people({0: _pose(0.5)})
        f = self._render()
        np.testing.assert_array_equal(f.blue[C + MASK + 1:C + 2 * FULL - INTERVAL // 2 + 1], 1.0)
        np.testing.assert_array_equal(f.blue[C - 2 * FULL + INTERVAL // 2:C - MASK], 1.0)
        self.assertEqual(float(f.blue[C + 2 * REACH + 1:].sum() + f.blue[:C - 2 * REACH].sum()), 0.0)

    def test_the_hit_button_marks_everyone_for_the_hit_frames(self) -> None:
        M = self.cfg.mask
        self._people({0: _pose(0.5), 1: _pose(0.25)})            # the playhead is nowhere near
        for frames, expected in ((1, [0, 1, 0, 0]), (3, [0, 1, 1, 1, 0])):
            self.cfg.events.hit_frames = frames
            levels = []
            for i in range(len(expected)):
                if i == 1:
                    type(self.cfg.override).hit.fire(self.cfg.override)
                f = self._render()
                levels.append((float(f.blue[C]), float(f.blue[IRES // 4])))
            for tick, (a, b) in enumerate(levels):
                want = M.flash_brightness if expected[tick] else M.brightness
                self.assertAlmostEqual(a, want, places=6, msg=f"tick {tick} of {frames}")
                self.assertAlmostEqual(b, want, places=6, msg=f"tick {tick} of {frames}")

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
        self.cfg.presence.release_seconds = 1.0
        self._people({0: _pose(0.3, sims={1: 1.0}), 1: _pose(0.7, sims={0: 1.0})})
        self._render()
        self._people({0: _pose(0.3, sims={1: 1.0})})             # the partner is gone
        lit = [int(np.count_nonzero(self._render().blue)) for _ in range(40)]
        self.assertLess(max(abs(b - a) for a, b in zip(lit, lit[1:])), 250)    # closing, never snapping shut
        self.assertLess(lit[-1], 2 * REACH + 10)                 # back at the rest reach

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
        self.assertEqual(float(held.blue[C + REACH]), 0.0)      # the window closes
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
