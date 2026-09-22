"""Tests for the dummy: the figure the pipeline reads back as its set degrees, the poses file,
the morph, and the merge step."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from modules.pose.features import Angles, AngleLandmark, Azimuth, BBox, Distance, LegDeviation, Points2D, PointLandmark as P, TorsoTilt
from modules.pose.frame import Frame
from modules.pose.nodes import (AngleCalibrator, AngleCalibratorSettings, AngleExtractor, AngleExtractorSettings,
                                LegDeviationExtractor, LegDeviationExtractorSettings, TorsoTiltExtractor,
                                TorsoTiltExtractorSettings)
from modules.pose.trackers import FilterPipeline

from apps.white_space.pose import Dummy, DummySettings, Measures, REST, dummy_id

POSES = Path('apps/white_space/data/poses.json')
JOINTS = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee')
ID = dummy_id(4)            # four live players: the dummy is 4, the ghosts start at 5


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


POSES_FILE = json.loads(POSES.read_text(encoding='utf-8'))
NEUTRAL = Measures(**POSES_FILE['neutral'])                     # the saved neutral: a body's, not the extractor's zero
UP = Measures(**POSES_FILE['raised'])                           # the calibrator's two reference poses
ROWS = ('neutral', 'raised', 'arms out level, a T', 'left arm up, right hanging', 'right arm up, left hanging',
        'a T, both elbows folded', 'a T, left elbow folded', 'a T, right elbow folded', 'a T, leaning left', 'a T, leaning right',
        'a T, in a crouch', '|__', '__|')


def from_neutral(**deltas: float) -> Measures:
    """NEUTRAL with joints moved by the given degrees, kept in −180..180."""
    return replace(NEUTRAL, **{name: (getattr(NEUTRAL, name) + delta + 180.0) % 360.0 - 180.0 for name, delta in deltas.items()})


def extractor_degrees(m: Measures, name: str) -> float:
    """What the angle extractor reads at a joint set to ``m``: its setting plus its rest."""
    return getattr(m, name) + REST[name]


class FigureTest(unittest.TestCase):
    """The dummy's degrees are the angle extractor's angles: what is set is what it reads."""

    def setUp(self) -> None:
        self.angle_settings = AngleExtractorSettings()
        self.extractor = AngleExtractor(self.angle_settings)

    def _read(self, m: Measures) -> np.ndarray:
        points = Dummy.points(m, self.angle_settings.aspect_ratio)
        return self.extractor.process(Frame(0, 0, features={Points2D: points}))[Angles].values

    def _assert_reads_as_set(self, m: Measures) -> None:
        read = self._read(m)
        for name in JOINTS:
            with self.subTest(joint=name):
                self.assertAlmostEqual(wrap(float(read[AngleLandmark[name]]) - math.radians(extractor_degrees(m, name))), 0.0, delta=1e-3)

    def test_the_default_reads_as_set(self) -> None:
        self._assert_reads_as_set(Measures())

    def test_every_saved_pose_reads_as_set_when_upright(self) -> None:
        for name, values in POSES_FILE.items():
            with self.subTest(pose=name):
                self._assert_reads_as_set(replace(Measures(**values), torso=0.0))   # a lean moves the hips' reading

    def test_the_arms_read_as_set_round_the_circle(self) -> None:
        for degrees in range(0, 360, 30):
            with self.subTest(degrees=degrees):
                self._assert_reads_as_set(Measures(left_shoulder=degrees, right_shoulder=(degrees + 90) % 360,
                                                   left_elbow=(degrees + 180) % 360, right_elbow=(degrees + 45) % 360))

    def test_a_mixed_pose_reads_as_set(self) -> None:
        self._assert_reads_as_set(Measures(left_shoulder=45.0, right_shoulder=135.0, left_elbow=-120.0, right_elbow=70.0,
                                           left_hip=-30.0, right_hip=-80.0, left_knee=-60.0, right_knee=-135.0))

    def test_the_arms_read_as_set_under_a_lean(self) -> None:
        m = Measures(torso=30.0, left_shoulder=45.0, right_shoulder=135.0, left_elbow=-120.0, right_elbow=70.0)
        read = self._read(m)
        for name in ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'):
            with self.subTest(joint=name):
                self.assertAlmostEqual(wrap(float(read[AngleLandmark[name]]) - math.radians(extractor_degrees(m, name))), 0.0, delta=1e-3)

    def test_the_rest_is_every_joint_at_zero(self) -> None:
        points = Dummy.points(Measures(), 1.0).values
        self.assertGreater(points[P.left_wrist][1], points[P.left_elbow][1])       # the arms hang
        self.assertGreater(points[P.left_ankle][1], points[P.left_knee][1])        # the legs stand,
        thigh, shin = points[P.left_knee] - points[P.left_hip], points[P.left_ankle] - points[P.left_knee]
        self.assertAlmostEqual(float(thigh[0] * shin[1] - thigh[1] * shin[0]), 0.0, places=5)   # the knee straight

    def test_the_legs_stand_still_under_a_lean(self) -> None:
        legs = dict(left_hip=-30.0, right_hip=-80.0, left_knee=-60.0, right_knee=-135.0)
        upright, leaning = Dummy.points(Measures(**legs), 1.0).values, Dummy.points(Measures(torso=30.0, **legs), 1.0).values
        for lm in (P.left_hip, P.right_hip, P.left_knee, P.right_knee, P.left_ankle, P.right_ankle):
            with self.subTest(landmark=lm.name):
                np.testing.assert_allclose(leaning[lm], upright[lm], atol=1e-6)

    def test_the_sides_are_named_as_the_pipeline_names_people(self) -> None:
        points = Dummy.points(Measures(), 1.0).values
        self.assertGreater(points[P.left_shoulder][0], points[P.right_shoulder][0])    # left on image-right
        self.assertGreater(points[P.left_hip][0], points[P.right_hip][0])

    def test_equal_degrees_are_a_mirror_symmetric_figure(self) -> None:
        points = Dummy.points(Measures(left_shoulder=70.0, right_shoulder=70.0, left_elbow=-135.0, right_elbow=-135.0,
                                       left_hip=-30.0, right_hip=-30.0, left_knee=-60.0, right_knee=-60.0), 1.0).values
        for left, right in ((P.left_elbow, P.right_elbow), (P.left_wrist, P.right_wrist),
                            (P.left_knee, P.right_knee), (P.left_ankle, P.right_ankle)):
            self.assertAlmostEqual(float(points[left][0] + points[right][0]), 1.0, places=5)     # mirrored about x = 0.5
            self.assertAlmostEqual(float(points[left][1]), float(points[right][1]), places=5)

    def test_the_torso_leans_the_upper_body(self) -> None:
        points = Dummy.points(Measures(torso=30.0), 1.0).values
        hips = (points[P.left_hip] + points[P.right_hip]) / 2.0
        shoulders = (points[P.left_shoulder] + points[P.right_shoulder]) / 2.0
        self.assertGreater(shoulders[0], hips[0] + 0.1)                              # to image right

    def test_the_crop_squeezes_y_by_the_aspect(self) -> None:
        square, crop = Dummy.points(Measures(), 1.0).values, Dummy.points(Measures(), 0.75).values
        np.testing.assert_allclose(crop[:, 0], square[:, 0], atol=1e-6)
        np.testing.assert_allclose(crop[:, 1], square[:, 1] * 0.75, atol=1e-6)



class ReadingsTest(unittest.TestCase):
    """What the pipeline reads of the dummy, with the calibrator set to the dummy's own raw
    readings in the saved neutral and with the arms straight up: self-consistency, so figure and
    calibration cannot drift apart. No test pins the preset to the dummy: the calibrator is tuned
    on a person, and the dummy reading a little off after that is information."""

    def setUp(self) -> None:
        self.angle_settings = AngleExtractorSettings()
        self.extractor = AngleExtractor(self.angle_settings)
        raw = self._raw(NEUTRAL)
        up = self._raw(UP)
        self.calibration = AngleCalibratorSettings()
        self.calibration.shoulder_neutral = math.degrees(float(raw[AngleLandmark.left_shoulder]))   # the settings are degrees
        self.calibration.shoulder_raised = math.degrees(float(up[AngleLandmark.left_shoulder]))
        self.calibration.elbow_neutral = math.degrees(float(raw[AngleLandmark.left_elbow]))
        self.calibration.elbow_raised = math.degrees(float(up[AngleLandmark.left_elbow]))
        self.calibration.hip_neutral = math.degrees(float(raw[AngleLandmark.left_hip]))
        self.calibration.knee_neutral = math.degrees(float(raw[AngleLandmark.left_knee]))
        self.pipeline = FilterPipeline([self.extractor, AngleCalibrator(self.calibration),
                                        LegDeviationExtractor(LegDeviationExtractorSettings()),
                                        TorsoTiltExtractor(TorsoTiltExtractorSettings())])

    def _frame(self, m: Measures) -> Frame:
        return Frame(0, 0, features={Points2D: Dummy.points(m, self.angle_settings.aspect_ratio)})

    def _raw(self, m: Measures) -> np.ndarray:
        return self.extractor.process(self._frame(m))[Angles].values

    def _read(self, m: Measures) -> Frame:
        return self.pipeline.process(self._frame(m))

    def test_neutral_reads_zero_on_every_joint(self) -> None:
        angles = self._read(NEUTRAL)[Angles]
        for name in JOINTS:
            with self.subTest(joint=name):
                self.assertAlmostEqual(angles[AngleLandmark[name]], 0.0, delta=0.02 * math.pi)

    def test_arms_raised_reads_pi_at_the_shoulders_and_zero_at_the_elbows(self) -> None:
        angles = self._read(UP)[Angles]
        for name in ('left_shoulder', 'right_shoulder'):
            with self.subTest(joint=name):
                self.assertAlmostEqual(abs(angles[AngleLandmark[name]]), math.pi, delta=0.02 * math.pi)
        for name in ('left_elbow', 'right_elbow'):
            with self.subTest(joint=name):
                self.assertAlmostEqual(angles[AngleLandmark[name]], 0.0, delta=0.02 * math.pi)

    def test_an_elbow_folded_half_a_turn_from_neutral_reads_pi(self) -> None:
        angles = self._read(from_neutral(left_elbow=-180.0, right_elbow=-180.0))[Angles]
        for name in ('left_elbow', 'right_elbow'):
            with self.subTest(joint=name):
                self.assertAlmostEqual(abs(angles[AngleLandmark[name]]), math.pi, delta=0.02 * math.pi)

    def test_the_torso_reads_as_the_bend(self) -> None:
        for torso, bend in ((0.0, 0.0), (22.5, 0.5), (45.0, 1.0), (-45.0, -1.0)):
            with self.subTest(torso=torso):
                self.assertAlmostEqual(self._read(Measures(torso=torso))[TorsoTilt].value, bend, delta=1e-3)

    def test_the_rows_read_as_the_results_table(self) -> None:
        def row(name: str) -> Frame:
            return self._read(Measures(**POSES_FILE[name]))

        def shoulders(f: Frame) -> tuple[float, float]:
            a = f[Angles]
            return abs(a[AngleLandmark.left_shoulder]) / math.pi, abs(a[AngleLandmark.right_shoulder]) / math.pi

        def elbows(f: Frame) -> tuple[float, float]:
            a = f[Angles]
            return abs(a[AngleLandmark.left_elbow]) / math.pi, abs(a[AngleLandmark.right_elbow]) / math.pi

        t = row('arms out level, a T')
        np.testing.assert_allclose(shoulders(t), (0.5, 0.5), atol=0.02)          # half registration by construction
        np.testing.assert_allclose(elbows(t), (0.0, 0.0), atol=0.02)
        np.testing.assert_allclose(shoulders(row('left arm up, right hanging')), (1.0, 0.0), atol=0.02)
        np.testing.assert_allclose(shoulders(row('right arm up, left hanging')), (0.0, 1.0), atol=0.02)
        np.testing.assert_allclose(elbows(row('a T, both elbows folded')), (1.0, 1.0), atol=0.02)
        np.testing.assert_allclose(elbows(row('a T, left elbow folded')), (1.0, 0.0), atol=0.02)
        np.testing.assert_allclose(elbows(row('a T, right elbow folded')), (0.0, 1.0), atol=0.02)
        for name, bend in (('a T, leaning left', -1.0), ('a T, leaning right', 1.0)):
            with self.subTest(pose=name):
                leaning = row(name)
                self.assertAlmostEqual(leaning[TorsoTilt].value, bend, delta=0.02)
                self.assertAlmostEqual(leaning[LegDeviation].value, 0.75, delta=0.05)     # the lean moves the hips' reading
        crouch = row('a T, in a crouch')
        self.assertAlmostEqual(crouch[LegDeviation].value, 1.0, delta=0.02)
        self.assertAlmostEqual(crouch[TorsoTilt].value, 0.0, delta=0.02)

    def test_the_legs_read_as_the_leg_deviation(self) -> None:
        for hips, knees, deviation in ((0.0, 0.0, 0.0), (-30.0, -45.0, 0.5), (-60.0, -90.0, 1.0)):   # from neutral
            with self.subTest(hips=hips, knees=knees):
                m = from_neutral(left_hip=hips, right_hip=hips, left_knee=knees, right_knee=knees)
                self.assertAlmostEqual(self._read(m)[LegDeviation].value, deviation, delta=0.02)


class DummyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'poses.json'
        self.path.write_text(json.dumps({'up': {'left_shoulder': 180.0, 'right_shoulder': 180.0}}), encoding='utf-8')
        self.cfg = DummySettings()
        self.cfg.morph = 1.0
        self.dummy = Dummy(self.cfg, AngleExtractorSettings(), AngleCalibratorSettings(), track_id=ID, poses_path=self.path)
        self.out: list[dict] = []
        self.dummy.add_frames_callback(self.out.append)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- the poses --

    def test_the_poses_file_fills_the_select(self) -> None:
        self.assertEqual(self.cfg.poses, ['up'])

    def test_the_shipped_poses_are_the_rows_of_the_results_table(self) -> None:
        self.assertEqual(tuple(POSES_FILE), ROWS)

    def test_the_dummy_starts_in_the_saved_neutral(self) -> None:
        self.path.write_text(json.dumps({'up': {'left_shoulder': 180.0}, 'neutral': {'left_shoulder': -20.0, 'left_hip': -10.0}}), encoding='utf-8')
        cfg = DummySettings()
        cfg.left_shoulder = 90.0                                  # where a preset left it
        dummy = Dummy(cfg, AngleExtractorSettings(), AngleCalibratorSettings(), track_id=ID, poses_path=self.path)
        out: list[dict] = []
        dummy.add_frames_callback(out.append)
        self.assertEqual(cfg.pose, 'neutral')
        self.assertEqual(cfg.left_shoulder, -20.0)
        self.assertEqual(cfg.left_hip, -10.0)
        cfg.enabled = True
        dummy.process({})
        points = Dummy.points(Measures(left_shoulder=-20.0, left_hip=-10.0), AngleExtractorSettings().aspect_ratio)
        np.testing.assert_allclose(out[0][ID][Points2D].values, points.values, atol=1e-6)     # in neutral at once, no morph

    def test_picking_a_pose_sets_the_measures(self) -> None:
        self.cfg.pose = 'up'
        self.assertEqual(self.cfg.left_shoulder, 180.0)
        self.assertEqual(self.cfg.right_shoulder, 180.0)

    def test_saving_adds_a_pose_to_the_file_and_the_select(self) -> None:
        self.cfg.left_elbow = 120.0
        self.cfg.name = 'bent'
        DummySettings.save.fire(self.cfg)
        self.assertEqual(self.cfg.poses, ['up', 'bent'])
        self.assertEqual(self.cfg.pose, 'bent')
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))['bent']['left_elbow'], 120.0)

    # -- the morph --

    def test_a_change_morphs_over_the_morph_time(self) -> None:
        self.dummy.update(0.0)
        self.cfg.left_shoulder = 90.0
        self.assertAlmostEqual(self.dummy.update(0.0).left_shoulder, 0.0, places=6)      # the change starts a morph
        self.assertAlmostEqual(self.dummy.update(0.5).left_shoulder, 45.0, places=6)     # half way at half time
        self.assertAlmostEqual(self.dummy.update(0.5).left_shoulder, 90.0, places=6)
        self.assertAlmostEqual(self.dummy.update(0.5).left_shoulder, 90.0, places=6)     # and stays

    def test_morph_zero_snaps(self) -> None:
        self.cfg.morph = 0.0
        self.cfg.left_shoulder = 90.0
        self.assertEqual(self.dummy.update(0.0).left_shoulder, 90.0)

    def test_circular_measures_take_the_shortest_route(self) -> None:
        self.cfg.morph = 0.0
        self.cfg.azimuth, self.cfg.left_shoulder = 350.0, 30.0
        self.dummy.update(0.0)                                           # settled at the start
        self.cfg.morph = 1.0
        self.cfg.azimuth, self.cfg.left_shoulder = 10.0, -30.0
        self.dummy.update(0.0)
        m = self.dummy.update(0.5)
        self.assertAlmostEqual(m.azimuth % 360.0, 0.0, places=6)          # through 0, not 180
        self.assertAlmostEqual(m.left_shoulder % 360.0, 0.0, places=6)    # through 0

    def test_the_distance_morphs_in_a_straight_line(self) -> None:
        self.cfg.morph = 0.0
        self.cfg.distance = 0.0
        self.dummy.update(0.0)                                           # settled at the start
        self.cfg.morph = 1.0
        self.cfg.distance = 1.0
        self.dummy.update(0.0)
        self.assertAlmostEqual(self.dummy.update(0.5).distance, 0.5, places=6)   # not a way round a circle

    def test_a_tie_goes_up_through_the_front_on_both_sides(self) -> None:
        self.dummy.update(0.0)
        self.cfg.left_shoulder = self.cfg.right_shoulder = 180.0
        self.dummy.update(0.0)
        m = self.dummy.update(0.5)
        self.assertAlmostEqual(m.left_shoulder, 90.0, places=6)
        self.assertAlmostEqual(m.right_shoulder, 90.0, places=6)

    def test_a_change_mid_morph_starts_from_where_it_is(self) -> None:
        self.dummy.update(0.0)
        self.cfg.left_shoulder = 90.0
        self.dummy.update(0.0)
        self.dummy.update(0.5)                                           # at 45
        self.cfg.left_shoulder = 0.0
        self.assertAlmostEqual(self.dummy.update(0.0).left_shoulder, 45.0, places=6)
        self.assertAlmostEqual(self.dummy.update(1.0).left_shoulder, 0.0, places=6)

    # -- the step --

    def test_disabled_passes_the_frames_through(self) -> None:
        frames = {0: Frame(0, 0)}
        self.dummy.process(frames)
        self.assertIs(self.out[-1], frames)

    def test_enabled_adds_the_dummy_at_its_id(self) -> None:
        self.cfg.enabled = True
        self.cfg.morph = 0.0
        self.cfg.azimuth = 90.0
        self.cfg.distance = 0.8
        self.cfg.left_shoulder = 180.0
        self.dummy.process({0: Frame(0, 0)})
        frames = self.out[-1]
        self.assertEqual(set(frames), {0, ID})
        f = frames[ID]
        self.assertEqual(f.track_id, ID)
        for ft in (Points2D, Azimuth, Distance, BBox, Angles):
            self.assertIn(ft, f)
        self.assertAlmostEqual(f[Azimuth].value, math.pi / 2, places=5)
        self.assertAlmostEqual(f[Distance].value, 0.8, places=5)
        self.assertGreater(abs(float(f[Angles].values[AngleLandmark.left_shoulder])), 0.8 * math.pi)   # raised, at the default calibration

    def test_solo_leaves_the_live_players_out(self) -> None:
        self.cfg.enabled = True
        self.cfg.solo = True
        self.dummy.process({0: Frame(0, 0)})
        self.assertEqual(set(self.out[-1]), {ID})

    def test_solo_without_enabled_passes_the_frames_through(self) -> None:
        self.cfg.solo = True
        frames = {0: Frame(0, 0)}
        self.dummy.process(frames)
        self.assertIs(self.out[-1], frames)

    def test_the_lerp_filters_derive_the_rest(self) -> None:
        self.cfg.enabled = True
        self.cfg.morph = 0.0
        self.cfg.torso = 45.0
        self.cfg.left_hip = self.cfg.right_hip = -60.0
        self.cfg.left_knee = self.cfg.right_knee = -90.0
        self.dummy.process({})
        filters = FilterPipeline([LegDeviationExtractor(LegDeviationExtractorSettings()),
                                  TorsoTiltExtractor(TorsoTiltExtractorSettings())])
        f = filters.process(self.out[-1][ID])
        self.assertAlmostEqual(f[LegDeviation].value, 1.0, delta=1e-3)
        self.assertAlmostEqual(f[TorsoTilt].value, 1.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
