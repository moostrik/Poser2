"""Tests for the dummy: the figure the pipeline reads back as its set degrees, the poses file,
the morph, and the merge step."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.pose.features import Angles, AngleLandmark, Azimuth, BBox, LegDeviation, Points2D, PointLandmark as P, TorsoTilt
from modules.pose.frame import Frame
from modules.pose.nodes import (AngleExtractor, AngleExtractorSettings, LegDeviationExtractor,
                                LegDeviationExtractorSettings, TorsoTiltExtractor, TorsoTiltExtractorSettings)
from modules.pose.trackers import FilterPipeline

from apps.white_space.pose import Dummy, DummySettings, Measures, dummy_id

POSES = Path('apps/white_space/data/poses.json')
JOINTS = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee')
ID = dummy_id(4)            # four live players: the dummy is 4, the ghosts start at 5


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class FigureTest(unittest.TestCase):
    """The pipeline's extractors read the figure back as the set degrees, outward as negative."""

    def setUp(self) -> None:
        self.angle_settings = AngleExtractorSettings()
        self.extractor = AngleExtractor(self.angle_settings)

    def _points(self, m: Measures) -> np.ndarray:
        return Dummy.points(m, self.extractor, self.angle_settings.aspect_ratio).values

    def _angles(self, m: Measures) -> np.ndarray:
        points = Dummy.points(m, self.extractor, self.angle_settings.aspect_ratio)
        return self.extractor.process(Frame(0, 0, features={Points2D: points}))[Angles].values

    def _assert_joints(self, m: Measures) -> None:
        read = self._angles(m)
        for name in JOINTS:
            with self.subTest(joint=name):
                self.assertAlmostEqual(wrap(float(read[AngleLandmark[name]]) + math.radians(getattr(m, name))), 0.0, delta=1e-3)

    def test_the_sides_are_named_as_the_pipeline_names_people(self) -> None:
        points = self._points(Measures())
        self.assertGreater(points[P.left_shoulder][0], points[P.right_shoulder][0])    # left on image-right
        self.assertGreater(points[P.left_hip][0], points[P.right_hip][0])

    def test_at_neutral_the_limbs_hang_beside_the_body(self) -> None:
        points = self._points(Measures())
        for shoulder, elbow, wrist, outward in ((P.left_shoulder, P.left_elbow, P.left_wrist, 1.0),
                                                (P.right_shoulder, P.right_elbow, P.right_wrist, -1.0)):
            self.assertGreater(points[elbow][1], points[shoulder][1])                  # below
            self.assertGreater(points[wrist][1], points[elbow][1])
            self.assertGreater((points[elbow][0] - points[shoulder][0]) * outward, -0.02)   # not across the body
        for hip, knee, ankle in ((P.left_hip, P.left_knee, P.left_ankle), (P.right_hip, P.right_knee, P.right_ankle)):
            self.assertLess(abs(points[knee][0] - points[hip][0]), 0.04)               # the leg under the hip
            self.assertLess(abs(points[ankle][0] - points[knee][0]), 0.04)

    def test_a_t_puts_the_arms_out_level(self) -> None:
        points = self._points(Measures(left_shoulder=90.0, right_shoulder=90.0))
        self.assertGreater(points[P.left_elbow][0], points[P.left_shoulder][0] + 0.1)     # outward
        self.assertLess(points[P.right_elbow][0], points[P.right_shoulder][0] - 0.1)
        self.assertAlmostEqual(points[P.left_wrist][1], points[P.left_shoulder][1], delta=0.1)    # about level (90° from the torso line)

    def test_a_crouch_moves_the_knees_out(self) -> None:
        points = self._points(Measures(left_hip=60.0, right_hip=60.0, left_knee=90.0, right_knee=90.0))
        self.assertGreater(points[P.left_knee][0], points[P.left_hip][0] + 0.05)
        self.assertLess(points[P.right_knee][0], points[P.right_hip][0] - 0.05)

    def test_neutral_reads_zero_on_every_joint(self) -> None:
        self._assert_joints(Measures())

    def test_the_rows_read_back_as_set(self) -> None:
        for name, values in json.loads(POSES.read_text(encoding='utf-8')).items():
            with self.subTest(pose=name):
                self._assert_joints(Measures(**values))

    def test_the_shoulder_reads_back_round_the_circle(self) -> None:
        for degrees in range(0, 360, 30):
            with self.subTest(degrees=degrees):
                self._assert_joints(Measures(left_shoulder=degrees, right_shoulder=(degrees + 90) % 360))

    def test_the_joints_read_back_with_the_torso_turned(self) -> None:
        self._assert_joints(Measures(torso=30.0, left_shoulder=90.0, right_shoulder=45.0, left_elbow=60.0,
                                     left_hip=40.0, right_knee=70.0))

    def test_equal_degrees_are_a_mirror_symmetric_figure(self) -> None:
        points = Dummy.points(Measures(left_shoulder=90.0, right_shoulder=90.0, left_elbow=45.0, right_elbow=45.0),
                              self.extractor, self.angle_settings.aspect_ratio).values
        for left, right in ((P.left_elbow, P.right_elbow), (P.left_wrist, P.right_wrist)):
            self.assertAlmostEqual(float(points[left][0] + points[right][0]), 1.0, places=5)     # mirrored about x = 0.5
            self.assertAlmostEqual(float(points[left][1]), float(points[right][1]), places=5)

    def test_the_torso_reads_as_the_bend(self) -> None:
        tilt = TorsoTiltExtractor(TorsoTiltExtractorSettings())
        for torso, bend in ((0.0, 0.0), (22.5, 0.5), (45.0, 1.0), (-45.0, -1.0)):
            with self.subTest(torso=torso):
                points = Dummy.points(Measures(torso=torso), self.extractor, self.angle_settings.aspect_ratio)
                self.assertAlmostEqual(tilt.process(Frame(0, 0, features={Points2D: points}))[TorsoTilt].value, bend, delta=1e-3)

    def test_the_legs_read_as_the_leg_deviation(self) -> None:
        legs = LegDeviationExtractor(LegDeviationExtractorSettings())
        for hips, knees, deviation in ((0.0, 0.0, 0.0), (30.0, 45.0, 0.5), (60.0, 90.0, 1.0)):
            with self.subTest(hips=hips, knees=knees):
                m = Measures(left_hip=hips, right_hip=hips, left_knee=knees, right_knee=knees)
                frame = self.extractor.process(Frame(0, 0, features={Points2D: Dummy.points(m, self.extractor, self.angle_settings.aspect_ratio)}))
                self.assertAlmostEqual(legs.process(frame)[LegDeviation].value, deviation, delta=1e-3)


class DummyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'poses.json'
        self.path.write_text(json.dumps({'up': {'left_shoulder': 180.0, 'right_shoulder': 180.0}}), encoding='utf-8')
        self.cfg = DummySettings()
        self.cfg.morph = 1.0
        self.dummy = Dummy(self.cfg, AngleExtractorSettings(), track_id=ID, poses_path=self.path)
        self.out: list[dict] = []
        self.dummy.add_frames_callback(self.out.append)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- the poses --

    def test_the_poses_file_fills_the_select(self) -> None:
        self.assertEqual(self.cfg.poses, ['up'])

    def test_the_shipped_poses_are_the_rows(self) -> None:
        poses = json.loads(POSES.read_text(encoding='utf-8'))
        self.assertEqual(len(poses), 10)
        self.assertEqual(poses['a T']['left_shoulder'], 90.0)
        self.assertEqual(poses['a T, in a crouch']['left_knee'], 90.0)

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
        self.cfg.azimuth, self.cfg.left_shoulder = 10.0, 330.0
        self.dummy.update(0.0)
        m = self.dummy.update(0.5)
        self.assertAlmostEqual(m.azimuth % 360.0, 0.0, places=6)          # through 0, not 180
        self.assertAlmostEqual(m.left_shoulder % 360.0, 0.0, places=6)    # through 0

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
        self.cfg.left_shoulder = 180.0
        self.dummy.process({0: Frame(0, 0)})
        frames = self.out[-1]
        self.assertEqual(set(frames), {0, ID})
        f = frames[ID]
        self.assertEqual(f.track_id, ID)
        for ft in (Points2D, Azimuth, BBox, Angles):
            self.assertIn(ft, f)
        self.assertAlmostEqual(f[Azimuth].value, math.pi / 2, places=5)
        self.assertAlmostEqual(wrap(float(f[Angles].values[AngleLandmark.left_shoulder]) - math.pi), 0.0, delta=1e-3)

    def test_the_lerp_filters_derive_the_rest(self) -> None:
        self.cfg.enabled = True
        self.cfg.morph = 0.0
        self.cfg.torso = 45.0
        self.cfg.left_hip = self.cfg.right_hip = 60.0
        self.cfg.left_knee = self.cfg.right_knee = 90.0
        self.dummy.process({})
        filters = FilterPipeline([LegDeviationExtractor(LegDeviationExtractorSettings()),
                                  TorsoTiltExtractor(TorsoTiltExtractorSettings())])
        f = filters.process(self.out[-1][ID])
        self.assertAlmostEqual(f[LegDeviation].value, 1.0, delta=1e-3)
        self.assertAlmostEqual(f[TorsoTilt].value, 1.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
