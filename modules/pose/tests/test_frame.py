"""Tests for the pose Frame contract — NaN dummies for missing features, immutability, functional updates."""

import math
import unittest

import numpy as np

from modules.pose.features import Age, Angles, AngleLandmark, Points2D
from modules.pose.frame import Frame, reidentify, replace

from ._builders import scalar


class FrameAccessTest(unittest.TestCase):
    def test_missing_feature_is_nan_dummy_with_zero_scores(self) -> None:
        f = Frame(track_id=1, cam_id=2, time_stamp=3.0)
        angles = f[Angles]
        self.assertIsInstance(angles, Angles)
        self.assertTrue(np.all(np.isnan(angles.values)))
        self.assertTrue(np.all(angles.scores == 0.0))
        self.assertTrue(math.isnan(f[Age].value))

    def test_contains_and_len_count_only_set_features(self) -> None:
        f = Frame(0, 0, 0.0, {Age: Age.from_value(1.0)})
        _ = f[Angles]                                   # reading a missing feature does not add it
        self.assertIn(Age, f)
        self.assertNotIn(Angles, f)
        self.assertEqual(len(f), 1)

    def test_identity_slots(self) -> None:
        f = Frame(track_id=4, cam_id=2, time_stamp=12.5)
        self.assertEqual((f.track_id, f.cam_id, f.time_stamp), (4, 2, 12.5))

    def test_timestamp_defaults_to_now(self) -> None:
        self.assertGreater(Frame(0, 0).time_stamp, 0.0)


class FrameImmutabilityTest(unittest.TestCase):
    def test_setting_or_deleting_attributes_raises(self) -> None:
        f = Frame(0, 0, 0.0)
        with self.assertRaises(AttributeError):
            f.track_id = 5                              # type: ignore[misc]
        with self.assertRaises(AttributeError):
            del f._features

    def test_features_dict_is_copied_at_construction(self) -> None:
        features: dict = {Age: Age.from_value(1.0)}
        f = Frame(0, 0, 0.0, features)
        features[Angles] = scalar(Angles, {AngleLandmark.head: 0.1})
        self.assertNotIn(Angles, f)


class ReplaceTest(unittest.TestCase):
    def test_merges_updates_and_keeps_identity(self) -> None:
        age = Age.from_value(1.0)
        src = Frame(track_id=3, cam_id=1, time_stamp=7.0, features={Age: age})
        angles = scalar(Angles, {AngleLandmark.head: 0.2})
        out = replace(src, {Angles: angles})
        self.assertIs(out[Age], age)
        self.assertIs(out[Angles], angles)
        self.assertEqual((out.track_id, out.cam_id, out.time_stamp), (3, 1, 7.0))

    def test_overwrites_existing_feature(self) -> None:
        src = Frame(0, 0, 0.0, {Age: Age.from_value(1.0)})
        out = replace(src, {Age: Age.from_value(2.0)})
        self.assertAlmostEqual(out[Age].value, 2.0)

    def test_source_frame_is_unchanged(self) -> None:
        age = Age.from_value(1.0)
        src = Frame(0, 0, 0.0, {Age: age})
        replace(src, {Age: Age.from_value(2.0), Angles: Angles.create_dummy()})
        self.assertIs(src[Age], age)
        self.assertEqual(len(src), 1)


class ReidentifyTest(unittest.TestCase):
    def test_changes_only_track_id(self) -> None:
        pts = Points2D.create_dummy()
        src = Frame(track_id=1, cam_id=2, time_stamp=3.0, features={Points2D: pts})
        out = reidentify(src, 9)
        self.assertEqual((out.track_id, out.cam_id, out.time_stamp), (9, 2, 3.0))
        self.assertIs(out[Points2D], pts)
        self.assertEqual(src.track_id, 1)


if __name__ == "__main__":
    unittest.main()
