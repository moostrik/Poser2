"""Tests for the HDF5 pose chunk format: write_chunk / read_chunk round-trip."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.pose.features import Age, AngleLandmark, Angles, Points2D
from modules.pose.recorder import read_chunk, write_chunk

from ._builders import frame, scalar, skeleton


class FrameIoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / 'sub' / 'pose_000.h5'

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip(self) -> None:
        pts = skeleton(left_elbow=0.4)
        angles = scalar(Angles, {AngleLandmark.head: 0.25, AngleLandmark.left_knee: -0.5}, score=0.7)
        entries = [
            (100.0, {0: frame(0, 100.0, cam_id=2, features={Points2D: pts, Angles: angles}),
                     1: frame(1, 100.0, cam_id=1, features={Points2D: pts})}),
            (100.5, {1: frame(1, 100.5, cam_id=0, features={Points2D: pts, Angles: angles})}),
        ]
        write_chunk(self.path, entries, recording_start=90.0, chunk_start=100.0, chunk_index=3,
                    feature_types=[Points2D, Angles])
        data = read_chunk(self.path)

        self.assertEqual((data['recording_start'], data['chunk_start'], data['chunk_index']), (90.0, 100.0, 3))
        np.testing.assert_array_equal(data['timestamps'], [100.0, 100.5])
        self.assertEqual(set(data['tracks']), {0, 1})

        t0, t1 = data['tracks'][0], data['tracks'][1]
        np.testing.assert_array_equal(t0['present'], [True, False])
        np.testing.assert_array_equal(t0['cam_ids'], [2, -1])
        np.testing.assert_array_equal(t1['present'], [True, True])
        np.testing.assert_array_equal(t1['cam_ids'], [1, 0])

        np.testing.assert_array_equal(t0['features']['Points2D']['values'][0], pts.values)
        np.testing.assert_array_equal(t0['features']['Points2D']['scores'][0], pts.scores)
        np.testing.assert_array_equal(t0['features']['Angles']['values'][0], angles.values)
        np.testing.assert_array_equal(t0['features']['Angles']['scores'][0], angles.scores)

    def test_absent_track_or_feature_is_nan_with_zero_scores(self) -> None:
        entries = [
            (0.0, {0: frame(0, 0.0, features={Points2D: skeleton()})}),
            (1.0, {}),
        ]
        write_chunk(self.path, entries, 0.0, 0.0, 0, feature_types=[Points2D, Angles])
        feats = read_chunk(self.path)['tracks'][0]['features']
        self.assertTrue(np.all(np.isnan(feats['Points2D']['values'][1])))       # track absent
        self.assertTrue(np.all(feats['Points2D']['scores'][1] == 0.0))
        self.assertTrue(np.all(np.isnan(feats['Angles']['values'][0])))         # feature absent
        self.assertTrue(np.all(feats['Angles']['scores'][0] == 0.0))

    def test_only_listed_features_are_written(self) -> None:
        entries = [(0.0, {0: frame(0, 0.0, features={Points2D: skeleton(), Age: Age.from_value(3.0)})})]
        write_chunk(self.path, entries, 0.0, 0.0, 0, feature_types=[Age])
        self.assertEqual(set(read_chunk(self.path)['tracks'][0]['features']), {'Age'})

    def test_no_entries_writes_nothing(self) -> None:
        write_chunk(self.path, [], 0.0, 0.0, 0, feature_types=[Points2D])
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
