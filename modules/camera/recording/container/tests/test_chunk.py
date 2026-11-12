"""A chunk's metadata: its lines written and read back."""

import math
import unittest

import numpy as np

from modules.camera import CameraTracklet, ChunkInfo, DepthAlignment, FrameMetadata, FrameReadings, TrackingStatus
from modules.camera.recording.container.chunk import frame_line, info_line, parse_metadata
from modules.camera.recording.tests.recordings import LENS
from modules.utils import Rect


class MetadataTest(unittest.TestCase):
    def test_the_lines_read_back(self) -> None:
        info = ChunkInfo(900, 720, LENS, True, 60.0)
        points = np.array([[0.1, 0.2, 0.9]] * 17, dtype=np.float32)
        tracklet = CameraTracklet(cam_id=2, id=4, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=7, timestamp=1.5, keypoints=points)
        box = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        text = (info_line(info) + frame_line(0, FrameMetadata(1.5, FrameReadings(-30.0, 1.0, 1000, 100), 0.01, [tracklet], 0.033,
                                                              {3: (box, points)}, 0.05))
                + frame_line(1, FrameMetadata(1.6, FrameReadings(math.nan, 1.0, 1000, 100), None, None, None))
                + frame_line(2, FrameMetadata(1.7, None, 0.01, [], 0.03)))
        metadata = parse_metadata(text, cam_id=2)
        self.assertEqual(metadata.info, info)
        first = metadata.frames[0]
        self.assertEqual((first.timestamp, first.readings, first.video_latency, first.tracklet_latency),
                         (1.5, FrameReadings(-30.0, 1.0, 1000, 100), 0.01, 0.033))
        assert first.tracklets is not None and first.tracklets[0].keypoints is not None
        self.assertEqual((first.tracklets[0].cam_id, first.tracklets[0].id, first.tracklets[0].status, first.tracklets[0].roi, first.tracklets[0].age),
                         (2, 4, TrackingStatus.TRACKED, Rect(0.1, 0.2, 0.3, 0.4), 7))
        np.testing.assert_array_almost_equal(first.tracklets[0].keypoints, points)
        assert first.poses is not None
        np.testing.assert_array_almost_equal(first.poses[3][0], box)
        np.testing.assert_array_almost_equal(first.poses[3][1], points)
        self.assertEqual(first.pose_latency, 0.05)
        self.assertIsNone(metadata.frames[1].poses)
        assert metadata.frames[1].readings is not None
        self.assertTrue(math.isnan(metadata.frames[1].readings.tilt))
        self.assertIsNone(metadata.frames[1].tracklets)
        self.assertIsNone(metadata.frames[2].readings)
        self.assertEqual(metadata.frames[2].tracklets, [])

    def test_the_depth_alignment_reads_back(self) -> None:
        alignment = DepthAlignment([[212.4, 0.0, 240.6], [0.0, 212.4, 149.4], [0.0, 0.0, 1.0]], LENS, (896, 720),
                                   [[1.0, 0.0, 0.0, -0.0376], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        info = ChunkInfo(896, 720, LENS, True, 60.0, alignment)
        self.assertEqual(parse_metadata(info_line(info)).info, info)

    def test_a_line_cut_off_by_a_crash_ends_the_list(self) -> None:
        text = info_line(ChunkInfo(900, 720, LENS, True, 60.0)) + frame_line(0, FrameMetadata(1.5)) + '{"frame": 1, "ti'
        self.assertEqual(len(parse_metadata(text).frames), 1)


if __name__ == '__main__':
    unittest.main()
