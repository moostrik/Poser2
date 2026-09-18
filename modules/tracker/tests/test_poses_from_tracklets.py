"""Tests for PosesFromTracklets: which latched tracklets become pose frames, and what they carry."""

import math
import time
import unittest

from modules.pose.features import Azimuth, BBox, BBoxAzimuth, Distance
from modules.tracker import PanoramicAnnotation, PosesFromTracklets, PosesFromTrackletsSettings, Tracklet
from modules.utils import Rect


def tracklet(world_id: int, seconds_ago: float = 0.0, world_angle: float | None = None,
             zone_distance: float = math.nan) -> Tracklet:
    annotation = None if world_angle is None else PanoramicAnnotation(10.0, world_angle, False,
                                                                      zone_distance=zone_distance)
    return Tracklet(cam_id=0, id=world_id, roi=Rect(x=0.4, y=0.2, width=0.1, height=0.6),
                    last_active=time.time() - seconds_ago, annotation=annotation)


class PosesFromTrackletsCase(unittest.TestCase):

    def setUp(self) -> None:
        self.config = PosesFromTrackletsSettings()
        self.config.detection_timeout = 1.0
        self.poses = PosesFromTracklets(self.config, num_tracks=4)


class TestDetectionTimeout(PosesFromTrackletsCase):
    """A person last detected longer ago than `detection_timeout` is not cropped; the tracker leaves
    that call to pose."""

    def test_a_fresh_box_is_posed(self) -> None:
        self.poses.set_tracklets({0: tracklet(0)})
        self.assertEqual(set(self.poses.process().keys()), {0})
        self.assertTrue(self.poses.is_ready())

    def test_a_stale_box_is_not(self) -> None:
        self.poses.set_tracklets({0: tracklet(0, seconds_ago=1.5), 1: tracklet(1)})
        self.assertEqual(set(self.poses.process().keys()), {1})

    def test_a_latched_box_goes_stale_between_tracker_updates(self) -> None:
        # One `set_tracklets`, two frame bangs. Lowering `detection_timeout` below the box's age stands in
        # for time passing with no new update from the tracker.
        self.poses.set_tracklets({0: tracklet(0, seconds_ago=0.5)})
        self.assertEqual(set(self.poses.process().keys()), {0})
        self.config.detection_timeout = 0.3
        self.assertEqual(self.poses.process(), {})
        self.assertFalse(self.poses.is_ready())


class TestFrameContents(PosesFromTrackletsCase):

    def test_the_box_and_the_world_azimuth_are_carried(self) -> None:
        self.poses.set_tracklets({2: tracklet(2, world_angle=90.0)})
        frame = self.poses.process()[2]
        rect = frame[BBox].to_rect()
        self.assertAlmostEqual(rect.bottom, 0.8, places=5)
        self.assertAlmostEqual(frame[BBoxAzimuth].value, math.pi / 2.0, places=5)
        self.assertNotIn(Azimuth, frame)          # the eye azimuth is derived downstream

    def test_the_distance_in_the_zone_is_carried(self) -> None:
        self.poses.set_tracklets({2: tracklet(2, world_angle=90.0, zone_distance=0.25)})
        distance = self.poses.process()[2][Distance]
        self.assertAlmostEqual(distance.value, 0.25, places=6)
        self.assertEqual(distance.score, 1.0)

    def test_without_a_reading_the_distance_is_absent(self) -> None:
        self.poses.set_tracklets({1: tracklet(1, world_angle=90.0), 2: tracklet(2)})
        frames = self.poses.process()
        for id in (1, 2):                         # no reading; no panoramic annotation
            with self.subTest(id=id):
                self.assertNotIn(Distance, frames[id])
                self.assertTrue(math.isnan(frames[id][Distance].value))
                self.assertEqual(frames[id][Distance].score, 0.0)

    def test_an_id_beyond_the_slots_warns_and_is_not_posed(self) -> None:
        with self.assertLogs('modules.tracker.poses_from_tracklets', level='WARNING'):
            self.poses.set_tracklets({5: tracklet(5)})
        self.assertEqual(self.poses.process(), {})


if __name__ == '__main__':
    unittest.main()
