"""Tests for TrackerFrameSource: which latched tracklets become pose frames, and what they carry."""

import math
import threading
import time
import unittest

import numpy as np

from modules.pose import Azimuth, BBox, BBoxAzimuth, Distance, Points2D
from modules.tracker import PanoramicAnnotation, TrackerFrameSource, TrackerFrameSourceSettings, Tracklet
from modules.utils import Rect


def stamps(seconds_ago: float = 0.0) -> dict[str, float]:
    """A tracklet's three times, `seconds_ago` before now."""
    t: float = time.perf_counter() - seconds_ago
    return {'timestamp': t, 'created_at': t, 'last_active': t}


def camera_0_now() -> dict[int, float]:
    """A set of one: camera 0's image captured now."""
    return {0: time.perf_counter()}


def tracklet(world_id: int, seconds_ago: float = 0.0, world_angle: float | None = None,
             zone_distance: float = math.nan, cam_id: int = 0) -> Tracklet:
    annotation = None if world_angle is None else PanoramicAnnotation(10.0, world_angle, False,
                                                                      zone_distance=zone_distance)
    return Tracklet(cam_id=cam_id, id=world_id, roi=Rect(x=0.4, y=0.2, width=0.1, height=0.6),
                    annotation=annotation, **stamps(seconds_ago))


class TrackerFrameSourceCase(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = TrackerFrameSourceSettings()
        self.settings.detection_timeout = 1.0
        self.poses = TrackerFrameSource(self.settings, max_players=4)


class TestDetectionTimeout(TrackerFrameSourceCase):
    """A person last detected longer ago than `detection_timeout` is not cropped; the tracker leaves
    that call to pose."""

    def test_a_fresh_box_is_posed(self) -> None:
        self.poses.set_tracklets({0: tracklet(0)})
        self.assertEqual(set(self.poses.process(camera_0_now()).keys()), {0})
        self.assertTrue(self.poses.is_ready(time.perf_counter()))

    def test_a_stale_box_is_not(self) -> None:
        self.poses.set_tracklets({0: tracklet(0, seconds_ago=1.5), 1: tracklet(1)})
        self.assertEqual(set(self.poses.process(camera_0_now()).keys()), {1})

    def test_a_latched_box_goes_stale_between_tracker_updates(self) -> None:
        # One `set_tracklets`, two frame bangs. Lowering `detection_timeout` below the box's age stands in
        # for time passing with no new update from the tracker.
        self.poses.set_tracklets({0: tracklet(0, seconds_ago=0.5)})
        self.assertEqual(set(self.poses.process(camera_0_now()).keys()), {0})
        self.settings.detection_timeout = 0.3
        self.assertEqual(self.poses.process(camera_0_now()), {})
        self.assertFalse(self.poses.is_ready(time.perf_counter()))

    def test_a_player_is_judged_on_its_own_cameras_time(self) -> None:
        now: float = time.perf_counter()
        self.poses.set_tracklets({0: tracklet(0, seconds_ago=0.5, cam_id=0), 1: tracklet(1, seconds_ago=0.5, cam_id=1)})
        # camera 1's image of the set was captured 0.6 s later than camera 0's: its player is 1.1 s old there
        self.assertEqual(set(self.poses.process({0: now, 1: now + 0.6})), {0})

    def test_a_player_whose_camera_is_not_in_the_set_is_not_posed(self) -> None:
        self.poses.set_tracklets({0: tracklet(0, cam_id=0), 1: tracklet(1, cam_id=1)})
        self.assertEqual(set(self.poses.process(camera_0_now())), {0})


class TestFrameContents(TrackerFrameSourceCase):

    def test_the_box_and_the_world_azimuth_are_carried(self) -> None:
        self.poses.set_tracklets({2: tracklet(2, world_angle=90.0)})
        frame = self.poses.process(camera_0_now())[2]
        rect = frame[BBox].to_rect()
        self.assertAlmostEqual(rect.bottom, 0.8, places=5)
        self.assertAlmostEqual(frame[BBoxAzimuth].value, math.pi / 2.0, places=5)
        self.assertNotIn(Azimuth, frame)          # the eye azimuth is derived downstream

    def test_the_distance_in_the_zone_is_carried(self) -> None:
        self.poses.set_tracklets({2: tracklet(2, world_angle=90.0, zone_distance=0.25)})
        distance = self.poses.process(camera_0_now())[2][Distance]
        self.assertAlmostEqual(distance.value, 0.25, places=6)
        self.assertEqual(distance.score, 1.0)

    def test_without_a_reading_the_distance_is_absent(self) -> None:
        self.poses.set_tracklets({1: tracklet(1, world_angle=90.0), 2: tracklet(2)})
        frames = self.poses.process(camera_0_now())
        for id in (1, 2):                         # no reading; no panoramic annotation
            with self.subTest(id=id):
                self.assertNotIn(Distance, frames[id])
                self.assertTrue(math.isnan(frames[id][Distance].value))
                self.assertEqual(frames[id][Distance].score, 0.0)

    def test_the_cameras_keypoints_are_normalised_to_the_host_crops_window(self) -> None:
        # A frame twice as wide as high; the box is a square quarter of it. The crop window, aspect
        # 1:2 about the box, is the frame's full height and a quarter of its width, and becomes the
        # pose's box; a keypoint at the box's top-left corner lands at the window's left edge, a
        # quarter down, and one at the box's centre in the window's centre.
        self.settings.crop_aspect = 0.5
        poses = TrackerFrameSource(self.settings, max_players=4, frame_aspect=2.0)
        keypoints = np.full((17, 3), (0.375, 0.5, 0.9), dtype=np.float32)
        keypoints[0, :2] = (0.25, 0.25)
        poses.set_tracklets({0: Tracklet(cam_id=0, id=0, roi=Rect(x=0.25, y=0.25, width=0.25, height=0.5),
                                         keypoints=keypoints, **stamps())})
        frame = poses.process(camera_0_now())[0]
        box = frame[BBox].to_rect()
        for got, want in zip((box.x, box.y, box.width, box.height), (0.25, 0.0, 0.25, 1.0)):
            self.assertAlmostEqual(got, want, places=5)
        points = frame[Points2D]
        np.testing.assert_allclose(points.values[0], (0.0, 0.25), atol=1e-5)
        np.testing.assert_allclose(points.values[1], (0.5, 0.5), atol=1e-5)
        self.assertAlmostEqual(float(points.scores[0]), 0.9, places=5)

    def test_an_id_beyond_the_slots_warns_and_is_not_posed(self) -> None:
        with self.assertLogs('modules.tracker.tracker_frame_source', level='WARNING'):
            self.poses.set_tracklets({5: tracklet(5)})
        self.assertEqual(self.poses.process(camera_0_now()), {})


class TestThread(TrackerFrameSourceCase):
    """`submit` queues a frame bang for the thread: its poses with its images, then the poses asked with its camera."""

    def setUp(self) -> None:
        super().setUp()
        self.order: list[tuple[str, str, object]] = []
        self.poses.add_frames_callback(lambda frames, images: self.order.append(('poses', threading.current_thread().name, images)))
        self.poses.add_asked_callback(lambda now, cam_id, asked: self.order.append(('asked', threading.current_thread().name, (cam_id, set(asked)))))
        self.poses.set_tracklets({0: tracklet(0)})

    def tearDown(self) -> None:
        self.poses.stop()

    def wait_for(self, count: int) -> None:
        end: float = time.perf_counter() + 1.0
        while len(self.order) < count and time.perf_counter() < end:
            time.sleep(0.001)

    def test_a_bang_runs_on_the_thread_with_its_images(self) -> None:
        self.poses.start()
        self.poses.submit(camera_0_now(), 'images 1')
        self.wait_for(2)
        self.assertEqual(self.order, [('poses', 'TrackerFrameSource', 'images 1'), ('asked', 'TrackerFrameSource', (0, {0}))])

    def test_beyond_two_waiting_the_oldest_is_dropped_and_counted(self) -> None:
        release: threading.Event = threading.Event()
        held: threading.Event = threading.Event()

        def hold(frames: object, images: object) -> None:
            if images == 'first':
                held.set()
                release.wait(2.0)
        self.poses.add_frames_callback(hold)
        self.poses.start()
        self.poses.submit(camera_0_now(), 'first')
        self.assertTrue(held.wait(1.0))                     # the thread is held: what comes now waits
        for name in ('a', 'b', 'c'):
            self.poses.submit(camera_0_now(), name)
        release.set()
        self.wait_for(6)
        self.assertEqual([images for kind, _, images in self.order if kind == 'poses'], ['first', 'b', 'c'])
        self.assertEqual(self.settings.dropped, 1)

    def test_a_bang_before_start_is_an_error_and_one_after_stop_is_dropped(self) -> None:
        with self.assertRaises(RuntimeError):
            self.poses.submit(camera_0_now(), 'early')
        self.poses.start()
        self.poses.stop()
        self.poses.submit(camera_0_now(), 'late')
        self.assertEqual(self.order, [])

    def test_the_poses_asked_are_told_per_camera_with_its_time(self) -> None:
        now: float = time.perf_counter()
        self.poses.set_tracklets({0: tracklet(0, cam_id=0), 1: tracklet(1, cam_id=1)})
        times: list[tuple[int, float, set]] = []
        self.poses.add_asked_callback(lambda timestamp, cam_id, asked: times.append((cam_id, timestamp, set(asked))))
        self.poses._bang({0: now, 1: now + 0.01}, 'set')          # the thread's work, on the test's thread
        self.assertEqual(times, [(0, now, {0}), (1, now + 0.01, {1})])

    def test_with_interval_two_every_other_bang_is_posed_and_the_others_told_not_asked(self) -> None:
        self.settings.pose_interval = 2
        for name in ('a', 'b', 'c', 'd'):
            self.poses._bang(camera_0_now(), name)
        self.assertEqual([images for kind, _, images in self.order if kind == 'poses'], ['a', 'c'])
        self.assertEqual([asked for kind, _, asked in self.order if kind == 'asked'], [(0, {0}), (0, set()), (0, {0}), (0, set())])


if __name__ == '__main__':
    unittest.main()
