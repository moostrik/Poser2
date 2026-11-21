"""Tests for smile's face finder: it judges a detector's frame, keeps its time, picks the subject and cuts the portrait
from that frame's own CPU frame (synthetic faces, no model, no thread)."""

import unittest
from typing import cast

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import GpuImage
from modules.inference.face import Face, FaceDetectorSettings, FaceFrame, head_pose

from .. import FaceFinder, FaceFinderSettings, FaceObservations
from .test_quality import keypoints


def frame_with(faces: tuple[Face, ...], timestamp: float, image: np.ndarray) -> FaceFrame:
    camera = CameraFrame(0, FrameType.VIDEO, image, timestamp)
    return FaceFrame(cam_id=0, timestamp=timestamp, width=image.shape[1], height=image.shape[0], faces=faces,
                     image=GpuImage(0, timestamp, torch.zeros(1), cast(torch.cuda.Event, None), camera))   # no GPU work: no event


def face(box: tuple[float, float, float, float], score: float = 0.9, eyes: float = 0.3) -> Face:
    kps = keypoints()
    kps = kps - kps.mean(axis=0) + np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    return Face(bbox=np.array(box), score=score, kps=kps, landmarks=np.zeros((106, 2)), pose=head_pose(kps), eyes=eyes, eye_dist=0.1)


class TestFaceFinder(unittest.TestCase):
    def setUp(self):
        self.detector_settings = FaceDetectorSettings()
        self.finder = FaceFinder(FaceFinderSettings(), self.detector_settings, 100, 125, (1000, 1250))
        self.observed: list[FaceObservations] = []
        self.finder.add_frame_callback(self.observed.append)
        self.image = np.full((1250, 1000, 3), 128, dtype=np.uint8)

    def test_the_faces_have_the_time_of_their_camera_frame(self):
        # not the time of the judgement: a camera that stopped must go stale
        self.finder.judge(frame_with((), 12.5, self.image))
        self.assertEqual(len(self.observed), 1)
        self.assertEqual(self.observed[0].timestamp, 12.5)
        self.assertFalse(self.observed[0].qualified)

    def test_the_largest_qualifying_face_is_the_subject(self):
        small, large = face((400.0, 400.0, 600.0, 650.0)), face((300.0, 300.0, 700.0, 800.0))
        observed = self.finder.judge(frame_with((small, large), 1.0, self.image))
        self.assertTrue(observed.qualified)
        self.assertAlmostEqual(observed.subject.bbox[2] - observed.subject.bbox[0], 0.4)      # the large one, 400 of 1000 px

    def test_closed_eyes_reject_the_face(self):
        observed = self.finder.judge(frame_with((face((300.0, 300.0, 700.0, 800.0), eyes=0.05),), 1.0, self.image))
        self.assertFalse(observed.qualified)

    def test_the_portrait_is_cut_from_the_qualifying_frames_own_cpu_frame(self):
        self.assertIsNone(self.finder.best())
        qualifying = np.full((1250, 1000, 3), 200, dtype=np.uint8)
        self.finder.judge(frame_with((face((300.0, 300.0, 700.0, 800.0)),), 1.0, qualifying))
        self.finder.judge(frame_with((face((300.0, 300.0, 700.0, 800.0), eyes=0.05),), 2.0, self.image))   # a later frame, rejected
        portrait = self.finder.best()
        self.assertIsNotNone(portrait)
        self.assertEqual(portrait.shape, (125, 100, 3))
        self.assertEqual(int(portrait[60, 50, 0]), 200)          # from the qualifying frame, not the later one
        self.finder.reset()
        self.assertIsNone(self.finder.best())

    def test_in_a_frame_larger_than_the_view_the_faces_are_judged_in_the_view(self):
        # 2000 x 1250 shown at 1000 x 1250: the view is the middle, x 500 to 1500
        wide = np.full((1250, 2000, 3), 90, dtype=np.uint8)
        wide[:, 800:1200] = 220                                      # the face's region, which the portrait is cut from
        outside = face((50.0, 300.0, 450.0, 800.0))                 # its center at x 250: not shown, not counted
        inside = face((800.0, 300.0, 1200.0, 800.0))
        observed = self.finder.judge(frame_with((outside, inside), 1.0, wide))
        self.assertEqual(len(observed.faces), 1)
        subject = observed.subject
        assert subject is not None
        self.assertAlmostEqual(subject.bbox[0], 0.3)                # (800 - 500) / 1000, in the view
        self.assertAlmostEqual(subject.bbox[2], 0.7)
        self.assertTrue(observed.qualified)
        portrait = self.finder.best()                               # cut from the full frame, at the face's place in it
        assert portrait is not None
        self.assertEqual(int(portrait[60, 50, 0]), 220)

    def test_a_crop_past_the_views_edge_is_outside(self):
        # the same face against the frame's edge would pass; against the view's it reaches past
        wide = np.full((1250, 2000, 3), 128, dtype=np.uint8)
        near_edge = face((520.0, 300.0, 920.0, 800.0))
        observed = self.finder.judge(frame_with((near_edge,), 1.0, wide))
        subject = observed.subject
        assert subject is not None
        self.assertLess(subject.edge, 0.0)
        self.assertFalse(observed.qualified)

    def test_the_calibration_sets_the_detectors_pitch_zero(self):
        self.finder.judge(frame_with((face((300.0, 300.0, 700.0, 800.0)),), 1.0, self.image))
        self.finder.settings.calibrate = True
        self.assertAlmostEqual(self.detector_settings.pitch_zero, 0.48, places=2)


if __name__ == '__main__':
    unittest.main()
