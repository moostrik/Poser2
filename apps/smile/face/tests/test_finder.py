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
        self.finder = FaceFinder(FaceFinderSettings(), self.detector_settings, 100, 125)
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

    def test_the_calibration_sets_the_detectors_pitch_zero(self):
        self.finder.judge(frame_with((face((300.0, 300.0, 700.0, 800.0)),), 1.0, self.image))
        self.finder.settings.calibrate = True
        self.assertAlmostEqual(self.detector_settings.pitch_zero, 0.48, places=2)


if __name__ == '__main__':
    unittest.main()
