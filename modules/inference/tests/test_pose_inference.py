"""PoseInference crops exactly the images it is handed with the poses, never a newer upload. Runs on a CUDA install;
the pose model is built, not loaded."""

import unittest

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import ImageSnapshot, PoseInference, SerialUploader, SerialUploaderSettings, crop, pose
from modules.pose import BBox, Frame, PointLandmark, Points2D
from modules.utils import Rect


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class PoseInferenceImagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = SerialUploader(SerialUploaderSettings())
        self.snapshots: list[ImageSnapshot] = []
        self.uploader.add_snapshot_callback(self.snapshots.append)
        crop_settings = crop.Settings()
        crop_settings.initialize()
        self.pose_settings = pose.Settings()
        self.pose_settings.initialize()
        self.inference = PoseInference(crop_settings, self.pose_settings)
        self.pose = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.25, 0.25, 0.5, 0.5))})

    def test_the_crop_is_cut_from_the_images_it_was_given(self) -> None:
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0))
        given = self.snapshots[-1]
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 2.0))    # a newer image
        cropped, crops = self.inference.cut_crops({0: self.pose}, given)
        self.assertEqual(cropped[0].timestamp, 1.0)             # stamped with the given image's capture time
        self.assertIn(0, crops)

    def test_poses_without_images_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            self.inference.cut_crops({0: self.pose}, None)

    def test_with_the_host_model_off_the_cameras_keypoints_come_out(self) -> None:
        self.pose_settings.enabled = False
        out: list = []
        self.inference.add_frames_callback(out.append)
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0))
        n: int = len(PointLandmark)
        keypoints = Points2D.from_xy_arrays(np.full(n, 0.5), np.full(n, 0.5), np.full(n, 0.9, np.float32))
        with_points = Frame(track_id=0, cam_id=0, timestamp=1.0,
                            features={BBox: BBox.from_rect(Rect(0.25, 0.25, 0.5, 0.5)), Points2D: keypoints})
        without = Frame(track_id=1, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.1, 0.1, 0.2, 0.2))})
        self.inference.submit({0: with_points, 1: without}, self.snapshots[-1])
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0]), {0})                      # the camera's pose, not the one without points
        self.assertTrue(np.allclose(out[0][0][Points2D].values, keypoints.values))


if __name__ == '__main__':
    unittest.main()
