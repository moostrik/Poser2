"""PoseInference crops exactly the images it is handed with the poses, never the uploader's newest. Runs on a CUDA
install; the pose model is built, not loaded."""

import unittest

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import PoseInference, Uploader, UploaderSettings, crop, pose
from modules.pose import BBox, Frame
from modules.utils import Rect


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class PoseInferenceImagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = Uploader(UploaderSettings())
        crop_settings = crop.Settings()
        crop_settings.initialize()
        pose_settings = pose.Settings()
        pose_settings.initialize()
        self.inference = PoseInference(crop_settings, pose_settings, self.uploader)
        self.pose = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.25, 0.25, 0.5, 0.5))})

    def test_the_crop_is_cut_from_the_images_it_was_given(self) -> None:
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0))
        given = self.uploader.snapshot()
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 2.0))    # a newer image
        cropped, crops = self.inference.cut_crops({0: self.pose}, given)
        self.assertEqual(cropped[0].timestamp, 1.0)             # stamped with the given image's capture time
        self.assertIn(0, crops)

    def test_poses_without_images_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            self.inference.cut_crops({0: self.pose}, None)


if __name__ == '__main__':
    unittest.main()
