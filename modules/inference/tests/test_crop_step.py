"""The crop step with the pose model behind it: the crops cut from exactly the images handed with the poses, never a
newer upload, at every size from the same images and boxes; with the host model off the camera's keypoints come out.
Runs on a CUDA install; the pose model is built, not loaded."""

import unittest

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import CropExtractor, ImageSnapshot, SerialUploader, SerialUploaderSettings, crop, pose
from modules.pose import BBox, Frame, PointLandmark, Points2D
from modules.utils import Rect


def crop_settings(width: int = 192, height: int = 256) -> crop.CropExtractorSettings:
    settings = crop.CropExtractorSettings()
    settings.output_width, settings.output_height = width, height
    settings.initialize()
    return settings


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class CropStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = SerialUploader(SerialUploaderSettings())
        self.snapshots: list[ImageSnapshot] = []
        self.uploader.add_snapshot_callback(self.snapshots.append)
        self.pose = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.25, 0.25, 0.5, 0.5))})

    def upload_two(self) -> ImageSnapshot:
        """The first image's snapshot, a newer image uploaded after it."""
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0))
        given = self.snapshots[-1]
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 2.0))
        return given

    def test_the_crop_is_cut_from_the_images_it_was_given(self) -> None:
        crops = CropExtractor(crop_settings())
        cut: list = []
        crops.add_image_callback(lambda frames, images: cut.append((frames, images)))
        crops.submit({0: self.pose}, self.upload_two())
        self.assertEqual(cut[0][0][0].timestamp, 1.0)            # stamped with the given image's capture time
        self.assertIn(0, cut[0][1])

    def test_every_size_is_cut_from_the_same_images_and_region(self) -> None:
        pose_crop, segmentation_crop = crop_settings(192, 256), crop_settings(768, 1024)
        crops = CropExtractor(pose_crop, segmentation_crop)
        cut: dict[str, tuple] = {}
        crops.add_image_callback(lambda frames, images: cut.setdefault('pose', (frames, images)))
        crops.add_image_callback(lambda frames, images: cut.setdefault('segmentation', (frames, images)), segmentation_crop)
        crops.submit({0: self.pose}, self.upload_two())
        self.assertEqual(tuple(cut['pose'][1][0].crop.shape), (3, 256, 192))
        self.assertEqual(tuple(cut['segmentation'][1][0].crop.shape), (3, 1024, 768))
        self.assertEqual(cut['segmentation'][0][0].timestamp, 1.0)                                  # the same images
        self.assertEqual(cut['segmentation'][0][0][BBox].to_rect(), cut['pose'][0][0][BBox].to_rect())   # one region
        self.assertIs(cut['segmentation'][1][0].ready, cut['pose'][1][0].ready)                      # one pass

    def test_a_size_it_does_not_cut_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            CropExtractor(crop_settings()).add_image_callback(lambda frames, images: None, crop_settings())

    def test_poses_without_images_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            CropExtractor(crop_settings()).submit({0: self.pose}, None)

    def test_with_the_host_model_off_the_cameras_keypoints_come_out(self) -> None:
        pose_settings = pose.PoseEstimatorSettings()
        pose_settings.initialize()
        pose_settings.enabled = False
        estimator = pose.PoseEstimator(pose_settings)
        crops = CropExtractor(crop_settings())
        crops.add_image_callback(estimator.submit)
        out: list = []
        estimator.add_frames_callback(out.append)
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0))
        n: int = len(PointLandmark)
        keypoints = Points2D.from_xy_arrays(np.full(n, 0.5), np.full(n, 0.5), np.full(n, 0.9, np.float32))
        with_points = Frame(track_id=0, cam_id=0, timestamp=1.0,
                            features={BBox: BBox.from_rect(Rect(0.25, 0.25, 0.5, 0.5)), Points2D: keypoints})
        without = Frame(track_id=1, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.1, 0.1, 0.2, 0.2))})
        crops.submit({0: with_points, 1: without}, self.snapshots[-1])
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0]), {0})                      # the camera's pose, not the one without points
        self.assertTrue(np.allclose(out[0][0][Points2D].values, keypoints.values))


if __name__ == '__main__':
    unittest.main()
