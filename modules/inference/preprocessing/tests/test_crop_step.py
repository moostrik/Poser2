"""The crop step with the pose model behind it: the crops cut from exactly the images of the poses' capture times,
never a newer upload, at every size from the same images and boxes, each size at its interval, the poses asked told
per camera, a snapshot no longer kept cut from nothing, the finished crops written; with the host model off the
camera's keypoints come out.
Runs on a CUDA install; the pose model is built, not loaded."""

import unittest

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import CropExtractor, CropExtractorSettings, ImageSnapshot, SerialUploader, SerialUploaderSettings, crop, pose
from modules.pose import BBox, Frame, PointLandmark, Points2D
from modules.utils import Rect


def crop_settings(width: int = 192, height: int = 256) -> CropExtractorSettings:
    settings = CropExtractorSettings()
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

    def upload_two(self, crops: CropExtractor) -> dict[int, float]:
        """The first image's capture times, a newer image uploaded after it; both snapshots kept by `crops`."""
        self.uploader.add_snapshot_callback(crops.set_snapshot)
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0))
        given = self.snapshots[-1][1]
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 2.0))
        return given

    def test_the_crop_is_cut_from_the_images_it_was_given(self) -> None:
        crops = CropExtractor(crop_settings())
        cut: list = []
        crops.add_image_callback(lambda frames, images: cut.append((frames, images)))
        crops.process({0: self.pose}, self.upload_two(crops))
        self.assertEqual(cut[0][0][0].timestamp, 1.0)            # stamped with the given image's capture time
        self.assertIn(0, cut[0][1])

    def test_every_size_is_cut_from_the_same_images_and_region(self) -> None:
        pose_crop, segmentation_crop = crop_settings(192, 256), crop_settings(768, 1024)
        crops = CropExtractor(pose_crop, segmentation_crop)
        cut: dict[str, tuple] = {}
        def cut_as(size: str):
            def on_cut(frames, images) -> None:
                cut.setdefault(size, (frames, images))
            return on_cut
        crops.add_image_callback(cut_as('pose'))
        crops.add_image_callback(cut_as('segmentation'), segmentation_crop)
        crops.process({0: self.pose}, self.upload_two(crops))
        self.assertEqual(tuple(cut['pose'][1][0].crop.shape), (3, 256, 192))
        self.assertEqual(tuple(cut['segmentation'][1][0].crop.shape), (3, 1024, 768))
        self.assertEqual(cut['segmentation'][0][0].timestamp, 1.0)                                  # the same images
        self.assertEqual(cut['segmentation'][0][0][BBox].to_rect(), cut['pose'][0][0][BBox].to_rect())   # one region
        self.assertIs(cut['segmentation'][1][0].ready, cut['pose'][1][0].ready)                      # one pass

    def test_the_poses_asked_are_told_per_camera_with_its_time(self) -> None:
        crops = CropExtractor(crop_settings())
        asked: list[tuple[int, float, set]] = []
        crops.add_asked_callback(lambda timestamp, cam_id, poses: asked.append((cam_id, timestamp, set(poses))))
        crops.process({0: self.pose}, self.upload_two(crops))
        self.assertEqual(asked, [(0, 1.0, {0})])

    def test_with_interval_two_every_other_submit_is_cut_and_the_others_told_not_asked(self) -> None:
        pose_crop, segmentation_crop = crop_settings(192, 256), crop_settings(768, 1024)
        pose_crop.interval = 2
        crops = CropExtractor(pose_crop, segmentation_crop)
        cut: dict[str, int] = {'pose': 0, 'segmentation': 0}
        def count(size: str):
            def on_cut(frames, images) -> None:
                cut[size] += 1
            return on_cut
        crops.add_image_callback(count('pose'))
        crops.add_image_callback(count('segmentation'), segmentation_crop)
        asked: list[set] = []
        crops.add_asked_callback(lambda timestamp, cam_id, poses: asked.append(set(poses)))
        cam_timestamps: dict[int, float] = self.upload_two(crops)
        for _ in range(4):
            crops.process({0: self.pose}, cam_timestamps)
        self.assertEqual(cut, {'pose': 2, 'segmentation': 4})
        self.assertEqual(asked, [{0}, set(), {0}, set()])

    def test_a_size_it_does_not_cut_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            CropExtractor(crop_settings()).add_image_callback(lambda frames, images: None, crop_settings())

    def test_a_snapshot_no_longer_kept_cuts_nothing_and_tells_the_poses_asked_none(self) -> None:
        crops = CropExtractor(crop_settings())
        cut: list = []
        crops.add_image_callback(lambda frames, images: cut.append(images))
        asked: list[tuple[int, float, set]] = []
        crops.add_asked_callback(lambda timestamp, cam_id, poses: asked.append((cam_id, timestamp, set(poses))))
        self.upload_two(crops)
        crops.process({0: self.pose}, {0: 0.5})
        self.assertEqual(cut, [])
        self.assertEqual(asked, [(0, 0.5, set())])

    def test_the_finished_crops_are_written_when_handed_on(self) -> None:
        crops = CropExtractor(crop_settings())
        finished: list[bool] = []

        def on_finished(frames, images) -> None:
            ready = images[0].ready
            assert ready is not None
            finished.append(ready.query())
        crops.add_finished_image_callback(on_finished)
        crops.process({0: self.pose}, self.upload_two(crops))
        self.assertEqual(finished, [True])

    def test_poses_without_capture_times_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            CropExtractor(crop_settings()).process({0: self.pose}, None)

    def test_with_the_host_model_off_the_cameras_keypoints_come_out(self) -> None:
        pose_settings = pose.PoseEstimatorSettings()
        pose_settings.initialize()
        pose_settings.enabled = False
        estimator = pose.PoseEstimator(pose_settings)
        crops = CropExtractor(crop_settings())
        crops.add_image_callback(estimator.submit)
        out: list = []
        estimator.add_frames_callback(out.append)
        self.uploader.add_snapshot_callback(crops.set_snapshot)
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0))
        n: int = len(PointLandmark)
        keypoints = Points2D.from_xy_arrays(np.full(n, 0.5), np.full(n, 0.5), np.full(n, 0.9, np.float32))
        with_points = Frame(track_id=0, cam_id=0, timestamp=1.0,
                            features={BBox: BBox.from_rect(Rect(0.25, 0.25, 0.5, 0.5)), Points2D: keypoints})
        without = Frame(track_id=1, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.1, 0.1, 0.2, 0.2))})
        crops.process({0: with_points, 1: without}, self.snapshots[-1][1])
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0]), {0})                      # the camera's pose, not the one without points
        self.assertTrue(np.allclose(out[0][0][Points2D].values, keypoints.values))


if __name__ == '__main__':
    unittest.main()
