"""A pose leaves the crop step stamped with the capture time of the image it was cut from. Runs on a CUDA install."""

import unittest

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import CropExtractor, Uploader, crop
from modules.pose import BBox, Frame, FrameDict
from modules.utils import Rect


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class CropStampTest(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = Uploader()
        self.settings = crop.Settings()
        self.settings.initialize()
        self.extractor = CropExtractor(self.settings)
        self.results: list[FrameDict] = []
        self.extractor.add_image_callback(lambda poses, crops: self.results.append(poses))

    def test_the_uploader_returns_each_image_with_its_capture_time(self) -> None:
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 12.5))
        images, _, timestamps, _ = self.uploader.snapshot()
        self.assertEqual(set(images), {0})
        self.assertEqual(timestamps, {0: 12.5})

    def test_a_pose_takes_the_time_of_the_image_it_is_cropped_from(self) -> None:
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 12.5))
        images, prev_images, timestamps, ready = self.uploader.snapshot()
        pose = Frame(track_id=0, cam_id=0, timestamp=12.4833, features={BBox: BBox.from_rect(Rect(0.25, 0.25, 0.5, 0.5))})
        self.extractor.process({0: pose}, images, prev_images, timestamps, ready)
        self.assertEqual(self.results[0][0].timestamp, 12.5)

    def test_the_crops_are_queued_with_the_event_their_readers_wait_on(self) -> None:
        crops: list = []
        self.extractor.add_image_callback(lambda poses, images: crops.append(images))
        image = np.zeros((48, 64, 3), np.uint8)
        image[:, 16:48] = 255                                    # white in the middle half of each row
        self.uploader._upload(CameraFrame(0, FrameType.VIDEO, image, 1.0))
        images, prev_images, timestamps, ready = self.uploader.snapshot()
        pose = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.25, 0.0, 0.5, 1.0))})
        self.extractor.process({0: pose}, images, prev_images, timestamps, ready)
        cut = crops[0][0]
        self.assertIsNotNone(cut.ready)
        reader = torch.cuda.Stream()
        reader.wait_event(cut.ready)
        with torch.cuda.stream(reader):
            middle = float(cut.crop[:, cut.crop.shape[1] // 2, cut.crop.shape[2] // 2].float().mean())
        self.assertAlmostEqual(middle, 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
