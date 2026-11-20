"""The batch uploader: a set in, its images and the set out on its own thread, the oldest whole set dropped beyond
its bound. Runs on a CUDA install."""

import threading
import time
import unittest

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType

from .. import BatchUploader, GpuImage, ImageSnapshot
from ..batch_uploader import WAITING


def _set(timestamp: float, cams: int = 4) -> dict[int, CameraFrame]:
    """A moment of `cams` mono cameras, each frame stamped `timestamp` plus its camera's offset."""
    return {cam_id: CameraFrame(cam_id, FrameType.VIDEO, np.full((48, 64), cam_id * 10, np.uint8), timestamp + cam_id * 0.001)
            for cam_id in range(cams)}


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class BatchUploaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = BatchUploader()
        self.images: list[GpuImage] = []
        self.snapshots: list[tuple[ImageSnapshot, str]] = []
        self.uploader.add_gpu_image_callback(self.images.append)
        self.uploader.add_snapshot_callback(lambda snapshot: self.snapshots.append((snapshot, threading.current_thread().name)))

    def tearDown(self) -> None:
        self.uploader.stop()

    def wait_for(self, count: int) -> None:
        end: float = time.perf_counter() + 5.0
        while len(self.snapshots) < count and time.perf_counter() < end:
            time.sleep(0.005)

    def test_a_set_comes_out_as_its_images_and_one_snapshot(self) -> None:
        self.uploader.start()
        self.uploader.submit_set(_set(1.0))
        self.wait_for(1)
        (images, timestamps, ready), thread = self.snapshots[0]
        ready.synchronize()
        self.assertEqual(thread, 'BatchUploader')
        self.assertEqual(sorted(images), [0, 1, 2, 3])
        self.assertAlmostEqual(timestamps[2], 1.002)
        self.assertAlmostEqual(float(images[3][0, 0, 0]), 30 / 255, places=3)     # each camera's own image
        self.assertEqual(sorted(image.cam_id for image in self.images), [0, 1, 2, 3])
        # each image keeps the camera frame it was uploaded from
        self.assertTrue(all(image.frame is not None and image.frame.cam_id == image.cam_id and image.frame.timestamp == image.timestamp
                            for image in self.images))

    def test_beyond_the_bound_the_oldest_whole_set_is_dropped_and_counted(self) -> None:
        for t in range(WAITING + 2):
            self.uploader.submit_set(_set(float(t)))
        self.uploader.start()
        self.wait_for(WAITING)
        times = [snapshot[1][0] for snapshot, _ in self.snapshots]
        self.assertEqual(times, [float(t) for t in range(2, WAITING + 2)])
        self.assertEqual(self.uploader.dropped, 2)
        self.assertTrue(all(len(snapshot[0]) == 4 for snapshot, _ in self.snapshots))      # never a set cut short


if __name__ == '__main__':
    unittest.main()
