"""`nv12_to_rgb`: the colors the camera's NV12 frame stands for, on plain CPU tensors. The uploader's hand-off: the
camera's thread only queues, the uploader's thread hands each frame on in order, and a full queue drops the oldest
of its kind (on a CUDA install)."""

import threading
import time
import unittest

import numpy as np
import torch

import torch.nn.functional as F

from modules.camera import CameraFrame, FrameType
from modules.inference import GpuDepth, GpuImage, Uploader, UploaderSettings
from modules.inference.uploader import QUEUED, nv12_to_rgb, warp_grid


def _plain(y: int, u: int, v: int) -> torch.Tensor:
    """A 2x2 frame of one color as RGB in 0..255."""
    luma = torch.full((2, 2), y, dtype=torch.uint8)
    chroma = torch.tensor([[u, v]], dtype=torch.uint8)
    return nv12_to_rgb(luma, chroma) * 255.0


class Nv12ToRgbTest(unittest.TestCase):
    def test_limited_range_black_white_and_grey(self) -> None:
        self.assertTrue(torch.allclose(_plain(16, 128, 128), torch.zeros(3, 2, 2), atol=0.5))
        self.assertTrue(torch.allclose(_plain(235, 128, 128), torch.full((3, 2, 2), 255.0), atol=0.5))

    def test_red_comes_out_red(self) -> None:
        red = _plain(81, 90, 240)       # BT.601 limited-range red
        self.assertTrue(torch.allclose(red[:, 0, 0], torch.tensor([255.0, 0.0, 0.0]), atol=2.0))

    def test_the_chroma_covers_every_pixel_of_its_block(self) -> None:
        luma = torch.full((2, 4), 81, dtype=torch.uint8)
        chroma = torch.tensor([[90, 240, 128, 128]], dtype=torch.uint8)     # red, then grey
        rgb = nv12_to_rgb(luma, chroma)
        self.assertEqual(rgb.shape, (3, 2, 4))
        self.assertGreater(float(rgb[0, 1, 1]), float(rgb[0, 1, 2]))


class WarpGridTest(unittest.TestCase):
    def test_the_identity_keeps_the_frame(self) -> None:
        image = torch.arange(12.0).reshape(1, 1, 3, 4)
        warped = F.grid_sample(image, warp_grid(torch.eye(3), 3, 4), align_corners=False)
        self.assertTrue(torch.allclose(warped, image, atol=1e-3))

    def test_a_shift_right_moves_the_picture_right(self) -> None:
        image = torch.zeros(1, 1, 1, 4)
        image[0, 0, 0, 1] = 1.0
        shift = torch.tensor([[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        warped = F.grid_sample(image, warp_grid(shift, 1, 4), align_corners=False)
        self.assertAlmostEqual(float(warped[0, 0, 0, 2]), 1.0, places=3)

    def test_a_pixel_from_behind_the_camera_is_empty(self) -> None:
        image = torch.ones(1, 1, 2, 2)
        behind = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        warped = F.grid_sample(image, warp_grid(behind, 2, 2), align_corners=False)
        self.assertEqual(float(warped.max()), 0.0)


def _color(timestamp: float, cam_id: int = 0) -> CameraFrame:
    return CameraFrame(cam_id, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), timestamp)


def _depth(timestamp: float, cam_id: int = 0) -> CameraFrame:
    return CameraFrame(cam_id, FrameType.DEPTH, np.zeros((30, 40), np.uint16), timestamp)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class HandOffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = Uploader(UploaderSettings())
        self.handed: list[tuple[str, float, str]] = []      # kind, time, the thread it came on
        self.uploader.add_gpu_image_callback(
            lambda image: self.handed.append(('image', image.timestamp, threading.current_thread().name)))
        self.uploader.add_gpu_depth_callback(
            lambda depth: self.handed.append(('depth', depth.timestamp, threading.current_thread().name)))

    def tearDown(self) -> None:
        self.uploader.stop()

    def wait_for(self, count: int) -> None:
        end = time.perf_counter() + 5.0
        while len(self.handed) < count and time.perf_counter() < end:
            time.sleep(0.005)

    def test_frames_are_handed_on_in_order_on_the_uploader_s_thread(self) -> None:
        self.uploader.start()
        for t in (1.0, 2.0):
            self.uploader.submit_frame(_color(t))
            self.uploader.submit_frame(_depth(t))
        self.wait_for(4)
        self.assertEqual([(kind, t) for kind, t, _ in self.handed], [('image', 1.0), ('depth', 1.0), ('image', 2.0), ('depth', 2.0)])
        self.assertTrue(all(thread == 'Uploader' for _, _, thread in self.handed))

    def test_submit_only_queues(self) -> None:
        self.uploader.submit_frame(_color(1.0))         # not started: nothing is handed on, nothing waits
        self.assertEqual(self.handed, [])

    def test_a_full_queue_drops_the_oldest_of_its_kind_and_counts_it(self) -> None:
        for t in range(QUEUED + 2):
            self.uploader.submit_frame(_color(float(t)))
        self.uploader.submit_frame(_depth(0.5))
        self.uploader.start()
        self.wait_for(QUEUED + 1)
        images = [t for kind, t, _ in self.handed if kind == 'image']
        self.assertEqual(images, [float(t) for t in range(2, QUEUED + 2)])     # the newest QUEUED
        self.assertIn(('depth', 0.5), [(kind, t) for kind, t, _ in self.handed])
        self.assertEqual(self.uploader.dropped, 2)

    def test_the_image_handed_on_is_on_the_gpu_with_its_event(self) -> None:
        images: list[GpuImage] = []
        self.uploader.add_gpu_image_callback(images.append)
        self.uploader.start()
        self.uploader.submit_frame(_color(3.0))
        self.wait_for(1)
        images[0].ready.synchronize()
        self.assertTrue(images[0].image.is_cuda)
        self.assertEqual(tuple(images[0].image.shape), (3, 48, 64))

    def test_the_depth_handed_on_is_metres_on_the_gpu_with_its_event(self) -> None:
        depths: list[GpuDepth] = []
        self.uploader.add_gpu_depth_callback(depths.append)
        self.uploader.start()
        frame = _depth(3.0)
        frame.image[:] = 1500
        self.uploader.submit_frame(frame)
        self.wait_for(1)
        depths[0].ready.synchronize()
        self.assertTrue(depths[0].depth.is_cuda)
        self.assertEqual(tuple(depths[0].depth.shape), (30, 40))        # no alignment: on the color's view already
        self.assertAlmostEqual(float(depths[0].depth.max()), 1.5, places=5)

    def test_stop_ends_the_thread_with_frames_still_queued(self) -> None:
        self.uploader.start()
        for t in range(10):
            self.uploader.submit_frame(_color(float(t)))
        self.uploader.stop()
        self.assertFalse(self.uploader.is_alive())


if __name__ == '__main__':
    unittest.main()
