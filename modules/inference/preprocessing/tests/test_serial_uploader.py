"""`nv12_to_rgb`: the colors the camera's NV12 frame stands for, on plain CPU tensors. The serial uploader's hand-off:
the camera's thread only queues, the uploader's thread hands each frame on in order, with its snapshot, and a full
queue drops the oldest of its kind (on a CUDA install). `ImageUpload`: a stack uploaded at once gives each image as
one by one does."""

import threading
import time
import unittest

import numpy as np
import torch

import torch.nn.functional as F

from modules.camera import CameraFrame, CameraTracklet, CameraTracklets, FrameType, TrackingStatus, upright_image
from modules.utils import Rect
from modules.inference.tests.wait import wait_for

from .. import GpuDepth, GpuImage, ImageSnapshot, ImageUpload, SerialUploader, SerialUploaderSettings
from ..image_upload import nv12_to_rgb, warp_grid
from ..serial_uploader import QUEUED


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
        chroma = torch.tensor([[90, 240, 128, 128]], dtype=torch.uint8)     # red, then gray
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


def _tracklet(roi: Rect, timestamp: float) -> CameraTracklet:
    return CameraTracklet(cam_id=0, id=1, status=TrackingStatus.TRACKED, roi=roi, age=1, timestamp=timestamp)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class HandOffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.uploader = SerialUploader(SerialUploaderSettings())
        self.handed: list[tuple[str, float, str]] = []      # kind, time, the thread it came on
        self.uploader.add_gpu_image_callback(
            lambda image: self.handed.append(('image', image.timestamp, threading.current_thread().name)))
        self.uploader.add_gpu_depth_callback(
            lambda depth: self.handed.append(('depth', depth.timestamp, threading.current_thread().name)))

    def tearDown(self) -> None:
        self.uploader.stop()

    def wait_for(self, count: int) -> None:
        wait_for(lambda: len(self.handed) >= count, 5.0)

    def test_frames_are_handed_on_in_order_on_the_uploader_s_thread(self) -> None:
        self.uploader.start()
        for t in (1.0, 2.0):
            self.uploader.submit_frame(_color(t))
            self.uploader.submit_frame(_depth(t))
        self.wait_for(4)
        self.assertEqual([(kind, t) for kind, t, _ in self.handed], [('image', 1.0), ('depth', 1.0), ('image', 2.0), ('depth', 2.0)])
        self.assertTrue(all(thread == 'SerialUploader' for _, _, thread in self.handed))

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
        frame = _color(3.0)
        self.uploader.submit_frame(frame)
        self.wait_for(1)
        images[0].ready.synchronize()
        self.assertTrue(images[0].image.is_cuda)
        self.assertEqual(tuple(images[0].image.shape), (3, 48, 64))
        self.assertIs(images[0].frame, frame)              # the camera frame it was uploaded from

    def test_the_depth_handed_on_is_meters_and_confidence_on_the_gpu_with_its_event(self) -> None:
        depths: list[GpuDepth] = []
        self.uploader.add_gpu_depth_callback(depths.append)
        self.uploader.start()
        frame = _depth(3.0)
        frame.image[:] = 1500
        sure = CameraFrame(0, FrameType.DEPTH, frame.image, 4.0, confidence=np.full((30, 40), 51, np.uint8))
        self.uploader.submit_frame(frame)
        self.uploader.submit_frame(sure)
        self.wait_for(2)
        depths[1].ready.synchronize()
        self.assertTrue(depths[0].depth.is_cuda)
        self.assertEqual(tuple(depths[0].depth.shape), (30, 40, 2))     # no alignment: on the color's view already
        self.assertAlmostEqual(float(depths[0].depth[..., 0].max()), 1.5, places=5)
        self.assertEqual(float(depths[0].depth[..., 1].max()), 1.0)    # the frame has none: sure
        self.assertAlmostEqual(float(depths[1].depth[..., 1].max()), 0.2, places=5)

    def test_tracklets_are_straightened_with_their_own_frame(self) -> None:
        handed: list[CameraTracklets] = []
        self.uploader.add_tracklet_callback(handed.append)
        shift = np.array([[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.uploader.submit_frame(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0, homography=shift))
        self.uploader.submit_frame(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 2.0, flip_h=True))
        box = Rect(0.1, 0.2, 0.3, 0.4)
        self.uploader.submit_tracklets(CameraTracklets(0, 1.0, [_tracklet(box, 1.0)]))   # the frame of 1.0, after the frame of 2.0
        self.uploader.submit_tracklets(CameraTracklets(0, 2.0, [_tracklet(box, 2.0)]))
        self.uploader.start()
        end = time.perf_counter() + 5.0
        while len(handed) < 2 and time.perf_counter() < end:
            time.sleep(0.005)
        self.assertAlmostEqual(handed[0].tracklets[0].roi.x, 0.35)       # shifted, not mirrored
        self.assertAlmostEqual(handed[1].tracklets[0].roi.x, 0.6)        # mirrored, not shifted
        self.assertTrue(all(message.straightened for message in handed))

    def test_tracklets_are_turned_and_flipped_with_their_frame(self) -> None:
        handed: list[CameraTracklets] = []
        self.uploader.add_tracklet_callback(handed.append)
        self.uploader.submit_frame(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0, rotate90=True, flip_v=True))
        self.uploader.submit_tracklets(CameraTracklets(0, 1.0, [_tracklet(Rect(0.1, 0.2, 0.3, 0.4), 1.0)]))
        self.uploader.start()
        end = time.perf_counter() + 5.0
        while not handed and time.perf_counter() < end:
            time.sleep(0.005)
        roi = handed[0].tracklets[0].roi
        self.assertAlmostEqual(roi.x, 0.2)      # transposed: x is the delivered y
        self.assertAlmostEqual(roi.y, 1.0 - 0.1 - 0.3)     # then flipped top to bottom

    def test_a_depth_frame_still_to_be_turned_is_refused_with_one_error(self) -> None:
        depths: list[GpuDepth] = []
        self.uploader.add_gpu_depth_callback(depths.append)
        self.uploader.start()
        with self.assertLogs('modules.inference.preprocessing.serial_uploader', 'ERROR') as logs:
            for t in (1.0, 2.0):
                self.uploader.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((30, 40), 1500, np.uint16), t, rotate90=True))
            time.sleep(0.2)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(depths, [])

    def test_tracklets_straightened_already_pass_as_they_are(self) -> None:
        handed: list[CameraTracklets] = []
        self.uploader.add_tracklet_callback(handed.append)
        self.uploader.submit_frame(CameraFrame(0, FrameType.VIDEO, np.zeros((48, 64, 3), np.uint8), 1.0, flip_h=True))
        message = CameraTracklets(0, 1.0, [_tracklet(Rect(0.1, 0.2, 0.3, 0.4), 1.0)], straightened=True)
        self.uploader.submit_tracklets(message)
        self.uploader.start()
        end = time.perf_counter() + 5.0
        while not handed and time.perf_counter() < end:
            time.sleep(0.005)
        self.assertIs(handed[0], message)

    def test_each_image_is_handed_on_as_a_snapshot_of_one(self) -> None:
        snapshots: list[ImageSnapshot] = []
        self.uploader.add_snapshot_callback(snapshots.append)
        self.uploader.start()
        self.uploader.submit_frame(_color(1.0))
        self.uploader.submit_frame(_color(2.0))
        self.wait_for(2)
        end = time.perf_counter() + 5.0
        while len(snapshots) < 2 and time.perf_counter() < end:
            time.sleep(0.005)
        first, second = snapshots
        self.assertEqual((first[1], second[1]), ({0: 1.0}, {0: 2.0}))
        self.assertIsNot(second[0][0], first[0][0])                      # each its own image

    def test_stop_ends_the_thread_with_frames_still_queued(self) -> None:
        self.uploader.start()
        for t in range(10):
            self.uploader.submit_frame(_color(float(t)))
        self.uploader.stop()
        self.assertFalse(self.uploader._thread.is_alive())


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class ImageUploadTest(unittest.TestCase):
    def test_a_stack_gives_each_image_as_one_by_one_does(self) -> None:
        rng = np.random.default_rng(0)
        shift = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        frames = [CameraFrame(0, FrameType.VIDEO, rng.integers(0, 255, (48, 64), np.uint8), 1.0),
                  CameraFrame(1, FrameType.VIDEO, rng.integers(0, 255, (48, 64), np.uint8), 1.0, homography=shift),
                  CameraFrame(2, FrameType.VIDEO, rng.integers(0, 255, (48, 64), np.uint8), 1.0, flip_h=True)]
        stream = torch.cuda.Stream()
        stacked, ready = ImageUpload().upload(frames, stream)
        ready.synchronize()
        single = ImageUpload()
        for frame, image in zip(frames, stacked):
            alone, alone_ready = single.upload([frame], stream)
            alone_ready.synchronize()
            self.assertEqual(tuple(image.shape), (3, 48, 64))
            self.assertTrue(torch.allclose(image.float(), alone[0].float(), atol=1e-3))

    def test_the_torch_path_turns_and_flips_as_upright_image_does(self) -> None:
        rng = np.random.default_rng(2)
        bgr = rng.integers(0, 255, (6, 10, 3), np.uint8)
        stream = torch.cuda.Stream()
        upload = ImageUpload()
        for rotate90, flip_v, flip_h in ((True, False, False), (False, True, False), (True, True, True)):
            frame = CameraFrame(0, FrameType.VIDEO, bgr, 1.0, rotate90=rotate90, flip_v=flip_v, flip_h=flip_h)
            images, ready = upload.upload([frame], stream, kernel=False)
            ready.synchronize()
            expected = torch.from_numpy(upright_image(frame)[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0     # RGB CHW
            self.assertEqual(tuple(images[0].shape), tuple(expected.shape))
            self.assertTrue(torch.allclose(images[0].float().cpu(), expected, atol=1e-3), (rotate90, flip_v, flip_h))

    def test_the_kernel_turns_and_flips_as_the_torch_path_does(self) -> None:
        rng = np.random.default_rng(3)
        shift = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        stream = torch.cuda.Stream()
        upload = ImageUpload()
        for image, nv12 in ((rng.integers(0, 255, (48, 64, 3), np.uint8), False), (rng.integers(0, 255, (72, 64), np.uint8), True)):
            for rotate90, flip_v, flip_h in ((True, False, False), (False, True, False), (True, True, True)):
                for homography in (None, shift):
                    frame = CameraFrame(0, FrameType.VIDEO, image, 1.0, nv12=nv12, homography=homography,
                                        rotate90=rotate90, flip_v=flip_v, flip_h=flip_h)
                    by_kernel, ready = upload.upload([frame], stream, kernel=True)
                    by_torch, torch_ready = upload.upload([frame], stream, kernel=False)
                    torch_ready.synchronize()
                    self.assertEqual(tuple(by_kernel[0].shape), tuple(by_torch[0].shape))
                    self.assertLessEqual(float((by_kernel[0].float() - by_torch[0].float()).abs().max()), 1.0 / 255.0)

    def test_an_nv12_stack_all_straightened_gives_each_image_as_one_by_one_does(self) -> None:
        rng = np.random.default_rng(1)
        shift = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        frames = [CameraFrame(cam, FrameType.VIDEO, rng.integers(0, 255, (72, 64), np.uint8), 1.0, nv12=True,
                              homography=shift, flip_h=cam == 1) for cam in range(3)]
        stream = torch.cuda.Stream()
        stacked, ready = ImageUpload().upload(frames, stream)
        ready.synchronize()
        single = ImageUpload()
        for frame, image in zip(frames, stacked):
            alone, alone_ready = single.upload([frame], stream)
            alone_ready.synchronize()
            self.assertEqual(tuple(image.shape), (3, 48, 64))
            self.assertTrue(torch.allclose(image.float(), alone[0].float(), atol=1e-3))


if __name__ == '__main__':
    unittest.main()
