"""smile's board: each slice is empty until written, then holds the newest it was given."""

import threading
import unittest

import numpy as np

from apps.smile.board import OnlineCell, RenderBoard
from apps.smile.face import FaceObservations


class TestRenderBoard(unittest.TestCase):
    def setUp(self) -> None:
        self.board = RenderBoard()

    def test_the_faces_are_the_newest_judged(self) -> None:
        self.assertIsNone(self.board.get_faces())
        first, second = FaceObservations((), 1.0), FaceObservations((), 2.0)
        self.board.set_faces(first)
        self.board.set_faces(second)
        self.assertIs(self.board.get_faces(), second)

    def test_the_capture_and_its_depth_are_kept_apart(self) -> None:
        self.assertIsNone(self.board.get_capture())
        self.assertIsNone(self.board.get_capture_depth())
        capture, depth = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)
        self.board.set_capture(capture)
        self.board.set_capture_depth(depth)
        self.assertIs(self.board.get_capture(), capture)
        self.assertIs(self.board.get_capture_depth(), depth)

    def test_the_generator_images_are_kept_apart(self) -> None:
        self.assertIsNone(self.board.get_diffusion_image())
        self.assertIsNone(self.board.get_diffusion_depth())
        image, depth = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)
        self.board.set_diffusion_image(image)
        self.board.set_diffusion_depth(depth)
        self.assertIs(self.board.get_diffusion_image(), image)
        self.assertIs(self.board.get_diffusion_depth(), depth)

    def test_the_status_and_the_warmup(self) -> None:
        self.assertIsNone(self.board.get_status())
        self.assertIsNone(self.board.get_warmup())
        self.board.set_status('no camera frames')
        self.board.set_warmup(3)
        self.assertEqual(self.board.get_status(), 'no camera frames')
        self.assertEqual(self.board.get_warmup(), 3)
        self.board.set_status(None)
        self.board.set_warmup(None)
        self.assertIsNone(self.board.get_status())
        self.assertIsNone(self.board.get_warmup())

    def test_the_online_cell_and_none(self) -> None:
        self.assertIsNone(self.board.get_online_cell())
        cell = OnlineCell(1, 12, 'qr', 'http://relay/?pass=a')
        self.board.set_online_cell(cell)
        self.assertEqual(self.board.get_online_cell(), cell)
        self.board.set_online_cell(None)
        self.assertIsNone(self.board.get_online_cell())

    def test_the_camera_image_is_the_module_slice(self) -> None:
        self.assertIsNone(self.board.get_camera_image(0))

    def test_writers_on_other_threads_leave_one_newest(self) -> None:
        images = [np.full((2, 2, 3), i, np.uint8) for i in range(200)]
        threads = [threading.Thread(target=lambda part=part: [self.board.set_capture(image) for image in part])
                   for part in (images[:100], images[100:])]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertTrue(any(self.board.get_capture() is image for image in images))


if __name__ == '__main__':
    unittest.main()
