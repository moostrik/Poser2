"""`nv12_to_rgb`: the colors the camera's NV12 frame stands for, on plain CPU tensors."""

import unittest

import torch

import torch.nn.functional as F

from modules.inference.uploader import nv12_to_rgb, warp_grid


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


if __name__ == '__main__':
    unittest.main()
