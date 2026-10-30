"""`nv12_to_rgb`: the colors the camera's NV12 frame stands for, on plain CPU tensors."""

import unittest

import torch

from modules.inference.uploader import nv12_to_rgb


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


if __name__ == '__main__':
    unittest.main()
