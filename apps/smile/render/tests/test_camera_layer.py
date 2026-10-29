"""Tests for the camera layer's warm-up block sizes (no GL needed)."""

import unittest

from apps.smile.render.layers.camera_layer import warmup_block


class TestWarmupBlock(unittest.TestCase):
    def test_the_display_size_halves_from_256(self):
        self.assertEqual([warmup_block(1024, 1280, step) for step in range(6)], [256, 128, 64, 32, 16, 8])

    def test_the_portrait_size_halves_from_224(self):
        self.assertEqual([warmup_block(896, 1120, step) for step in range(6)], [224, 112, 56, 28, 14, 7])

    def test_every_block_tiles_the_image_whole(self):
        for width, height in ((1080, 1920), (1366, 768), (900, 1125)):
            for step in range(8):
                block = warmup_block(width, height, step)
                self.assertEqual((width % block, height % block), (0, 0), (width, height, step))
        # no exact halving: the nearest common divisor, never a block that is cut at the edge
        self.assertEqual([warmup_block(1080, 1920, step) for step in range(6)], [120, 60, 30, 15, 8, 4])

    def test_a_far_step_is_single_pixels(self):
        self.assertEqual(warmup_block(1024, 1280, 20), 1)


if __name__ == '__main__':
    unittest.main()
