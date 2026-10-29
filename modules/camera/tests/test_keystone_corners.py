import unittest

from modules.camera import keystone_corners, square_crop


class TestKeystoneCorners(unittest.TestCase):
    """The four points the keystone pins are the OAK-D homography's: the top edge narrowed and the
    bottom widened by half the spread each side, mirrored by the flips (`oak.pipeline.find_keystone_warp`)."""

    def test_spread_and_flips(self) -> None:
        src, dst = keystone_corners(1000, 500, -0.2, False, False)
        self.assertEqual(src, ((0, 0), (1000, 0), (1000, 500), (0, 500)))
        self.assertEqual(dst, ((-100, 0), (1100, 0), (900, 500), (100, 500)))
        _, flipped = keystone_corners(1000, 500, -0.2, True, True)
        self.assertEqual(flipped, ((1100, 500), (-100, 500), (100, 0), (900, 0)))

    def test_square_spread_is_quoted_on_the_source_height(self) -> None:
        _, dst = keystone_corners(800, 800, 0.1, False, False, offset_base=800)
        self.assertEqual(dst[0], (40, 0))
        self.assertEqual(square_crop((1280, 800), 800), (240, 0, 800, 800))


if __name__ == '__main__':
    unittest.main()
