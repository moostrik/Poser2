"""The keystone's mesh must be bit-identical to the pre-step-3 `find_perspective_warp*`.

hd_trio and deep_flow ran on those builders at `perspective = -0.22` for years. Step 3 renamed
the control to `keystone` and added `tilt` beside it; nothing about those two installations may
change. The reference implementations below are copied from the last commit before step 3
(`git show 6ac6fb27:modules/oak/camera/pipeline.py`, kept verbatim since) and return plain tuples so the comparison
is on numbers, not on depthai objects.
"""

import unittest

import cv2
import numpy as np

from modules.camera.geometry import keystone_mesh_points


# ── Reference: the pre-step-3 builders, verbatim apart from returning tuples ──────────────

def reference_wide(width, height, width_offset, flip_h, flip_v, mesh_w, mesh_h):
    src_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    dst_points = np.array([[width_offset, 0], [width - width_offset, 0],
                           [width + width_offset, height], [-width_offset, height]], dtype=np.float32)
    if flip_h:
        dst_points[:, 0] = width - dst_points[:, 0]
    if flip_v:
        dst_points[:, 1] = height - dst_points[:, 1]
    H_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_points, dst_points))
    out = []
    for y in np.linspace(0, height - 1, mesh_h):
        for x in np.linspace(0, width - 1, mesh_w):
            src = H_inv @ np.array([x, y, 1.0])
            src /= src[2]
            out.append((float(src[0]), float(src[1])))
    return out


def reference_square(src_width, src_height, square_size, width_offset, flip_h, flip_v, mesh_w, mesh_h):
    square_size = square_size - 1
    x_offset = (src_width - square_size) / 2
    y_offset = (src_height - square_size) / 2
    corners = np.array([[x_offset, y_offset], [x_offset + square_size, y_offset],
                        [x_offset + square_size, y_offset + square_size],
                        [x_offset, y_offset + square_size]], dtype=np.float32)
    dst = np.array([[width_offset, 0], [square_size - width_offset, 0],
                    [square_size + width_offset, square_size], [-width_offset, square_size]], dtype=np.float32)
    if flip_h:
        dst[:, 0] = square_size - dst[:, 0]
    if flip_v:
        dst[:, 1] = square_size - dst[:, 1]
    H_inv = np.linalg.inv(cv2.getPerspectiveTransform(corners, dst))
    out = []
    for y in np.linspace(0, square_size - 1, mesh_h):
        for x in np.linspace(0, square_size - 1, mesh_w):
            src = H_inv @ np.array([x, y, 1.0])
            src /= src[2]
            out.append((float(src[0]), float(src[1])))
    return out


def as_float32(points) -> np.ndarray:
    """`dai.Point2f` stores float32, so the device only ever sees float32. Compare there."""
    return np.array(points, dtype=np.float32)


def assert_same(test: unittest.TestCase, got: np.ndarray, want: np.ndarray) -> None:
    test.assertEqual(got.shape, want.shape)
    np.testing.assert_array_equal(got, want)         # exact, in float32, no tolerance


FLIPS = [(False, False), (True, False), (False, True), (True, True)]


class KeystoneIsTheOldBuilderTest(unittest.TestCase):

    def test_wide_matches_reference_bit_for_bit(self) -> None:
        # White Space's geometry, at the value the other apps used, every flip combination.
        for flip_h, flip_v in FLIPS:
            with self.subTest(flip_h=flip_h, flip_v=flip_v):
                offset = 1280 * 0.5 * -0.22
                got = as_float32(keystone_mesh_points((0.0, 0.0, 1280.0, 800.0), offset, flip_h, flip_v, 2, 64))
                want = reference_wide(1280, 800, offset, flip_h, flip_v, 2, 64)
                assert_same(self, got, as_float32(want))

    def test_square_matches_reference_bit_for_bit(self) -> None:
        # hd_trio studio (720p color) and umu (1080p color), and deep_flow, at their values.
        cases = [(1280, 720, 720, -0.22), (1920, 1072, 1072, -0.22), (1280, 720, 720, 0.22)]
        for src_w, src_h, square, keystone in cases:
            for flip_h, flip_v in FLIPS:
                with self.subTest(src=(src_w, src_h), keystone=keystone, flip_h=flip_h, flip_v=flip_v):
                    offset = src_h * 0.5 * keystone
                    side = square - 1
                    rect = ((src_w - side) / 2, (src_h - side) / 2, float(side), float(side))
                    got = as_float32(keystone_mesh_points(rect, offset, flip_h, flip_v, 2, 64))
                    want = reference_square(src_w, src_h, square, offset, flip_h, flip_v, 2, 64)
                    assert_same(self, got, as_float32(want))


if __name__ == '__main__':
    unittest.main()
