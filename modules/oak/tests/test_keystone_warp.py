"""The keystone builders must be bit-identical to the pre-step-3 `find_perspective_warp*`.

hd_trio and deep_flow ran on those builders at `perspective = -0.22` for years. Step 3 renamed
the control to `keystone` and added `tilt` beside it; nothing about those two installations may
change. The reference implementations below are copied from the last commit before step 3
(`git show 6ac6fb27:modules/oak/camera/pipeline.py`) and return plain tuples so the comparison
is on numbers, not on depthai objects.
"""

import unittest

import cv2
import numpy as np

from modules.oak.camera.pipeline import (
    WarpConfig, build_warp_mesh, find_keystone_warp, find_keystone_warp_square,
    KEYSTONE_MESH_W, KEYSTONE_MESH_H,
)
from modules.oak import WARP_MESH


# ── Reference: the pre-step-3 builders, verbatim apart from returning tuples ──────────────

def _reference_wide(width, height, width_offset, flip_h, flip_v, mesh_w, mesh_h):
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


def _reference_square(src_width, src_height, square_size, width_offset, flip_h, flip_v, mesh_w, mesh_h):
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


def _as_tuples(points) -> np.ndarray:
    """`dai.Point2f` stores float32, so the device only ever sees float32. Compare there."""
    return np.array([(p.x, p.y) for p in points], dtype=np.float32)


def _ref32(points) -> np.ndarray:
    return np.array(points, dtype=np.float32)


def _assert_same(test: unittest.TestCase, got: np.ndarray, want: np.ndarray) -> None:
    test.assertEqual(got.shape, want.shape)
    np.testing.assert_array_equal(got, want)         # exact, in float32, no tolerance


FLIPS = [(False, False), (True, False), (False, True), (True, True)]


class KeystoneIsTheOldBuilderTest(unittest.TestCase):

    def test_wide_matches_reference_bit_for_bit(self) -> None:
        # White Space's geometry, at the value the other apps used, every flip combination.
        for flip_h, flip_v in FLIPS:
            with self.subTest(flip_h=flip_h, flip_v=flip_v):
                offset = 1280 * 0.5 * -0.22
                got = _as_tuples(find_keystone_warp(1280, 800, offset, flip_h, flip_v, 2, 64))
                want = _reference_wide(1280, 800, offset, flip_h, flip_v, 2, 64)
                _assert_same(self, got, _ref32(want))

    def test_square_matches_reference_bit_for_bit(self) -> None:
        # hd_trio studio (720p colour) and umu (1080p colour), and deep_flow, at their values.
        cases = [(1280, 720, 720, -0.22), (1920, 1072, 1072, -0.22), (1280, 720, 720, 0.22)]
        for src_w, src_h, square, keystone in cases:
            for flip_h, flip_v in FLIPS:
                with self.subTest(src=(src_w, src_h), keystone=keystone, flip_h=flip_h, flip_v=flip_v):
                    offset = src_h * 0.5 * keystone
                    got = _as_tuples(find_keystone_warp_square(src_w, src_h, square, offset, flip_h, flip_v, 2, 64))
                    want = _reference_square(src_w, src_h, square, offset, flip_h, flip_v, 2, 64)
                    _assert_same(self, got, _ref32(want))

    def test_mesh_dimensions_are_the_old_ones(self) -> None:
        self.assertEqual((KEYSTONE_MESH_W, KEYSTONE_MESH_H), (2, 64))


class BuildWarpMeshSelectsTest(unittest.TestCase):
    """`build_warp_mesh` hands each camera the one correction it uses."""

    def _mount(self, tilt: float, keystone: float, flip_h: bool = True) -> WarpConfig:
        return WarpConfig(flip_h=flip_h, flip_v=False, tilt=tilt, keystone=keystone, fov_h=127.0)

    def test_keystone_alone_uses_the_old_builder_square(self) -> None:
        # deep_flow: 1280x720 mode, square 720 output, keystone -0.22, flip_h.
        mesh, mw, mh = build_warp_mesh((1280, 720), (720, 720), 1280, self._mount(0.0, -0.22))
        self.assertEqual((mw, mh), (2, 64))
        want = _reference_square(1280, 720, 720, 720 * 0.5 * -0.22, True, False, 2, 64)
        _assert_same(self, _as_tuples(mesh), _ref32(want))

    def test_keystone_alone_uses_the_old_builder_wide(self) -> None:
        mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(0.0, -0.22))
        self.assertEqual((mw, mh), (2, 64))
        want = _reference_wide(1280, 800, 1280 * 0.5 * -0.22, True, False, 2, 64)
        _assert_same(self, _as_tuples(mesh), _ref32(want))

    def test_tilt_alone_uses_the_dense_tilt_mesh(self) -> None:
        mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(15.0, 0.0))
        self.assertEqual((mw, mh), (WARP_MESH, WARP_MESH))
        self.assertEqual(len(mesh), WARP_MESH * WARP_MESH)

    def test_both_zero_is_the_equirectangular_reprojection_not_identity(self) -> None:
        # No keystone and no tilt still goes through the tilt path, which reprojects the
        # equidistant frame to equirectangular. The centre pixel is the fixed point; the corners
        # are not, because a fisheye's corners bow.
        mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(0.0, 0.0, flip_h=False))
        pts = _as_tuples(mesh).reshape(mh, mw, 2)
        cx, cy = 1279.0 / 2.0, 799.0 / 2.0
        # An even grid has no point ON the centre; the reprojection is symmetric about it, so
        # the mean of the four middle points is exactly the centre.
        h, w = mh // 2, mw // 2
        centre = pts[h - 1:h + 1, w - 1:w + 1].reshape(-1, 2).mean(axis=0)
        self.assertAlmostEqual(float(centre[0]), cx, places=2)
        self.assertAlmostEqual(float(centre[1]), cy, places=2)
        top_left = pts[0, 0]
        self.assertGreater(float(np.hypot(top_left[0] - 0.0, top_left[1] - 0.0)), 1.0)

    def test_both_set_warns_and_takes_tilt(self) -> None:
        with self.assertLogs("modules.oak.camera.pipeline", level="WARNING") as log:
            mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(10.0, -0.22))
        self.assertEqual((mw, mh), (WARP_MESH, WARP_MESH))
        self.assertTrue(any("exclusive" in line for line in log.output))


if __name__ == "__main__":
    unittest.main()
