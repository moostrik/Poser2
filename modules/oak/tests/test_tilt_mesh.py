"""Tests for the tilt warp mesh and the lens-geometry helpers it rests on.

These cover `modules/oak/camera/definitions.py` only, which has no OpenCV dependency and opens
no device, so the whole warp model is testable without hardware. Before this the mesh builders
had no tests at all.
"""

import math
import unittest

import numpy as np

from modules.oak import (
    WARP_MESH, degrees_per_pixel, frame_fov, tilt_mesh_points,
    mono_frame_size, mode_size, frame_size,
)


MONO_W, MONO_H = 1280, 800
MONO_FOV_H = 127.0          # OAK-D Pro W, OV9282 wide, full 1280x800 readout


def mesh(tilt: float, *, flip_h: bool = False, flip_v: bool = False,
         src: tuple[int, int] = (MONO_W, MONO_H), out: tuple[int, int] | None = None,
         mode_width: int = MONO_W, fov_h: float = MONO_FOV_H,
         mesh_w: int = WARP_MESH, mesh_h: int = WARP_MESH) -> np.ndarray:
    return np.array(tilt_mesh_points(src, out or src, mode_width, fov_h, tilt,
                                     flip_h, flip_v, mesh_w, mesh_h))


class DegreesPerPixelTest(unittest.TestCase):
    """The one derived quantity everything else leans on."""

    def test_matches_the_published_vertical_field(self) -> None:
        # Both lenses are near-equidistant, so vfov = fov_h * rows / columns. The published
        # triples are the independent check on that model.
        _, ov9282_v = frame_fov(127.0, 1280, (1280, 800))
        self.assertAlmostEqual(ov9282_v, 79.5, delta=0.2)       # spec says 79.5
        _, imx378_v = frame_fov(95.0, 4056, (4056, 3040))
        self.assertAlmostEqual(imx378_v, 72.0, delta=1.0)       # spec says 72

    def test_survives_a_vertical_crop(self) -> None:
        # 720p mono is a pure vertical crop of the 800-row sensor: all 1280 columns kept.
        h_800, v_800 = frame_fov(127.0, 1280, (1280, 800))
        h_720, v_720 = frame_fov(127.0, 1280, (1280, 720))
        self.assertAlmostEqual(h_800, h_720, places=9)          # horizontal unchanged
        self.assertAlmostEqual(v_720, v_800 * 720 / 800, places=9)
        self.assertAlmostEqual(v_720, 71.44, delta=0.01)        # the old hand-set value was 71.6

    def test_square_crop_keeps_the_scale_and_squares_the_field(self) -> None:
        # A square crop removes columns; it must not change degrees-per-pixel.
        h, v = frame_fov(127.0, 1280, (800, 800))
        self.assertAlmostEqual(h, v, places=9)
        self.assertAlmostEqual(h, 127.0 * 800 / 1280, places=9)

    def test_zero_width_does_not_divide_by_zero(self) -> None:
        self.assertEqual(degrees_per_pixel(127.0, 0), 0.0)


class TiltMeshIdentityTest(unittest.TestCase):
    """The safety property: nothing moves until an angle is set."""

    def test_zero_tilt_is_the_exact_identity_grid(self) -> None:
        got = mesh(0.0)
        xs = np.linspace(0.0, MONO_W - 1.0, WARP_MESH)
        ys = np.linspace(0.0, MONO_H - 1.0, WARP_MESH)
        want = np.array([[x, y] for y in ys for x in xs])
        np.testing.assert_array_equal(got, want)                # bit-for-bit, not almost

    def test_zero_tilt_with_flips_is_the_exact_mirror(self) -> None:
        xs = np.linspace(0.0, MONO_W - 1.0, WARP_MESH)
        ys = np.linspace(0.0, MONO_H - 1.0, WARP_MESH)
        for flip_h in (False, True):
            for flip_v in (False, True):
                with self.subTest(flip_h=flip_h, flip_v=flip_v):
                    want = np.array([[(MONO_W - 1.0 - x) if flip_h else x,
                                      (MONO_H - 1.0 - y) if flip_v else y]
                                     for y in ys for x in xs])
                    np.testing.assert_array_equal(mesh(0.0, flip_h=flip_h, flip_v=flip_v), want)

    def test_mesh_length_matches_the_grid(self) -> None:
        self.assertEqual(len(mesh(0.0, mesh_w=8, mesh_h=5)), 40)


class TiltMeshGeometryTest(unittest.TestCase):
    """What makes the degrees mean degrees."""

    def test_centre_moves_by_exactly_the_tilt_angle(self) -> None:
        # Positive tilt = camera aimed up, so a level view reads from LOWER in the frame.
        dpp = degrees_per_pixel(MONO_FOV_H, MONO_W)
        for tilt in (5.0, 10.0, -7.5):
            with self.subTest(tilt=tilt):
                centre = mesh(tilt, mesh_w=3, mesh_h=3)[4]      # middle of a 3x3 grid
                self.assertAlmostEqual(centre[0], (MONO_W - 1) / 2.0, places=6)
                self.assertAlmostEqual(centre[1] - (MONO_H - 1) / 2.0, tilt / dpp, places=4)

    def test_a_column_stays_a_column_on_the_optical_axis(self) -> None:
        # The centre column is the axis of the rotation, so it may shift but never bend.
        got = mesh(12.0, mesh_w=3, mesh_h=9)
        centre_xs = got[1::3, 0]
        self.assertTrue(np.allclose(centre_xs, (MONO_W - 1) / 2.0, atol=1e-6))

    def test_tilt_is_antisymmetric(self) -> None:
        # Mirroring the frame vertically turns an up-tilt into a down-tilt. Reverse rows only,
        # not the flattened array, or the columns get mirrored too.
        up = mesh(9.0, mesh_w=5, mesh_h=5).reshape(5, 5, 2)
        down = mesh(-9.0, mesh_w=5, mesh_h=5).reshape(5, 5, 2)[::-1, :, :]
        cy = (MONO_H - 1) / 2.0
        np.testing.assert_allclose(up[..., 1] - cy, -(down[..., 1] - cy), atol=1e-6)
        np.testing.assert_allclose(up[..., 0], down[..., 0], atol=1e-6)

    def test_the_row_is_curved_not_straight(self) -> None:
        # This is the whole reason the mesh cannot be 2 columns wide. A straight source line
        # would have zero sagitta; the real one bows by tens of pixels.
        row = mesh(15.0, mesh_w=WARP_MESH, mesh_h=2)[:WARP_MESH]
        chord = np.linspace(row[0, 1], row[-1, 1], WARP_MESH)
        self.assertGreater(float(np.abs(row[:, 1] - chord).max()), 50.0)

    def test_thirty_two_columns_is_enough(self) -> None:
        # 32 x 32 must agree with a much denser mesh to well under a pixel, at the extreme of
        # the settable range. This is what licenses the chosen density.
        fine_n = (WARP_MESH - 1) * 4 + 1            # shares every 4th point with the coarse grid
        fine = mesh(30.0, mesh_w=fine_n, mesh_h=fine_n)
        coarse = mesh(30.0, mesh_w=WARP_MESH, mesh_h=WARP_MESH)
        fine_grid = fine.reshape(fine_n, fine_n, 2)[::4, ::4, :].reshape(-1, 2)
        self.assertLess(float(np.hypot(*(coarse - fine_grid).T).max()), 0.5)


class TiltMeshCropTest(unittest.TestCase):
    """The square crop is a window on the same lens, not a different one."""

    def test_square_crop_is_centred_and_identity_at_zero_tilt(self) -> None:
        got = mesh(0.0, src=(MONO_W, MONO_H), out=(MONO_H, MONO_H), mesh_w=3, mesh_h=3)
        x_off = (MONO_W - MONO_H) / 2.0
        self.assertAlmostEqual(got[0][0], x_off, places=6)       # top-left of the square
        self.assertAlmostEqual(got[0][1], 0.0, places=6)
        self.assertAlmostEqual(got[4][0], (MONO_W - 1) / 2.0, places=6)

    def test_square_crop_uses_the_uncropped_width_for_scale(self) -> None:
        # Same tilt, same degrees-per-pixel, whether or not the frame was cropped.
        dpp = degrees_per_pixel(MONO_FOV_H, MONO_W)
        centre = mesh(8.0, src=(MONO_W, MONO_H), out=(MONO_H, MONO_H), mesh_w=3, mesh_h=3)[4]
        self.assertAlmostEqual(centre[1] - (MONO_H - 1) / 2.0, 8.0 / dpp, places=4)


class FrameSizeTest(unittest.TestCase):
    """`mode_size` is the un-cropped frame; `frame_size` is what is delivered."""

    def test_mode_size_ignores_the_square_crop(self) -> None:
        self.assertEqual(mode_size(color=False), mono_frame_size(square=False))
        self.assertEqual(frame_size(False, square=True), (MONO_H, MONO_H))
        self.assertEqual(mode_size(color=False)[0], MONO_W)


if __name__ == "__main__":
    unittest.main()
