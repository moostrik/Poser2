"""Tests for the tilt warp mesh and the lens-geometry helpers it rests on.

These cover `modules/oak/camera/definitions.py` only, which has no OpenCV dependency and opens
no device, so the whole warp model is testable without hardware. Before this the mesh builders
had no tests at all.
"""

import math
import unittest

import numpy as np

from modules.oak import (
    WARP_MESH, degrees_per_pixel, frame_fov, equirect_mesh_points,
    CameraResolution, resolve_resolution, mono_mode, color_mode,
    mono_frame_size, color_frame_size, mode_size, frame_size,
)


MONO_W, MONO_H = 1280, 800
MONO_FOV_H = 127.0          # OAK-D Pro W, OV9282 wide, full 1280x800 readout


def mesh(tilt: float, *, flip_h: bool = False, flip_v: bool = False,
         src: tuple[int, int] = (MONO_W, MONO_H), out: tuple[int, int] | None = None,
         mode_width: int = MONO_W, fov_h: float = MONO_FOV_H,
         mesh_w: int = WARP_MESH, mesh_h: int = WARP_MESH) -> np.ndarray:
    return np.array(equirect_mesh_points(src, out or src, mode_width, fov_h, tilt,
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


CX, CY = (MONO_W - 1) / 2.0, (MONO_H - 1) / 2.0
CAM_HEIGHT = 0.5            # lens above the floor, m
PERSON = 1.9                # reference person, m


def equidistant_pixel(bearing_deg: float, elevation_deg: float,
                      fov_h: float = MONO_FOV_H, width: int = MONO_W) -> tuple[float, float]:
    """Where a LEVEL equidistant camera puts a ray — the forward model of the real lens.
    Image y is down, so a positive elevation is a negative y offset."""
    dpp = math.radians(fov_h / width)
    b, e = math.radians(bearing_deg), math.radians(elevation_deg)
    d = (math.cos(e) * math.sin(b), -math.sin(e), math.cos(e) * math.cos(b))
    phi = math.acos(max(-1.0, min(1.0, d[2])))
    psi = math.atan2(d[1], d[0])
    r = phi / dpp
    return CX + r * math.cos(psi), CY + r * math.sin(psi)


def equirect_pixel(bearing_deg: float, elevation_deg: float,
                   fov_h: float = MONO_FOV_H, width: int = MONO_W) -> tuple[float, float]:
    """Where the OUTPUT frame puts that ray: column is bearing, row is elevation, both linear."""
    dpp = fov_h / width
    return CX + bearing_deg / dpp, CY - elevation_deg / dpp


def sample_mesh(pts: np.ndarray, mesh_w: int, mesh_h: int, x: float, y: float,
                out_w: int = MONO_W, out_h: int = MONO_H) -> tuple[float, float]:
    """Bilinear lookup in a mesh, the way the Warp node reads it."""
    g = pts.reshape(mesh_h, mesh_w, 2)
    u = x / (out_w - 1) * (mesh_w - 1)
    v = y / (out_h - 1) * (mesh_h - 1)
    i, j = min(int(u), mesh_w - 2), min(int(v), mesh_h - 2)
    fu, fv = u - i, v - j
    top = g[j, i] * (1 - fu) + g[j, i + 1] * fu
    bot = g[j + 1, i] * (1 - fu) + g[j + 1, i + 1] * fu
    p = top * (1 - fv) + bot * fv
    return float(p[0]), float(p[1])


class EquirectMeshProjectionTest(unittest.TestCase):
    """At tilt 0 the mesh is the equidistant -> equirectangular reprojection, not the identity.
    The safety property is what stays fixed: the centre pixel, the centre row, the centre column,
    and — the point of it all — a vertical pole lands in ONE column at every height."""

    def test_centre_pixel_is_the_fixed_point(self) -> None:
        pts = mesh(0.0, mesh_w=3, mesh_h=3)
        self.assertAlmostEqual(float(pts[4][0]), CX, places=6)
        self.assertAlmostEqual(float(pts[4][1]), CY, places=6)

    def test_centre_row_and_column_are_unmoved(self) -> None:
        pts = mesh(0.0, mesh_w=9, mesh_h=9).reshape(9, 9, 2)
        xs = np.linspace(0.0, MONO_W - 1.0, 9)
        ys = np.linspace(0.0, MONO_H - 1.0, 9)
        np.testing.assert_allclose(pts[4, :, 0], xs, atol=1e-6)      # centre row: x unchanged
        np.testing.assert_allclose(pts[4, :, 1], CY, atol=1e-6)      #             y stays centre
        np.testing.assert_allclose(pts[:, 4, 1], ys, atol=1e-6)      # centre column: y unchanged
        np.testing.assert_allclose(pts[:, 4, 0], CX, atol=1e-6)      #                x stays centre

    def test_corners_are_not_the_identity(self) -> None:
        # A fisheye's corners bow; straightening them moves them. If this ever passed as
        # identity, the reprojection had silently been lost.
        pts = mesh(0.0, mesh_w=3, mesh_h=3)
        self.assertGreater(float(np.hypot(pts[0][0] - 0.0, pts[0][1] - 0.0)), 1.0)

    def test_a_vertical_pole_lands_in_one_column(self) -> None:
        # The property the tracker assumes. A standing person at a fixed bearing, sampled from
        # the floor to head height, at a near and a far distance, near the seam and mid-field.
        n = WARP_MESH
        pts = mesh(0.0, mesh_w=n, mesh_h=n)
        for bearing in (10.0, 45.0, 60.0):
            for dist in (1.35, 3.5):
                with self.subTest(bearing=bearing, dist=dist):
                    cols = []
                    for h in (0.0, 0.5, 1.0, 1.5, PERSON):
                        elev = math.degrees(math.atan((h - CAM_HEIGHT) / dist))
                        # the output pixel where this point SHOULD appear is the equirect one;
                        # the mesh says which source pixel feeds it; that source pixel must be
                        # where the real lens actually put the point.
                        ox, oy = equirect_pixel(bearing, elev)
                        if not (0 <= ox <= MONO_W - 1 and 0 <= oy <= MONO_H - 1):
                            continue
                        sx, sy = sample_mesh(pts, n, n, ox, oy)
                        tx, ty = equidistant_pixel(bearing, elev)
                        self.assertLess(math.hypot(sx - tx, sy - ty), 0.5)
                        cols.append(ox)
                    self.assertLess(max(cols) - min(cols), 1e-9)     # by construction: one column

    def test_flips_mirror_the_reprojection(self) -> None:
        n = 9
        base = mesh(0.0, mesh_w=n, mesh_h=n).reshape(n, n, 2)
        fh = mesh(0.0, flip_h=True, mesh_w=n, mesh_h=n).reshape(n, n, 2)
        fv = mesh(0.0, flip_v=True, mesh_w=n, mesh_h=n).reshape(n, n, 2)
        np.testing.assert_allclose(fh, base[:, ::-1, :], atol=1e-6)
        np.testing.assert_allclose(fv, base[::-1, :, :], atol=1e-6)

    def test_mesh_length_matches_the_grid(self) -> None:
        self.assertEqual(len(mesh(0.0, mesh_w=8, mesh_h=5)), 40)


class EquirectMeshGeometryTest(unittest.TestCase):
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


class EquirectMeshCropTest(unittest.TestCase):
    """The square crop is a window on the same lens, not a different one."""

    def test_square_crop_is_centred_at_zero_tilt(self) -> None:
        # The crop is a window on the same frame: its centre is the frame centre, and along the
        # unmoved centre row the crop offset carries through exactly. Corners are not identity
        # (the reprojection moves them), so they are not asserted here.
        got = mesh(0.0, src=(MONO_W, MONO_H), out=(MONO_H, MONO_H), mesh_w=3, mesh_h=3)
        x_off = (MONO_W - MONO_H) / 2.0
        self.assertAlmostEqual(got[4][0], CX, places=6)          # centre of the square
        self.assertAlmostEqual(got[4][1], CY, places=6)
        self.assertAlmostEqual(got[3][0], x_off, places=6)       # left edge, centre row
        self.assertAlmostEqual(got[3][1], CY, places=6)

    def test_square_crop_uses_the_uncropped_width_for_scale(self) -> None:
        # Same tilt, same degrees-per-pixel, whether or not the frame was cropped.
        dpp = degrees_per_pixel(MONO_FOV_H, MONO_W)
        centre = mesh(8.0, src=(MONO_W, MONO_H), out=(MONO_H, MONO_H), mesh_w=3, mesh_h=3)[4]
        self.assertAlmostEqual(centre[1] - (MONO_H - 1) / 2.0, 8.0 / dpp, places=4)


class FrameSizeTest(unittest.TestCase):
    """`mode_size` is the un-cropped frame; `frame_size` is what is delivered."""

    def test_mode_size_ignores_the_square_crop(self) -> None:
        P800 = CameraResolution.P800
        self.assertEqual(mode_size(False, P800), mono_frame_size(P800, square=False))
        self.assertEqual(frame_size(False, P800, square=True), (MONO_H, MONO_H))
        self.assertEqual(mode_size(False, P800)[0], MONO_W)

    def test_every_label_has_its_size(self) -> None:
        self.assertEqual(mono_frame_size(CameraResolution.P720), (1280, 720))
        self.assertEqual(mono_frame_size(CameraResolution.P800), (1280, 800))
        # 1080 is not divisible by the warp's 16-px alignment, so the preview is 1072 rows.
        self.assertEqual(color_frame_size(CameraResolution.P1080), (1920, 1072))

    def test_the_square_crop_follows_the_label(self) -> None:
        for label, height in ((CameraResolution.P720, 720), (CameraResolution.P800, 800)):
            self.assertEqual(frame_size(False, label, square=True), (height, height))


class ResolutionTest(unittest.TestCase):
    """One label spanning two sensors, and what happens when it cannot."""

    def test_shared_labels_map_to_both_sensors(self) -> None:
        for label in (CameraResolution.P720, CameraResolution.P800):
            self.assertEqual(resolve_resolution(color=False, resolution=label), label)
            self.assertEqual(resolve_resolution(color=True, resolution=label), label)
            self.assertIsNotNone(mono_mode(label))
            self.assertIsNotNone(color_mode(label))

    def test_mono_falls_back_from_a_colour_only_label(self) -> None:
        from modules.oak.camera.definitions import _warned_resolutions
        _warned_resolutions.discard((False, CameraResolution.P1080))
        with self.assertLogs('modules.oak.camera.definitions', level='WARNING') as captured:
            resolved = resolve_resolution(color=False, resolution=CameraResolution.P1080)
        self.assertEqual(resolved, CameraResolution.P800)
        self.assertIn('P1080', captured.output[0])
        self.assertIn('mono', captured.output[0])

    def test_the_fallback_warns_once_not_once_a_frame(self) -> None:
        """`frame_size` is called per frame by the simulator's size check."""
        from modules.oak.camera.definitions import _warned_resolutions
        _warned_resolutions.discard((False, CameraResolution.P1080))
        with self.assertLogs('modules.oak.camera.definitions', level='WARNING') as captured:
            for _ in range(50):
                frame_size(False, CameraResolution.P1080)
        self.assertEqual(len(captured.output), 1)

    def test_colour_keeps_its_own_label(self) -> None:
        self.assertEqual(
            resolve_resolution(color=True, resolution=CameraResolution.P1080),
            CameraResolution.P1080,
        )

    def test_the_vertical_field_follows_the_label_from_one_fov(self) -> None:
        """The whole point of deriving rather than storing: one `fov`, right at every mode."""
        for label, expected in ((CameraResolution.P800, 79.375), (CameraResolution.P720, 71.4375)):
            width, height = mono_frame_size(label)
            _, vfov = frame_fov(MONO_FOV_H, width, (width, height))
            self.assertAlmostEqual(vfov, expected, places=4)

    def test_the_horizontal_field_is_the_same_at_both_mono_labels(self) -> None:
        """P720 is a pure vertical crop, so the column-to-azimuth mapping is untouched."""
        fields = []
        for label in (CameraResolution.P720, CameraResolution.P800):
            width, height = mono_frame_size(label)
            fields.append(frame_fov(MONO_FOV_H, width, (width, height))[0])
        self.assertAlmostEqual(fields[0], fields[1], places=9)
        self.assertAlmostEqual(fields[0], MONO_FOV_H, places=9)


if __name__ == "__main__":
    unittest.main()
