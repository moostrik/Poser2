"""Tests for the warp mesh and the lens-geometry helpers it rests on.

These cover `modules/oak/camera/definitions.py` only, which has no OpenCV dependency and opens
no device, so the whole warp model is testable without hardware. The forward model of the real
lens (`equidistant_pixel`) is written independently here, in the plain form of the maths, so a
mistake in the module cannot pass by agreeing with itself.
"""

import math
import unittest

import numpy as np

from modules.oak import (
    WARP_MESH, degrees_per_pixel, frame_fov, warp_mesh_points, horizon_row,
    FrameWindow, frame_window, frame_coverage, coverage_summary,
    output_focal, source_lens, lens_field, lens_deviation,
    CameraResolution, resolve_resolution, mono_mode, color_mode,
    mono_frame_size, color_frame_size, mode_size, frame_size, full_frame_height, aligned_height,
)


MONO_W, MONO_H = 1280, 800
MONO_FOV_H = 127.0          # the delivered frame's azimuth span — the tracker's contract
CX, CY = (MONO_W - 1) / 2.0, (MONO_H - 1) / 2.0
DPP = MONO_FOV_H / MONO_W
F_OUT = MONO_W / math.radians(MONO_FOV_H)       # 577.5 px/rad, the output's focal

# The shared lens of the White Space rig, read off the four units' calibrations (see the lens
# geometry notes in definitions.py). The four units themselves, at 1280 x 800:
LENS_FOV = 128.9
LENS_CENTRE = (-10.5, 10.5)
UNITS = {                       # mxid tail: (focal px/rad, cx, cy)
    'F124D9D600': (575.4, 615.2, 409.9),
    '110AD3D200': (567.2, 634.3, 407.6),
    '31DDD2D200': (566.7, 632.6, 411.7),
    '1136D1D200': (565.0, 634.5, 410.3),
}

CAM_HEIGHT = 0.5            # lens above the floor, m
PERSON = 1.9                # reference person, m


def mesh(tilt: float, *, flip_h: bool = False, flip_v: bool = False,
         src: tuple[int, int] = (MONO_W, MONO_H), out: tuple[int, int] | None = None,
         mode_width: int = MONO_W, fov_h: float = MONO_FOV_H,
         mesh_w: int = WARP_MESH, mesh_h: int = WARP_MESH,
         lens_fov: float = 0.0, lens_centre: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    return np.array(warp_mesh_points(src, out or src, mode_width, fov_h, tilt,
                                     flip_h, flip_v, mesh_w, mesh_h, lens_fov, lens_centre))


def window(tilt: float, *, src: tuple[int, int] = (MONO_W, MONO_H), out: tuple[int, int] | None = None,
           lens_fov: float = 0.0, lens_centre: tuple[float, float] = (0.0, 0.0)) -> FrameWindow:
    return frame_window(src, out or src, MONO_W, MONO_FOV_H, tilt, lens_fov, lens_centre)


def equidistant_pixel(bearing_deg: float, elevation_deg: float, tilt_deg: float = 0.0,
                      focal: float = F_OUT, cx: float = CX, cy: float = CY) -> tuple[float, float]:
    """Where an equidistant camera aimed up by `tilt_deg` puts a world ray — the forward model of
    the real lens. Image y is down, so a positive elevation is a negative y offset.

    The tilt must be applied as a ROTATION of the ray, not by subtracting it from the elevation:
    turning an off-axis ray about the horizontal axis moves its bearing as well as its elevation.
    Subtracting instead is wrong by 14 px at 10 deg bearing and 129 px at 60 deg — the same family
    of error the mesh exists to correct.
    """
    b, e = math.radians(bearing_deg), math.radians(elevation_deg)
    d0, d1, d2 = math.cos(e) * math.sin(b), -math.sin(e), math.cos(e) * math.cos(b)
    t = math.radians(-tilt_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)
    d1, d2 = cos_t * d1 - sin_t * d2, sin_t * d1 + cos_t * d2
    phi = math.acos(max(-1.0, min(1.0, d2)))
    psi = math.atan2(d1, d0)
    r = phi * focal
    return cx + r * math.cos(psi), cy + r * math.sin(psi)


def output_pixel(win: FrameWindow, bearing_deg: float, elevation_deg: float,
                 out_w: int = MONO_W) -> tuple[float, float]:
    """Where the OUTPUT frame puts a world ray: the column is the bearing, linear; the row is
    the tangent of the elevation below the horizon row."""
    return (out_w - 1) / 2.0 + bearing_deg / DPP, win.row(elevation_deg)


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


class DegreesPerPixelTest(unittest.TestCase):
    """The horizontal scale everything else leans on."""

    def test_the_horizontal_field_is_the_contract(self) -> None:
        h_800, _ = frame_fov(127.0, 1280, (1280, 800), window(0.0))
        h_720, _ = frame_fov(127.0, 1280, (1280, 720), window(0.0, src=(1280, 720)))
        self.assertAlmostEqual(h_800, 127.0, places=9)
        self.assertAlmostEqual(h_720, 127.0, places=9)          # a vertical crop keeps every column

    def test_square_crop_keeps_the_scale_and_narrows_the_field(self) -> None:
        h, _ = frame_fov(127.0, 1280, (800, 800), window(0.0, out=(800, 800)))
        self.assertAlmostEqual(h, 127.0 * 800 / 1280, places=9)

    def test_output_focal_is_one_column_per_dpp(self) -> None:
        self.assertAlmostEqual(output_focal(127.0, 1280), 1.0 / math.radians(127.0 / 1280), places=9)
        self.assertEqual(output_focal(127.0, 0), 0.0)

    def test_zero_width_does_not_divide_by_zero(self) -> None:
        self.assertEqual(degrees_per_pixel(127.0, 0), 0.0)


class SourceLensTest(unittest.TestCase):
    """Three numbers, and what the frame reads off them."""

    def test_default_lens_is_the_old_model(self) -> None:
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H)
        self.assertAlmostEqual(focal, F_OUT, places=9)
        self.assertEqual((cx, cy), (CX, CY))

    def test_shared_lens_is_the_measured_one(self) -> None:
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        self.assertAlmostEqual(focal, 1280 / math.radians(128.9), places=9)
        self.assertAlmostEqual(focal, 568.9, delta=0.1)          # the four units' mean, 568.6, rounded to the step
        self.assertEqual((cx, cy), (CX - 10.5, CY + 10.5))

    def test_centre_offset_survives_the_720_crop(self) -> None:
        # The 720-row mode is a centre crop, so the offset from the frame centre is the same.
        _, cx8, cy8 = source_lens((1280, 800), 1280, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        _, cx7, cy7 = source_lens((1280, 720), 1280, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        self.assertEqual(cx8, cx7)
        self.assertAlmostEqual(cy8 - 399.5, cy7 - 359.5, places=9)

    def test_lens_field_reads_the_unit_from_its_focal(self) -> None:
        # The variant check: the wide lens spans ~127-130 deg across the frame, the narrow ~97.
        self.assertAlmostEqual(lens_field(UNITS['1136D1D200'][0], 1280), 129.8, delta=0.05)
        self.assertAlmostEqual(lens_field(UNITS['F124D9D600'][0], 1280), 127.5, delta=0.05)
        self.assertAlmostEqual(lens_field(756.0, 1280), 97.0, delta=0.1)
        self.assertTrue(math.isnan(lens_field(0.0, 1280)))

    def test_lens_deviation_names_the_outlier(self) -> None:
        # Unit F124's optical centre sits 24 px left of the frame centre, 14 px from the shared
        # lens's — 1.4 deg of azimuth zero. The other three are within 0.6 deg.
        errors = {tail: lens_deviation(*calib, (MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
                  for tail, calib in UNITS.items()}
        self.assertAlmostEqual(errors['F124D9D600'], 1.4, delta=0.05)
        for tail in ('110AD3D200', '31DDD2D200', '1136D1D200'):
            with self.subTest(unit=tail):
                self.assertLess(errors[tail], 0.6)

    def test_lens_deviation_is_zero_for_the_shared_lens_itself(self) -> None:
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        self.assertAlmostEqual(lens_deviation(focal, cx, cy, (MONO_W, MONO_H), MONO_W, MONO_FOV_H,
                                              LENS_FOV, LENS_CENTRE), 0.0, places=9)


class FrameWindowTest(unittest.TestCase):
    """Rows are tangents, and the window is pinned at the bottom."""

    def test_bottom_row_is_the_sensor_reach_below_the_axis(self) -> None:
        # Default lens: the sensor's bottom row is (H-1)/2 px = 39.64 deg below the axis.
        for tilt in (0.0, 16.0, -7.5):
            with self.subTest(tilt=tilt):
                win = window(tilt)
                self.assertAlmostEqual(win.elevation_bottom, tilt - math.degrees(CY / F_OUT), places=9)

    def test_rows_are_tangents_of_elevation(self) -> None:
        win = window(16.0, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
        self.assertAlmostEqual(win.row(win.elevation_bottom), MONO_H - 1, places=9)
        self.assertAlmostEqual(win.row(win.elevation_top), 0.0, places=9)
        self.assertAlmostEqual(win.row(0.0), win.horizon_px, places=9)
        for e in (-20.0, -5.0, 0.0, 12.0, 40.0):
            with self.subTest(elevation=e):
                self.assertAlmostEqual(win.horizon_px - win.row(e), F_OUT * math.tan(math.radians(e)), places=9)
                self.assertAlmostEqual(win.elevation(win.row(e)), e, places=9)

    def test_the_numbers_of_the_rig(self) -> None:
        # The shared lens at tilt 16 on 800 rows: the feet rule's -23.7 deg at the bottom
        # (-23.2 with the lens centre 10.5 px low), the tangent's 43.7 at the top.
        win = window(16.0, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
        self.assertAlmostEqual(win.elevation_bottom, -23.2, delta=0.05)
        self.assertAlmostEqual(win.elevation_top, 43.7, delta=0.05)
        self.assertAlmostEqual(win.horizon_px, 551.8, delta=0.1)
        self.assertAlmostEqual(horizon_row(win, MONO_H), 551.8 / 799.0, delta=0.001)
        _, vfov = frame_fov(MONO_FOV_H, MONO_W, (MONO_W, MONO_H), win)
        self.assertAlmostEqual(vfov, win.elevation_top - win.elevation_bottom, places=9)

    def test_the_horizon_may_leave_the_frame(self) -> None:
        # Tilt past the sensor's reach and the horizon is below the last row. Not an error here:
        # it is a mounting fact the coverage and the log make visible.
        win = window(45.0)
        self.assertGreater(win.horizon_px, MONO_H - 1)
        self.assertGreater(horizon_row(win, MONO_H), 1.0)

    def test_no_field_hands_the_frame_through(self) -> None:
        self.assertEqual(frame_window((MONO_W, MONO_H), (MONO_W, MONO_H), MONO_W, 0.0, 16.0).focal, 0.0)
        pts = mesh(16.0, fov_h=0.0, mesh_w=3, mesh_h=3)
        np.testing.assert_allclose(pts[4], (CX, CY), atol=1e-9)


class WarpMeshProjectionTest(unittest.TestCase):
    """At tilt 0 the mesh is the equidistant -> cylindrical reprojection, not the identity.
    The safety property is what stays fixed: the lens centre and the centre column, and — the
    point of it all — a vertical pole lands in ONE column at every height."""

    def test_the_lens_centre_is_the_fixed_point(self) -> None:
        win = window(0.0)
        n = 129
        sx, sy = sample_mesh(mesh(0.0, mesh_w=n, mesh_h=n), n, n, CX, win.row(0.0))
        self.assertAlmostEqual(sx, CX, delta=1e-3)
        self.assertAlmostEqual(sy, CY, delta=1e-3)

    def test_the_centre_column_is_straight(self) -> None:
        n = 9
        pts = mesh(0.0, mesh_w=n, mesh_h=n).reshape(n, n, 2)
        np.testing.assert_allclose(pts[:, 4, 0], CX, atol=1e-6)      # centre column: x stays centre
        self.assertTrue(np.all(np.diff(pts[:, 4, 1]) > 0))            # and runs down the sensor

    def test_corners_are_not_the_identity(self) -> None:
        # A fisheye's corners bow; straightening them moves them. If this ever passed as
        # identity, the reprojection had silently been lost.
        pts = mesh(0.0, mesh_w=3, mesh_h=3)
        self.assertGreater(float(np.hypot(pts[0][0] - 0.0, pts[0][1] - 0.0)), 1.0)

    def test_a_vertical_pole_lands_in_one_column(self) -> None:
        # The property the tracker assumes. A standing person at a fixed bearing, sampled from
        # the floor to head height, at a near and a far distance, near the seam and mid-field.
        # The output pixel where a point SHOULD appear is the cylindrical one; the mesh says which
        # source pixel feeds it; that source pixel must be where the real lens actually put it.
        n = WARP_MESH
        pts = mesh(0.0, mesh_w=n, mesh_h=n)
        win = window(0.0)
        for bearing in (10.0, 45.0, 60.0):
            for dist in (1.35, 3.5):
                with self.subTest(bearing=bearing, dist=dist):
                    cols = []
                    for h in (0.0, 0.5, 1.0, 1.5, PERSON):
                        elev = math.degrees(math.atan((h - CAM_HEIGHT) / dist))
                        ox, oy = output_pixel(win, bearing, elev)
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


class WarpMeshGeometryTest(unittest.TestCase):
    """What makes the degrees mean degrees, with the tilt and the real lens in."""

    def test_the_lens_centre_is_the_fixed_point_at_any_tilt(self) -> None:
        # The output's centre column at the row of elevation `tilt` looks along the optical axis,
        # so it reads the lens centre — not the frame centre — whatever the tilt.
        n = 129
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        for tilt in (0.0, 5.0, 16.0, -7.5):
            with self.subTest(tilt=tilt):
                win = window(tilt, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
                pts = mesh(tilt, mesh_w=n, mesh_h=n, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
                sx, sy = sample_mesh(pts, n, n, CX, win.row(tilt))
                # Sub-pixel, not places=6: the row of `tilt` falls between mesh rows, so the
                # lookup is bilinear. What matters is that the point does not MOVE.
                self.assertAlmostEqual(sx, cx, delta=0.01)
                self.assertAlmostEqual(sy, cy, delta=0.01)

    def test_the_bottom_row_reads_the_sensor_bottom_on_the_centre_column(self) -> None:
        # The window is pinned there: nothing is black at the bottom of the centre column, and
        # no sensor row below the axis is thrown away, at any tilt.
        n = 9
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        for tilt in (0.0, 8.0, 16.0, 30.0):
            with self.subTest(tilt=tilt):
                g = mesh(tilt, mesh_w=n, mesh_h=n, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE).reshape(n, n, 2)
                self.assertAlmostEqual(float(g[-1, n // 2, 1]), MONO_H - 1, places=6)
                np.testing.assert_allclose(g[:, n // 2, 0], cx, atol=1e-6)

    def test_the_horizon_row_reads_a_level_ray(self) -> None:
        # A level ray is `tilt` below the axis of a camera aimed up by `tilt`; the equidistant
        # lens puts it `focal * tilt` px below its centre. That is what must be found on the row
        # the window calls the horizon.
        n = 129
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        for tilt in (0.0, 8.0, 16.0):
            with self.subTest(tilt=tilt):
                win = window(tilt, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
                pts = mesh(tilt, mesh_w=n, mesh_h=n, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
                sx, sy = sample_mesh(pts, n, n, CX, win.horizon_px)
                self.assertAlmostEqual(sx, cx, places=3)
                self.assertAlmostEqual(sy, cy + focal * math.radians(tilt), delta=0.05)

    def test_rows_are_tangents_on_the_sensor_too(self) -> None:
        # On the centre column the source row of elevation `e` is `cy + focal * (tilt - e)` in
        # radians (equidistant), while the output row is `horizon - F * tan(e)`. The mesh must
        # join the two: a straight line in angle on one side, a tangent on the other.
        n = 129
        tilt = 16.0
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        win = window(tilt, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
        pts = mesh(tilt, mesh_w=n, mesh_h=n, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
        for e in (-20.0, -10.0, 0.0, 10.0, 25.0, 40.0):
            with self.subTest(elevation=e):
                _, sy = sample_mesh(pts, n, n, CX, win.row(e))
                self.assertAlmostEqual(sy, cy + focal * math.radians(tilt - e), delta=0.05)

    def test_a_vertical_pole_still_lands_in_one_column_when_tilted(self) -> None:
        # The property the tracker rests on has to survive the tilt and the lens offset. The
        # source pixel is where a camera aimed up by `tilt` with the REAL lens puts the ray.
        n = WARP_MESH
        tilt = 16.0
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        win = window(tilt, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
        pts = mesh(tilt, mesh_w=n, mesh_h=n, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
        for bearing in (10.0, 45.0, 60.0):
            for dist in (1.35, 3.5):
                with self.subTest(bearing=bearing, dist=dist):
                    cols = []
                    for h in (0.0, 0.5, 1.0, 1.5, PERSON):
                        elev = math.degrees(math.atan((h - CAM_HEIGHT) / dist))
                        ox, oy = output_pixel(win, bearing, elev)
                        if not (0 <= ox <= MONO_W - 1 and 0 <= oy <= MONO_H - 1):
                            continue
                        sx, sy = sample_mesh(pts, n, n, ox, oy)
                        tx, ty = equidistant_pixel(bearing, elev, tilt, focal, cx, cy)
                        self.assertLess(math.hypot(sx - tx, sy - ty), 0.5)
                        cols.append(ox)
                    self.assertGreaterEqual(len(cols), 3)
                    self.assertLess(max(cols) - min(cols), 1e-9)     # by construction: one column

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
        for tilt in (0.0, 16.0, 30.0):
            with self.subTest(tilt=tilt):
                fine = mesh(tilt, mesh_w=fine_n, mesh_h=fine_n, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
                coarse = mesh(tilt, mesh_w=WARP_MESH, mesh_h=WARP_MESH, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
                fine_grid = fine.reshape(fine_n, fine_n, 2)[::4, ::4, :].reshape(-1, 2)
                self.assertLess(float(np.hypot(*(coarse - fine_grid).T).max()), 0.5)


class WarpMeshCropTest(unittest.TestCase):
    """The square crop is a window on the same lens, not a different one."""

    def test_square_crop_keeps_the_centre_and_the_vertical_mapping(self) -> None:
        # The crop removes columns only: its centre column is the full frame's, and its rows
        # map exactly as the full frame's do, at any tilt.
        n = 9
        for tilt in (0.0, 8.0):
            with self.subTest(tilt=tilt):
                full = mesh(tilt, mesh_w=n, mesh_h=n, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE).reshape(n, n, 2)
                crop = mesh(tilt, src=(MONO_W, MONO_H), out=(MONO_H, MONO_H), mesh_w=n, mesh_h=n,
                            lens_fov=LENS_FOV, lens_centre=LENS_CENTRE).reshape(n, n, 2)
                np.testing.assert_allclose(crop[:, 4, :], full[:, 4, :], atol=1e-6)

    def test_square_crop_narrows_the_bearing_range(self) -> None:
        # The crop's edge column is at +/- (800-1)/2 columns of the full field, read off the same
        # lens. A dense mesh: the horizon row falls between mesh rows, and the fisheye bow makes a
        # coarse bilinear lookup there 10 px off.
        n = 129
        crop = mesh(0.0, src=(MONO_W, MONO_H), out=(MONO_H, MONO_H), mesh_w=n, mesh_h=n)
        win = window(0.0, out=(MONO_H, MONO_H))
        half = (MONO_H - 1) / 2.0 * DPP
        sx, sy = sample_mesh(crop, n, n, MONO_H - 1.0, win.row(0.0), MONO_H, MONO_H)
        tx, ty = equidistant_pixel(half, 0.0)
        self.assertAlmostEqual(sx, tx, delta=0.1)
        self.assertAlmostEqual(sy, ty, delta=0.1)


class FrameCoverageTest(unittest.TestCase):
    """Where the sensor did not look, and where it did."""

    def _coverage(self, tilt: float, flip_h: bool = False) -> np.ndarray:
        return frame_coverage((MONO_W, MONO_H), (MONO_W, MONO_H), MONO_W, MONO_FOV_H, tilt,
                              flip_h, False, LENS_FOV, LENS_CENTRE)

    def _edge_row(self, tilt: float, bearing: float, source_row: float) -> float:
        """Brute force, through the forward model only: the output row at `bearing` where the
        source row `source_row` (0 = the sensor's top edge) is crossed."""
        focal, cx, cy = source_lens((MONO_W, MONO_H), MONO_W, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        win = window(tilt, lens_fov=LENS_FOV, lens_centre=LENS_CENTRE)
        lo, hi = -80.0, 85.0                                     # elevation, degrees
        for _ in range(80):
            mid = (lo + hi) / 2.0
            _, sy = equidistant_pixel(bearing, mid, tilt, focal, cx, cy)
            if sy > source_row:
                lo = mid
            else:
                hi = mid
        return win.row(lo)

    def test_the_centre_column_is_covered_top_to_bottom(self) -> None:
        for tilt in (0.0, 16.0):
            with self.subTest(tilt=tilt):
                cov = self._coverage(tilt)
                self.assertEqual(tuple(cov[MONO_W // 2]), (0, MONO_H - 1))
                self.assertEqual(tuple(cov[MONO_W // 2 - 1]), (0, MONO_H - 1))

    def test_tilt_16_on_800_rows_has_no_arch(self) -> None:
        # The tangent reaches 43.7 deg at the top, and the sensor's edge columns reach 44 deg:
        # every column is covered top to bottom. The arch is a taller frame's (step 2) — or, at
        # tilt 0, the bottom's (next test).
        cov = self._coverage(16.0)
        self.assertTrue(np.all(cov[:, 0] == 0))
        self.assertTrue(np.all(cov[:, 1] == MONO_H - 1))

    def test_tilt_0_has_the_arch_at_the_bottom(self) -> None:
        # Pinned at the sensor's bottom reach on the centre column, -39 deg, which the edge
        # columns cannot follow (they reach -33): black at the bottom, growing toward the edges.
        cov = self._coverage(0.0)
        self.assertTrue(np.all(cov[:, 0] == 0))
        self.assertLess(cov[0, 1], MONO_H - 1 - 90)
        self.assertTrue(np.all(np.diff(cov[: MONO_W // 2, 1]) >= 0))     # monotonic toward the centre
        for column in (0, 200, 400):
            with self.subTest(column=column):
                bearing = (column - CX) * DPP
                expected = self._edge_row(0.0, bearing, MONO_H - 1.0)
                self.assertAlmostEqual(float(cov[column, 1]), expected, delta=1.5)

    def test_flip_h_mirrors_the_columns(self) -> None:
        plain = self._coverage(0.0)
        flipped = self._coverage(0.0, flip_h=True)
        np.testing.assert_array_equal(flipped, plain[::-1])

    def test_no_picture_is_minus_one(self) -> None:
        # Aim the camera so far up that the bottom rows see nothing on the edge columns.
        cov = frame_coverage((MONO_W, MONO_H), (MONO_W, MONO_H), MONO_W, 127.0, 60.0)
        self.assertEqual(cov.shape, (MONO_W, 2))
        self.assertTrue(np.all((cov[:, 0] >= -1) & (cov[:, 1] >= -1)))

    def test_the_summary_names_the_rows(self) -> None:
        cov = self._coverage(0.0)
        text = coverage_summary(cov, (MONO_W, MONO_H), MONO_W, MONO_FOV_H)
        self.assertIn('centre 0-799', text)
        self.assertIn(f'edges 0-{int(cov[0, 1])}', text)
        self.assertIn('seams 0-', text)


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

    def test_frame_height_is_the_delivered_height(self) -> None:
        # The warp's output is free in height; the sensor mode is not. 0 keeps the mode's rows.
        self.assertEqual(frame_size(False, CameraResolution.P800, height=1136), (1280, 1136))
        self.assertEqual(frame_size(False, CameraResolution.P720, height=960), (1280, 960))
        self.assertEqual(frame_size(False, CameraResolution.P800, height=0), (1280, 800))
        self.assertEqual(mode_size(False, CameraResolution.P800), (1280, 800))      # never the delivered one
        self.assertEqual(frame_size(False, CameraResolution.P800, square=True, height=1136), (800, 800))

    def test_frame_height_is_aligned_to_the_warp(self) -> None:
        from modules.oak.camera.definitions import _warned_heights
        _warned_heights.discard(1130)
        with self.assertLogs('modules.oak.camera.definitions', level='WARNING') as captured:
            self.assertEqual(aligned_height(1130), 1136)
        self.assertIn('1130', captured.output[0])
        self.assertEqual(aligned_height(1136), 1136)
        self.assertEqual(aligned_height(3), 16)

    def test_full_reach_height_is_what_the_doc_says(self) -> None:
        # The rows the centre column needs to carry every sensor row: 1152 at P800 and tilt 16,
        # 960 at P720 and tilt 15, with the shared lens (its centre 10.5 px low gives the sensor
        # 41.3 deg above the axis and 39.2 below). Both multiples of 16.
        self.assertEqual(full_frame_height((1280, 800), 1280, MONO_FOV_H, 16.0, LENS_FOV, LENS_CENTRE), 1152)
        self.assertEqual(full_frame_height((1280, 720), 1280, MONO_FOV_H, 15.0, LENS_FOV, LENS_CENTRE), 960)
        # And at that height the window's top IS the sensor's reach above the axis.
        win = frame_window((1280, 800), (1280, 1152), 1280, MONO_FOV_H, 16.0, LENS_FOV, LENS_CENTRE)
        focal, _, cy = source_lens((1280, 800), 1280, MONO_FOV_H, LENS_FOV, LENS_CENTRE)
        reach_up = math.degrees(cy / focal)
        self.assertGreaterEqual(win.elevation_top, 16.0 + reach_up - 0.05)
        self.assertLess(win.elevation_top, 16.0 + reach_up + 1.0)         # at most the alignment's spare rows


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

    def test_the_bottom_reach_follows_the_label(self) -> None:
        """The 720-row mode is a centre crop: 40 rows less below the axis, so the bottom of the
        window is 4 deg higher and the feet run out of frame sooner."""
        w800 = frame_window((1280, 800), (1280, 800), 1280, MONO_FOV_H, 0.0)
        w720 = frame_window((1280, 720), (1280, 720), 1280, MONO_FOV_H, 0.0)
        self.assertAlmostEqual(w800.elevation_bottom, -math.degrees(399.5 / F_OUT), places=9)
        self.assertAlmostEqual(w720.elevation_bottom, -math.degrees(359.5 / F_OUT), places=9)
        self.assertAlmostEqual(w800.elevation_bottom - w720.elevation_bottom, -4.0, delta=0.05)

    def test_the_horizontal_field_is_the_same_at_both_mono_labels(self) -> None:
        """P720 is a pure vertical crop, so the column-to-azimuth mapping is untouched."""
        fields = []
        for label in (CameraResolution.P720, CameraResolution.P800):
            width, height = mono_frame_size(label)
            win = frame_window((width, height), (width, height), width, MONO_FOV_H, 0.0)
            fields.append(frame_fov(MONO_FOV_H, width, (width, height), win)[0])
        self.assertAlmostEqual(fields[0], fields[1], places=9)
        self.assertAlmostEqual(fields[0], MONO_FOV_H, places=9)


if __name__ == "__main__":
    unittest.main()
