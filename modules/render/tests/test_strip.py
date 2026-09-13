"""The strip's own geometry — its rows, extent, vertical re-projection, coverage and 0/360 join.

Pure arithmetic with no GL, transcribed into `panoramicstitch.frag`, so these are what that shader
is checked by. The projection it builds on is tested against the tracker in `test_projection`.
"""

import math
import unittest

from modules.oak import frame_window
from modules.render.layers.panorama.strip import camera_elevation, centre_elevation, elevation_window, \
    panorama_coverage, strip_aspect_ratio, strip_elevation, strip_spans, strip_y
from modules.tracker import azimuth_to_camera_x, camera_azimuth, centre_distance, focus_distance, \
    row_from_elevation


# The White Space rig, as measured: four OAK-D Pro W 0.36 m out from the centre, 127 degrees each.
NUM_CAMERAS: int = 4
CAM_FOV: float = 127.0
TARGET_FOV: float = 360.0 / NUM_CAMERAS
CAMERA_RADIUS: float = 0.36
FOCUS_RADIUS: float = 2.25
# How far local 0 sits before a camera's own sector starts: half the field's spare.
FIELD_OFFSET: float = (CAM_FOV - TARGET_FOV) / 2.0


def _window(rows: int, tilt: float, src: tuple[int, int] = (1280, 720)):
    """The frame the camera delivers: `frame_window` with the ideal lens, as the tracker builds it."""
    return frame_window(src, (src[0], rows), src[0], CAM_FOV, tilt)


def _band(rows: int, tilt: float, src: tuple[int, int] = (1280, 720)) -> tuple[float, float]:
    """(bottom, top) of that frame at the camera — the tracker's published `angle_bottom`/`angle_top`."""
    w = _window(rows, tilt, src)
    return (w.elevation_bottom, w.elevation_top)


class TestCameraElevation(unittest.TestCase):
    """Checked against a point placed in 3-D and projected from the camera directly.

    Rig centre at the origin, camera `c` at `camera_radius` out along its own axis. A point on the
    focus cylinder at azimuth `a` and height `h` above the lens plane is at
    `(R cos a, R sin a, h)`; the centre sees it at elevation `atan(h/R)` and the camera at
    `atan(h/d)` with `d` the horizontal distance between them. No formula from the module is
    reused to build the expectation.
    """

    def _direct(self, azimuth: float, height: float, cam_id: int,
                focus_radius: float) -> tuple[float, float]:
        centre: float = math.radians(camera_azimuth(cam_id, TARGET_FOV))
        a: float = math.radians(azimuth)
        dx: float = focus_radius * math.cos(a) - CAMERA_RADIUS * math.cos(centre)
        dy: float = focus_radius * math.sin(a) - CAMERA_RADIUS * math.sin(centre)
        horizontal: float = math.hypot(dx, dy)
        return (math.degrees(math.atan(height / focus_radius)),      # seen from the centre
                math.degrees(math.atan(height / horizontal)))        # seen from the camera

    def test_matches_a_point_projected_in_three_dimensions(self) -> None:
        focus_radius: float = FOCUS_RADIUS
        for cam_id in range(NUM_CAMERAS):
            axis: float = camera_azimuth(cam_id, TARGET_FOV)
            for offset in (-50.0, -25.0, 0.0, 25.0, 50.0):
                for height in (-0.5, 0.4, 1.4):
                    from_centre, from_camera = self._direct(
                        axis + offset, height, cam_id, focus_radius)
                    self.assertAlmostEqual(
                        camera_elevation(from_centre, offset, CAMERA_RADIUS, focus_radius),
                        from_camera, places=9,
                        msg=f'cam {cam_id} offset {offset} height {height}')

    def test_the_horizon_never_moves(self) -> None:
        for offset in (-60.0, 0.0, 60.0):
            self.assertAlmostEqual(
                camera_elevation(0.0, offset, CAMERA_RADIUS, FOCUS_RADIUS), 0.0, places=12)

    def test_compression_matches_the_horizontal_on_the_axis(self) -> None:
        """The point of doing both axes: on the camera's own axis the two factors are the same
        `R / (R - r)`, so a person keeps their proportions."""
        focus_radius: float = FOCUS_RADIUS
        expected: float = focus_radius / (focus_radius - CAMERA_RADIUS)
        small: float = 1e-4
        vertical: float = camera_elevation(small, 0.0, CAMERA_RADIUS, focus_radius) / small
        horizontal: float = (
            azimuth_to_camera_x(camera_azimuth(0, TARGET_FOV) + small, 0, CAM_FOV, TARGET_FOV,
                                CAMERA_RADIUS, FOCUS_RADIUS)      # type: ignore[operator]
            - azimuth_to_camera_x(camera_azimuth(0, TARGET_FOV), 0, CAM_FOV, TARGET_FOV,
                                  CAMERA_RADIUS, FOCUS_RADIUS)    # type: ignore[operator]
        ) * CAM_FOV / small
        self.assertAlmostEqual(vertical, expected, places=6)
        self.assertAlmostEqual(horizontal, expected, places=4)

    def test_zero_camera_radius_is_the_identity(self) -> None:
        for elevation in (-30.0, 0.0, 12.5, 39.0):
            self.assertAlmostEqual(
                camera_elevation(elevation, 40.0, 0.0, FOCUS_RADIUS), elevation, places=12)


class TestCentreElevation(unittest.TestCase):
    """The vertical re-projection for a point whose distance is known, not assumed."""

    def test_round_trips_against_camera_elevation(self) -> None:
        """On the focus cylinder the two functions are inverses: `camera_elevation` takes the
        centre's view to the camera's, and this takes it back, given that point's real distances."""
        focus_radius: float = FOCUS_RADIUS
        for bearing in (-60.0, -25.0, 0.0, 25.0, 60.0):
            cam_distance: float = focus_distance(bearing, CAMERA_RADIUS, focus_radius)
            for elevation in (-20.0, -5.0, 8.0, 31.0):
                at_camera: float = camera_elevation(elevation, bearing, CAMERA_RADIUS, focus_radius)
                self.assertAlmostEqual(
                    centre_elevation(at_camera, cam_distance, focus_radius), elevation, places=9,
                    msg=f'bearing {bearing} elevation {elevation}')

    def test_matches_a_point_projected_in_three_dimensions(self) -> None:
        """Camera at the origin facing +x with the centre `camera_radius` behind it, a person `d` out
        at camera bearing `theta`, standing `h` above the lens plane. The expectation is built from
        the two horizontal distances directly, reusing no formula from the module."""
        for theta in (-50.0, 0.0, 35.0):
            for d in (1.5, 3.0, 6.0):
                for h in (-0.5, 0.4, 1.4):
                    t: float = math.radians(theta)
                    px: float = d * math.cos(t)
                    py: float = d * math.sin(t)
                    from_camera: float = math.degrees(math.atan(h / math.hypot(px, py)))
                    expected: float = math.degrees(
                        math.atan(h / math.hypot(px + CAMERA_RADIUS, py)))
                    self.assertAlmostEqual(
                        centre_elevation(from_camera, d,
                                         centre_distance(theta, d, CAMERA_RADIUS)),
                        expected, places=9, msg=f'theta {theta} d {d} h {h}')

    def test_the_horizon_never_moves(self) -> None:
        for d in (1.2, 4.0):
            self.assertAlmostEqual(
                centre_elevation(0.0, d, centre_distance(30.0, d, CAMERA_RADIUS)), 0.0, places=12)

    def test_zero_camera_radius_is_the_identity(self) -> None:
        d: float = 2.5
        for elevation in (-30.0, 0.0, 12.5, 39.0):
            self.assertAlmostEqual(
                centre_elevation(elevation, d, centre_distance(40.0, d, 0.0)), elevation,
                places=12)


class TestElevationWindow(unittest.TestCase):
    """The strip's vertical extent: what the frames carry, converted to the centre's view."""

    def test_the_band_is_the_frames_window(self) -> None:
        """P720 aimed up 12 on 720 rows: the bottom row is the sensor's lowest reach on the centre
        column, 12 - 35.7 = -23.7, and the rows run up from there as tangents to 38.9."""
        low, high = _band(720, 12.0)
        self.assertAlmostEqual(low, -23.66, delta=0.01)
        self.assertAlmostEqual(high, 38.9, delta=0.05)

    def test_no_tilt_pins_the_bottom_at_the_sensor_reach(self) -> None:
        low, high = _band(800, 0.0, (1280, 800))
        self.assertAlmostEqual(low, -39.64, delta=0.01)
        self.assertLess(high, 39.64)                     # tangent rows do not reach the sensor's top
        self.assertGreater(high, 29.0)

    def test_window_is_the_row_the_strip_draws(self) -> None:
        """P720 at tilt 12 on the R 2.25 cylinder, seen from the centre."""
        top, bottom = elevation_window(_band(720, 12.0), CAMERA_RADIUS, FOCUS_RADIUS)
        self.assertAlmostEqual(top, 34.1, delta=0.15)
        self.assertAlmostEqual(bottom, -20.2, delta=0.15)

    def test_the_window_is_narrower_than_the_band(self) -> None:
        """The centre is farther from the cylinder than a camera is, so it sees the same content
        over a smaller angle. Both ends must move inward, never outward."""
        band: tuple[float, float] = _band(720, 12.0)
        top, bottom = elevation_window(band, CAMERA_RADIUS, FOCUS_RADIUS)
        self.assertLess(top, band[1])
        self.assertGreater(bottom, band[0])

    def test_zero_camera_radius_leaves_the_band_alone(self) -> None:
        band: tuple[float, float] = _band(1152, 16.0, (1280, 800))
        top, bottom = elevation_window(band, 0.0, FOCUS_RADIUS)
        self.assertAlmostEqual(top, band[1], places=12)
        self.assertAlmostEqual(bottom, band[0], places=12)


class TestStripRows(unittest.TestCase):
    """The strip's rows are tangents of centre elevation, the same shape as the frames' rows."""

    WINDOW = (45.0, -18.0)          # (top, bottom) at the rig centre, as `elevation_window` gives it

    def test_round_trip(self) -> None:
        for e in (-18.0, -10.0, 0.0, 12.5, 30.0, 45.0):
            self.assertAlmostEqual(strip_elevation(strip_y(e, self.WINDOW), self.WINDOW), e, places=9)

    def test_the_window_ends_are_the_strip_ends(self) -> None:
        top, bottom = self.WINDOW
        self.assertAlmostEqual(strip_y(top, self.WINDOW), 0.0, places=12)
        self.assertAlmostEqual(strip_y(bottom, self.WINDOW), 1.0, places=12)

    def test_rows_are_tangents(self) -> None:
        # Equal steps in tangent are equal steps in y; equal steps in degrees are not.
        top, bottom = self.WINDOW
        span = math.tan(math.radians(top)) - math.tan(math.radians(bottom))
        for e in (-10.0, 0.0, 20.0, 40.0):
            self.assertAlmostEqual(strip_y(e, self.WINDOW),
                                   (math.tan(math.radians(top)) - math.tan(math.radians(e))) / span, places=12)
        ten_low = strip_y(0.0, self.WINDOW) - strip_y(10.0, self.WINDOW)
        ten_high = strip_y(30.0, self.WINDOW) - strip_y(40.0, self.WINDOW)
        # tan(40) - tan(30) is 1.49 x tan(10) - tan(0): the top spends more rows per degree.
        self.assertAlmostEqual(ten_high / ten_low, 1.49, delta=0.02)

    def test_same_shape_as_the_frame_rows(self) -> None:
        # A frame window and a strip window with the same ends map an elevation to the same
        # normalised row: the two row mappings are one mapping.
        top, bottom = self.WINDOW
        rows = 1000
        focal = (rows - 1) / (math.tan(math.radians(top)) - math.tan(math.radians(bottom)))
        horizon_row = focal * math.tan(math.radians(top)) / (rows - 1)
        focal_rows = focal / (rows - 1)
        for e in (-15.0, 0.0, 25.0, 44.0):
            self.assertAlmostEqual(strip_y(e, self.WINDOW), row_from_elevation(e, horizon_row, focal_rows), places=9)

    def test_aspect_is_a_full_turn_over_the_tangent_span(self) -> None:
        top, bottom = self.WINDOW
        span = math.tan(math.radians(top)) - math.tan(math.radians(bottom))
        self.assertAlmostEqual(strip_aspect_ratio(self.WINDOW), 2.0 * math.pi / span, places=12)
        # Taller than the old linear strip would have been, since the tangent spends rows at the top.
        self.assertLess(strip_aspect_ratio(self.WINDOW), 360.0 / (top - bottom))

    def test_degenerate_window_does_not_divide_by_zero(self) -> None:
        self.assertEqual(strip_y(3.0, (5.0, 5.0)), 0.5)
        self.assertGreater(strip_aspect_ratio((5.0, 5.0)), 1.0)


class TestStripSpans(unittest.TestCase):
    """The 0/360 join, in one place: a band that runs off one edge comes back on the other, and
    a silently clipped band would be invisible exactly where two cameras meet."""

    def test_a_band_inside_the_strip_is_one_span(self) -> None:
        self.assertEqual(strip_spans(0.25, 0.5), [(0.25, 0.5)])

    def test_a_band_that_wraps_is_two_spans_that_sum(self) -> None:
        spans = strip_spans(0.9, 0.2)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(sum(w for _x, w in spans), 0.2, places=12)
        self.assertAlmostEqual(spans[0][0], 0.9, places=12)
        self.assertAlmostEqual(spans[1][0], 0.0, places=12)

    def test_a_negative_left_edge_wraps_in(self) -> None:
        # A bar centred just past azimuth 0 starts at a negative x, which is the common case.
        spans = strip_spans(-0.05, 0.1)
        self.assertAlmostEqual(sum(w for _x, w in spans), 0.1, places=12)
        self.assertAlmostEqual(spans[0][0], 0.95, places=12)

    def test_a_band_ending_exactly_on_the_edge_stays_one_span(self) -> None:
        self.assertEqual(strip_spans(0.8, 0.2), [(0.8, 0.2)])

    def test_nothing_and_everything(self) -> None:
        self.assertEqual(strip_spans(0.3, 0.0), [])
        self.assertEqual(strip_spans(0.3, -0.1), [])
        self.assertEqual(strip_spans(0.3, 1.0), [(0.0, 1.0)])
        self.assertEqual(strip_spans(0.3, 2.0), [(0.0, 1.0)])

    def test_total_width_is_preserved_wherever_it_starts(self) -> None:
        for start in (0.0, 0.15, 0.5, 0.97, 1.0, 1.4, -1.2):
            with self.subTest(start=start):
                self.assertAlmostEqual(sum(w for _x, w in strip_spans(start, 0.3)), 0.3,
                                       places=12)


class TestCoverage(unittest.TestCase):
    def _coverage(self, azimuth: float, camera_radius: float = CAMERA_RADIUS) -> int:
        return panorama_coverage(azimuth, NUM_CAMERAS, CAM_FOV, TARGET_FOV,
                                 camera_radius, FOCUS_RADIUS)

    def test_two_on_the_seams_one_on_the_axes_without_parallax(self) -> None:
        for seam in (0.0, 90.0, 180.0, 270.0):
            self.assertEqual(self._coverage(seam, 0.0), 2, f'seam {seam}')
            self.assertEqual(self._coverage(seam + FIELD_OFFSET - 0.5, 0.0), 2)
            self.assertEqual(self._coverage(seam + FIELD_OFFSET + 0.5, 0.0), 1)
        for axis in (45.0, 135.0, 225.0, 315.0):
            self.assertEqual(self._coverage(axis, 0.0), 1, f'axis {axis}')

    def test_parallax_narrows_the_overlap_but_keeps_its_shape(self) -> None:
        """A camera pushed 0.36 m outward covers less of the cylinder, measured from the centre —
        so the band where two cameras see the same place is narrower than the bare field says."""
        self.assertEqual(self._coverage(90.0), 2)
        self.assertEqual(self._coverage(45.0), 1)
        self.assertEqual(self._coverage(90.0 + FIELD_OFFSET - 0.5), 1,
                         'the raw overlap band is not all doubly covered once parallax is on')

    def test_never_a_gap(self) -> None:
        for step in range(360):
            self.assertGreaterEqual(self._coverage(float(step)), 1, f'azimuth {step} uncovered')

    def test_wrap_is_symmetric_around_zero(self) -> None:
        self.assertEqual(self._coverage(359.0), self._coverage(1.0))


if __name__ == '__main__':
    unittest.main()
