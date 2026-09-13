"""The panorama map is the inverse of the tracker's forward geometry — prove it, with no GL.

The point of these tests is that the stitch shader is a transcription of `panorama_map`, and
`panorama_map` is checked against `Geometry` itself, not against a second copy of the arithmetic.
If the tracker's forward chain ever changes, the round trip breaks here rather than on the wall.
"""

import math
import unittest

from modules.oak import frame_window
from modules.tracker import azimuth_to_camera_x, camera_azimuth, camera_elevation, \
    camera_local_to_azimuth, centre_distance, centre_elevation, elevation_window, \
    focus_distance, fov_overlap, panorama_coverage, populated_band, row_from_elevation, \
    elevation_from_row, strip_spans, strip_y, strip_elevation, strip_aspect_ratio, wrap180
from modules.tracker.panoramic.geometry import Geometry
from modules.utils import Rect


# The White Space rig, as measured: four OAK-D Pro W on a 0.36 m ring, 127 degrees each.
NUM_CAMERAS: int = 4
CAM_FOV: float = 127.0
TARGET_FOV: float = 360.0 / NUM_CAMERAS
RING_RADIUS: float = 0.36
FOCUS_RADIUS: float = 2.25


def _geometry(ring_radius: float = RING_RADIUS) -> Geometry:
    geometry = Geometry(NUM_CAMERAS, CAM_FOV, TARGET_FOV)
    geometry.set_camera_radius(ring_radius)
    return geometry


def _cylinder_distance(theta_degrees: float, ring_radius: float, focus_radius: float) -> float:
    """Camera distance to the focus cylinder, from the camera's **own** bearing.

    `focus_distance` answers the same question from the centre's bearing, which is what the
    panorama has. This is the other parameterisation, needed to drive the forward model: the
    point is at K + d*(cos t, sin t) with |K| = r, and it lies on the cylinder when
    d^2 + 2*r*d*cos(t) + r^2 = R^2.
    """
    t: float = math.radians(theta_degrees)
    return -ring_radius * math.cos(t) + math.sqrt(
        focus_radius * focus_radius - (ring_radius * math.sin(t)) ** 2
    )


class TestFocusDistance(unittest.TestCase):
    def test_seam_person_is_about_two_metres_out(self) -> None:
        """A person on a seam at play-zone middle: 45 degrees off axis on a 2.25 m cylinder."""
        self.assertAlmostEqual(
            focus_distance(45.0, RING_RADIUS, FOCUS_RADIUS), 2.01, places=2
        )

    def test_symmetric_about_the_camera_axis(self) -> None:
        for bearing in (5.0, 20.0, 45.0, 63.5):
            self.assertAlmostEqual(
                focus_distance(bearing, RING_RADIUS, FOCUS_RADIUS),
                focus_distance(-bearing, RING_RADIUS, FOCUS_RADIUS),
                places=12,
            )

    def test_closest_straight_ahead_farthest_behind(self) -> None:
        """The camera is pushed toward the wall it faces, so its own axis is the *short* ray: the
        cylinder is `focus_radius - ring_radius` dead ahead and `+ ring_radius` behind."""
        ahead: float = focus_distance(0.0, RING_RADIUS, FOCUS_RADIUS)
        side: float = focus_distance(63.5, RING_RADIUS, FOCUS_RADIUS)
        behind: float = focus_distance(180.0, RING_RADIUS, FOCUS_RADIUS)
        self.assertAlmostEqual(ahead, FOCUS_RADIUS - RING_RADIUS, places=12)
        self.assertAlmostEqual(behind, FOCUS_RADIUS + RING_RADIUS, places=12)
        self.assertLess(ahead, side)
        self.assertLess(side, behind)

    def test_no_ring_means_one_distance_everywhere(self) -> None:
        for bearing in (0.0, 30.0, 63.5):
            self.assertAlmostEqual(
                focus_distance(bearing, 0.0, FOCUS_RADIUS), FOCUS_RADIUS, places=12
            )


class TestRoundTripAgainstGeometry(unittest.TestCase):
    """Forward through the tracker, back through the map, land on the column you started from."""

    def _forward(self, geometry: Geometry, x: float, cam_id: int, depth_radius: float) -> float:
        """The world azimuth the tracker gives a box centred at column `x`. Uses `Geometry`'s own
        public chain, not a re-implementation.

        The tracker corrects at its derived `parallax_radius`, so the fixture sets the zone to
        the depth under test. That keeps this a round trip rather than a tautology: it proves the
        tracker's forward direction and the map's inverse are the same triangle, and it breaks the
        moment anyone changes one without the other."""
        geometry.set_zone(depth_radius, depth_radius)
        return geometry.calc_angle(Rect(x, 0.0, 0.0, 0.0), cam_id)[1]

    def test_every_column_of_every_camera_round_trips(self) -> None:
        geometry: Geometry = _geometry()
        for cam_id in range(NUM_CAMERAS):
            for step in range(21):
                x: float = step / 20.0
                azimuth: float = self._forward(geometry, x, cam_id, FOCUS_RADIUS)
                back: float | None = azimuth_to_camera_x(
                    azimuth, cam_id, CAM_FOV, TARGET_FOV, RING_RADIUS, FOCUS_RADIUS
                )
                self.assertIsNotNone(back, f'cam {cam_id} column {x} fell outside its own field')
                assert back is not None
                self.assertAlmostEqual(back, x, places=9, msg=f'cam {cam_id} column {x}')

    def test_round_trips_at_other_depths_too(self) -> None:
        """The map is exact at whatever depth it is given; R 2.25 is a choice, not a constraint."""
        geometry: Geometry = _geometry()
        for radius in (1.5, 3.5, 6.0):
            for step in range(11):
                x: float = step / 10.0
                azimuth: float = self._forward(geometry, x, 2, radius)
                back: float | None = azimuth_to_camera_x(
                    azimuth, 2, CAM_FOV, TARGET_FOV, RING_RADIUS, radius
                )
                assert back is not None
                self.assertAlmostEqual(back, x, places=9, msg=f'R{radius} column {x}')

    def test_round_trips_with_the_correction_off(self) -> None:
        geometry: Geometry = _geometry(ring_radius=0.0)
        for step in range(11):
            x: float = step / 10.0
            azimuth: float = self._forward(geometry, x, 1, FOCUS_RADIUS)
            back: float | None = azimuth_to_camera_x(
                azimuth, 1, CAM_FOV, TARGET_FOV, 0.0, FOCUS_RADIUS
            )
            assert back is not None
            self.assertAlmostEqual(back, x, places=12)

    def _seam_ghost(self, radius: float) -> float:
        """How far (degrees of azimuth) the R 2.25 map misplaces a seam person who is really at
        `radius`. This is the ghost the stitch shows, and the whole content of step 4.3."""
        geometry: Geometry = _geometry()
        true_column: float | None = azimuth_to_camera_x(
            90.0, 0, CAM_FOV, TARGET_FOV, RING_RADIUS, radius
        )
        assert true_column is not None
        drawn: float = self._forward(geometry, true_column, 0, FOCUS_RADIUS)
        return drawn - 90.0

    def test_wrong_depth_is_wrong_by_a_bounded_amount(self) -> None:
        """The play zone is R 1.5 – R 3.5 and the image is aligned for its middle, so the ends ghost.
        Bounded, and in opposite directions — which is what makes the ghost readable rather than
        just wrong: nearer than the focus depth leans one way, farther the other."""
        near: float = self._seam_ghost(1.5)
        far: float = self._seam_ghost(3.5)
        self.assertGreater(near, 0.0)
        self.assertLess(far, 0.0)
        self.assertLess(abs(near), 4.0, f'R1.5 ghosts by {near:.2f} deg')
        self.assertLess(abs(far), 3.0, f'R3.5 ghosts by {far:.2f} deg')

    def test_no_ghost_at_the_focus_depth(self) -> None:
        self.assertAlmostEqual(self._seam_ghost(FOCUS_RADIUS), 0.0, places=9)


class TestCameraElevation(unittest.TestCase):
    """Checked against a point placed in 3-D and projected from the camera directly.

    Rig centre at the origin, camera `c` at `ring_radius` out along its own axis. A point on the
    focus cylinder at azimuth `a` and height `h` above the lens plane is at
    `(R cos a, R sin a, h)`; the centre sees it at elevation `atan(h/R)` and the camera at
    `atan(h/d)` with `d` the horizontal distance between them. No formula from the module is
    reused to build the expectation.
    """

    def _direct(self, azimuth: float, height: float, cam_id: int,
                focus_radius: float) -> tuple[float, float]:
        centre: float = math.radians(camera_azimuth(cam_id, TARGET_FOV))
        a: float = math.radians(azimuth)
        dx: float = focus_radius * math.cos(a) - RING_RADIUS * math.cos(centre)
        dy: float = focus_radius * math.sin(a) - RING_RADIUS * math.sin(centre)
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
                        camera_elevation(from_centre, offset, RING_RADIUS, focus_radius),
                        from_camera, places=9,
                        msg=f'cam {cam_id} offset {offset} height {height}')

    def test_the_horizon_never_moves(self) -> None:
        for offset in (-60.0, 0.0, 60.0):
            self.assertAlmostEqual(
                camera_elevation(0.0, offset, RING_RADIUS, FOCUS_RADIUS), 0.0, places=12)

    def test_compression_matches_the_horizontal_on_the_axis(self) -> None:
        """The point of doing both axes: on the camera's own axis the two factors are the same
        `R / (R - r)`, so a person keeps their proportions."""
        focus_radius: float = FOCUS_RADIUS
        expected: float = focus_radius / (focus_radius - RING_RADIUS)
        small: float = 1e-4
        vertical: float = camera_elevation(small, 0.0, RING_RADIUS, focus_radius) / small
        horizontal: float = (
            azimuth_to_camera_x(camera_azimuth(0, TARGET_FOV) + small, 0, CAM_FOV, TARGET_FOV,
                                RING_RADIUS, FOCUS_RADIUS)      # type: ignore[operator]
            - azimuth_to_camera_x(camera_azimuth(0, TARGET_FOV), 0, CAM_FOV, TARGET_FOV,
                                  RING_RADIUS, FOCUS_RADIUS)    # type: ignore[operator]
        ) * CAM_FOV / small
        self.assertAlmostEqual(vertical, expected, places=6)
        self.assertAlmostEqual(horizontal, expected, places=4)

    def test_no_ring_is_the_identity(self) -> None:
        for elevation in (-30.0, 0.0, 12.5, 39.0):
            self.assertAlmostEqual(
                camera_elevation(elevation, 40.0, 0.0, FOCUS_RADIUS), elevation, places=12)


class TestCentreDistance(unittest.TestCase):
    """A person's distance from the rig centre, given their distance from a camera."""

    def test_straight_ahead_adds_the_ring(self) -> None:
        self.assertAlmostEqual(centre_distance(0.0, 3.0, RING_RADIUS), 3.0 + RING_RADIUS, places=12)

    def test_symmetric_about_the_axis(self) -> None:
        for bearing in (5.0, 30.0, 63.5):
            self.assertAlmostEqual(centre_distance(bearing, 2.0, RING_RADIUS),
                                   centre_distance(-bearing, 2.0, RING_RADIUS), places=12)

    def test_no_ring_changes_nothing(self) -> None:
        for bearing in (0.0, 45.0, 90.0):
            self.assertAlmostEqual(centre_distance(bearing, 2.0, 0.0), 2.0, places=12)

    def test_inverts_focus_distance_on_the_cylinder(self) -> None:
        """`focus_distance` goes centre-bearing -> camera-distance; this goes camera-bearing ->
        centre-distance. Feed one the other's answer and the focus radius must come back."""
        focus_radius: float = FOCUS_RADIUS
        for centre_bearing in (0.0, 20.0, 55.0):
            d: float = focus_distance(centre_bearing, RING_RADIUS, focus_radius)
            phi: float = math.radians(centre_bearing)
            theta: float = phi + math.asin(
                max(-1.0, min(1.0, RING_RADIUS * math.sin(phi) / d)))
            self.assertAlmostEqual(centre_distance(math.degrees(theta), d, RING_RADIUS),
                                   focus_radius, places=9, msg=f'bearing {centre_bearing}')


class TestCentreElevation(unittest.TestCase):
    """The vertical re-projection for a point whose distance is known, not assumed."""

    def test_round_trips_against_camera_elevation(self) -> None:
        """On the focus cylinder the two functions are inverses: `camera_elevation` takes the
        centre's view to the camera's, and this takes it back, given that point's real distances."""
        focus_radius: float = FOCUS_RADIUS
        for bearing in (-60.0, -25.0, 0.0, 25.0, 60.0):
            cam_distance: float = focus_distance(bearing, RING_RADIUS, focus_radius)
            for elevation in (-20.0, -5.0, 8.0, 31.0):
                at_camera: float = camera_elevation(elevation, bearing, RING_RADIUS, focus_radius)
                self.assertAlmostEqual(
                    centre_elevation(at_camera, cam_distance, focus_radius), elevation, places=9,
                    msg=f'bearing {bearing} elevation {elevation}')

    def test_matches_a_point_projected_in_three_dimensions(self) -> None:
        """Camera at the origin facing +x with the centre `ring_radius` behind it, a person `d` out
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
                        math.atan(h / math.hypot(px + RING_RADIUS, py)))
                    self.assertAlmostEqual(
                        centre_elevation(from_camera, d,
                                         centre_distance(theta, d, RING_RADIUS)),
                        expected, places=9, msg=f'theta {theta} d {d} h {h}')

    def test_the_horizon_never_moves(self) -> None:
        for d in (1.2, 4.0):
            self.assertAlmostEqual(
                centre_elevation(0.0, d, centre_distance(30.0, d, RING_RADIUS)), 0.0, places=12)

    def test_no_ring_is_the_identity(self) -> None:
        d: float = 2.5
        for elevation in (-30.0, 0.0, 12.5, 39.0):
            self.assertAlmostEqual(
                centre_elevation(elevation, d, centre_distance(40.0, d, 0.0)), elevation,
                places=12)


def _window(rows: int, tilt: float, src: tuple[int, int] = (1280, 720)):
    """The frame the camera delivers: `frame_window` with the ideal lens, as the tracker builds it."""
    return frame_window(src, (src[0], rows), src[0], CAM_FOV, tilt)


class TestElevationWindow(unittest.TestCase):
    """The strip's vertical extent: what the frames carry, converted to the centre's view."""

    def test_the_band_is_the_frames_window(self) -> None:
        """P720 aimed up 12 on 720 rows: the bottom row is the sensor's lowest reach on the centre
        column, 12 - 35.7 = -23.7, and the rows run up from there as tangents to 38.9."""
        low, high = populated_band(_window(720, 12.0))
        self.assertAlmostEqual(low, -23.66, delta=0.01)
        self.assertAlmostEqual(high, 38.9, delta=0.05)

    def test_no_tilt_pins_the_bottom_at_the_sensor_reach(self) -> None:
        low, high = populated_band(_window(800, 0.0, (1280, 800)))
        self.assertAlmostEqual(low, -39.64, delta=0.01)
        self.assertLess(high, 39.64)                     # tangent rows do not reach the sensor's top
        self.assertGreater(high, 29.0)

    def test_window_is_the_row_the_strip_draws(self) -> None:
        """P720 at tilt 12 on the R 2.25 cylinder, seen from the centre."""
        top, bottom = elevation_window(populated_band(_window(720, 12.0)), RING_RADIUS,
                                       FOCUS_RADIUS)
        self.assertAlmostEqual(top, 34.1, delta=0.15)
        self.assertAlmostEqual(bottom, -20.2, delta=0.15)

    def test_the_window_is_narrower_than_the_band(self) -> None:
        """The centre is farther from the cylinder than a camera is, so it sees the same content
        over a smaller angle. Both ends must move inward, never outward."""
        band: tuple[float, float] = populated_band(_window(720, 12.0))
        top, bottom = elevation_window(band, RING_RADIUS, FOCUS_RADIUS)
        self.assertLess(top, band[1])
        self.assertGreater(bottom, band[0])

    def test_no_ring_leaves_the_band_alone(self) -> None:
        band: tuple[float, float] = populated_band(_window(1152, 16.0, (1280, 800)))
        top, bottom = elevation_window(band, 0.0, FOCUS_RADIUS)
        self.assertAlmostEqual(top, band[1], places=12)
        self.assertAlmostEqual(bottom, band[0], places=12)


class TestRowModel(unittest.TestCase):
    """The frames' rows are tangents of elevation; the two functions here are what the stitch
    shader transcribes and what the marks invert."""

    def _model(self, rows: int, tilt: float) -> tuple[float, float]:
        w = _window(rows, tilt)
        return w.horizon_px / (rows - 1), w.focal / (rows - 1)

    def test_round_trip(self) -> None:
        horizon_row, focal_rows = self._model(960, 15.0)
        for elevation in (-20.0, -5.0, 0.0, 12.5, 40.0, 52.0):
            row: float = row_from_elevation(elevation, horizon_row, focal_rows)
            self.assertAlmostEqual(elevation_from_row(row, horizon_row, focal_rows), elevation, places=9)

    def test_the_ends_of_the_window_are_the_ends_of_the_frame(self) -> None:
        w = _window(960, 15.0)
        horizon_row, focal_rows = self._model(960, 15.0)
        self.assertAlmostEqual(row_from_elevation(w.elevation_bottom, horizon_row, focal_rows), 1.0, places=9)
        self.assertAlmostEqual(row_from_elevation(w.elevation_top, horizon_row, focal_rows), 0.0, places=9)
        self.assertAlmostEqual(row_from_elevation(0.0, horizon_row, focal_rows), horizon_row, places=12)

    def test_rows_agree_with_the_window(self) -> None:
        w = _window(960, 15.0)
        horizon_row, focal_rows = self._model(960, 15.0)
        for row in (0.0, 0.25, 0.5, 0.778, 1.0):
            self.assertAlmostEqual(elevation_from_row(row, horizon_row, focal_rows),
                                   w.elevation(row * 959.0), places=9)

    def test_no_focal_is_flat(self) -> None:
        self.assertEqual(elevation_from_row(0.3, 0.5, 0.0), 0.0)


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


class TestNoRingIsLinear(unittest.TestCase):
    def test_map_collapses_to_the_offset(self) -> None:
        overlap: float = fov_overlap(CAM_FOV, TARGET_FOV)
        for cam_id in range(NUM_CAMERAS):
            for azimuth in (cam_id * TARGET_FOV + d for d in (0.0, 30.0, 89.0)):
                expected: float = (azimuth - TARGET_FOV * cam_id + overlap) / CAM_FOV
                back: float | None = azimuth_to_camera_x(
                    azimuth, cam_id, CAM_FOV, TARGET_FOV, 0.0, FOCUS_RADIUS
                )
                assert back is not None
                self.assertAlmostEqual(back, expected, places=12)

    def test_overlap_is_eighteen_and_a_half_degrees(self) -> None:
        self.assertAlmostEqual(fov_overlap(CAM_FOV, TARGET_FOV), 18.5, places=12)


class TestCameraLocalToAzimuth(unittest.TestCase):
    """The other direction, which is what the display draws bands and bars with: a camera's own
    local angle to the world azimuth its pixels land on."""

    def test_inverts_azimuth_to_camera_x(self) -> None:
        for cam_id in range(NUM_CAMERAS):
            for step in range(21):
                local: float = CAM_FOV * step / 20.0
                azimuth: float = camera_local_to_azimuth(
                    local, cam_id, CAM_FOV, TARGET_FOV, RING_RADIUS, FOCUS_RADIUS)
                back: float | None = azimuth_to_camera_x(
                    azimuth, cam_id, CAM_FOV, TARGET_FOV, RING_RADIUS, FOCUS_RADIUS)
                self.assertIsNotNone(back, f'cam {cam_id} local {local} left its own field')
                assert back is not None
                self.assertAlmostEqual(back * CAM_FOV, local, places=9,
                                       msg=f'cam {cam_id} local {local}')

    def test_a_camera_spans_less_than_its_field_at_the_focus_depth(self) -> None:
        """The free check in CALIBRATION.md, in one call: 127 degrees of lens covers 110.5
        degrees of the strip at R 2.25, because the camera sits 0.36 m outside the centre."""
        for cam_id in range(NUM_CAMERAS):
            left: float = camera_local_to_azimuth(0.0, cam_id, CAM_FOV, TARGET_FOV,
                                                  RING_RADIUS, FOCUS_RADIUS)
            right: float = camera_local_to_azimuth(CAM_FOV, cam_id, CAM_FOV, TARGET_FOV,
                                                   RING_RADIUS, FOCUS_RADIUS)
            # cam 0's left edge is below azimuth 0, so the span is a wrapped difference — the
            # same modulo the renderer takes before handing the band to `strip_spans`.
            self.assertAlmostEqual((right - left) % 360.0, 110.5, delta=0.1)
            bare: float = (
                camera_local_to_azimuth(CAM_FOV, cam_id, CAM_FOV, TARGET_FOV, 0.0,
                                        FOCUS_RADIUS)
                - camera_local_to_azimuth(0.0, cam_id, CAM_FOV, TARGET_FOV, 0.0, FOCUS_RADIUS)
            ) % 360.0
            self.assertAlmostEqual(bare, CAM_FOV, places=9)  # ring_radius 0: the bare field

    def test_no_ring_is_the_plain_offset(self) -> None:
        # With the cameras at the centre there is no triangle left: the map collapses to
        # `target_fov * cam_id + local - fov_overlap`, which is how the azimuth frame is defined
        # (CALIBRATION.md, *One frame: azimuth*).
        overlap: float = fov_overlap(CAM_FOV, TARGET_FOV)
        for cam_id in range(NUM_CAMERAS):
            for local in (0.0, 18.5, 63.5, 108.5, CAM_FOV):
                self.assertAlmostEqual(
                    camera_local_to_azimuth(local, cam_id, CAM_FOV, TARGET_FOV, 0.0,
                                            FOCUS_RADIUS),
                    (TARGET_FOV * cam_id + local - overlap) % 360.0, places=9)

    def test_the_axis_lands_on_the_axis(self) -> None:
        # Straight ahead the parallax triangle is degenerate, whatever the ring or the depth.
        for radius in (1.5, 2.25, 3.5):
            self.assertAlmostEqual(
                camera_local_to_azimuth(CAM_FOV / 2.0, 2, CAM_FOV, TARGET_FOV,
                                        RING_RADIUS, radius),
                camera_azimuth(2, TARGET_FOV), places=9)

    def test_monotonic_across_the_field(self) -> None:
        previous: float = -1.0
        for step in range(101):
            azimuth: float = camera_local_to_azimuth(CAM_FOV * step / 100.0, 1, CAM_FOV,
                                                     TARGET_FOV, RING_RADIUS, FOCUS_RADIUS)
            self.assertGreater(azimuth, previous)
            previous = azimuth


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


class TestCameraAzimuth(unittest.TestCase):
    def test_axes_sit_between_the_seams(self) -> None:
        self.assertEqual(
            [camera_azimuth(i, TARGET_FOV) for i in range(NUM_CAMERAS)], [45.0, 135.0, 225.0, 315.0]
        )

    def test_agrees_with_the_forward_model_at_frame_centre(self) -> None:
        # The frame centre is the camera's axis at any ring and any depth — the one bearing the
        # parallax triangle leaves alone.
        geometry: Geometry = _geometry()
        for cam_id in range(NUM_CAMERAS):
            _local, centre, _d = geometry.calc_angle(Rect(0.5, 0.0, 0.0, 0.0), cam_id)
            self.assertAlmostEqual(centre, camera_azimuth(cam_id, TARGET_FOV), places=9)

    def test_three_and_six_cameras(self) -> None:
        self.assertAlmostEqual(camera_azimuth(0, 120.0), 60.0, places=12)
        self.assertAlmostEqual(camera_azimuth(5, 60.0), 330.0, places=12)


class TestCoverage(unittest.TestCase):
    def _coverage(self, azimuth: float, ring_radius: float = RING_RADIUS) -> int:
        return panorama_coverage(azimuth, NUM_CAMERAS, CAM_FOV, TARGET_FOV,
                                 ring_radius, FOCUS_RADIUS)

    def test_two_on_the_seams_one_on_the_axes_without_parallax(self) -> None:
        overlap: float = fov_overlap(CAM_FOV, TARGET_FOV)
        for seam in (0.0, 90.0, 180.0, 270.0):
            self.assertEqual(self._coverage(seam, 0.0), 2, f'seam {seam}')
            self.assertEqual(self._coverage(seam + overlap - 0.5, 0.0), 2)
            self.assertEqual(self._coverage(seam + overlap + 0.5, 0.0), 1)
        for axis in (45.0, 135.0, 225.0, 315.0):
            self.assertEqual(self._coverage(axis, 0.0), 1, f'axis {axis}')

    def test_parallax_narrows_the_overlap_but_keeps_its_shape(self) -> None:
        """A camera pushed 0.36 m outward covers less of the cylinder, measured from the centre —
        so the band where two cameras see the same place is narrower than the bare field says."""
        overlap: float = fov_overlap(CAM_FOV, TARGET_FOV)
        self.assertEqual(self._coverage(90.0), 2)
        self.assertEqual(self._coverage(45.0), 1)
        self.assertEqual(self._coverage(90.0 + overlap - 0.5), 1,
                         'the raw overlap band is not all doubly covered once parallax is on')

    def test_never_a_gap(self) -> None:
        for step in range(360):
            self.assertGreaterEqual(self._coverage(float(step)), 1, f'azimuth {step} uncovered')

    def test_wrap_is_symmetric_around_zero(self) -> None:
        self.assertEqual(self._coverage(359.0), self._coverage(1.0))


class TestWrap180(unittest.TestCase):
    def test_folds_into_range(self) -> None:
        for angle, expected in ((0.0, 0.0), (180.0, -180.0), (181.0, -179.0),
                                (359.0, -1.0), (-1.0, -1.0), (720.0 + 45.0, 45.0)):
            self.assertAlmostEqual(wrap180(angle), expected, places=12)


if __name__ == '__main__':
    unittest.main()
