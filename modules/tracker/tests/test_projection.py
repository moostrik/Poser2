"""The projection is the inverse of the tracker's forward chain — prove it, with no GL.

The stitch shader is a transcription of `projection`, and `projection` is checked against the `Rig`
itself, not against a second copy of the arithmetic. If the tracker's forward chain ever changes,
the round trip breaks here rather than on the wall.
"""

import math
import unittest

from modules.oak import frame_window
from modules.tracker import azimuth_to_camera_x, camera_azimuth, camera_local_to_azimuth, \
    centre_distance, elevation_from_row, focus_distance, row_from_elevation, row_model, wrap180
from modules.tracker.panoramic.rig import Rig
from modules.utils import Rect


# The White Space rig, as measured: four OAK-D Pro W on a 0.36 m ring, 127 degrees each.
NUM_CAMERAS: int = 4
CAM_FOV: float = 127.0
TARGET_FOV: float = 360.0 / NUM_CAMERAS
RING_RADIUS: float = 0.36
FOCUS_RADIUS: float = 2.25
# How far local 0 sits before a camera's own sector starts: half the field's spare.
FIELD_OFFSET: float = (CAM_FOV - TARGET_FOV) / 2.0


def _rig(ring_radius: float = RING_RADIUS) -> Rig:
    rig = Rig(CAM_FOV, TARGET_FOV)
    rig.set_camera_radius(ring_radius)
    return rig


def _window(rows: int, tilt: float, src: tuple[int, int] = (1280, 720)):
    """The frame the camera delivers: `frame_window` with the ideal lens, as the tracker builds it."""
    return frame_window(src, (src[0], rows), src[0], CAM_FOV, tilt)


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


class TestRoundTripAgainstRig(unittest.TestCase):
    """Forward through the tracker, back through the projection, land on the column you started from."""

    def _forward(self, rig: Rig, x: float, cam_id: int, depth_radius: float) -> float:
        """The world azimuth the tracker gives a box centred at column `x`. Uses the `Rig`'s own
        public chain, not a re-implementation.

        The tracker corrects at its derived `parallax_radius`, so the fixture sets the zone to
        the depth under test. That keeps this a round trip rather than a tautology: it proves the
        tracker's forward direction and the inverse are the same triangle, and it breaks the
        moment anyone changes one without the other."""
        rig.set_zone(depth_radius, depth_radius)
        return rig.calc_angle(Rect(x, 0.0, 0.0, 0.0), cam_id)[1]

    def test_every_column_of_every_camera_round_trips(self) -> None:
        rig: Rig = _rig()
        for cam_id in range(NUM_CAMERAS):
            for step in range(21):
                x: float = step / 20.0
                azimuth: float = self._forward(rig, x, cam_id, FOCUS_RADIUS)
                back: float | None = azimuth_to_camera_x(
                    azimuth, cam_id, CAM_FOV, TARGET_FOV, RING_RADIUS, FOCUS_RADIUS
                )
                self.assertIsNotNone(back, f'cam {cam_id} column {x} fell outside its own field')
                assert back is not None
                self.assertAlmostEqual(back, x, places=9, msg=f'cam {cam_id} column {x}')

    def test_round_trips_at_other_depths_too(self) -> None:
        """The projection is exact at whatever depth it is given; R 2.25 is a choice, not a constraint."""
        rig: Rig = _rig()
        for radius in (1.5, 3.5, 6.0):
            for step in range(11):
                x: float = step / 10.0
                azimuth: float = self._forward(rig, x, 2, radius)
                back: float | None = azimuth_to_camera_x(
                    azimuth, 2, CAM_FOV, TARGET_FOV, RING_RADIUS, radius
                )
                assert back is not None
                self.assertAlmostEqual(back, x, places=9, msg=f'R{radius} column {x}')

    def test_round_trips_with_the_correction_off(self) -> None:
        rig: Rig = _rig(ring_radius=0.0)
        for step in range(11):
            x: float = step / 10.0
            azimuth: float = self._forward(rig, x, 1, FOCUS_RADIUS)
            back: float | None = azimuth_to_camera_x(
                azimuth, 1, CAM_FOV, TARGET_FOV, 0.0, FOCUS_RADIUS
            )
            assert back is not None
            self.assertAlmostEqual(back, x, places=12)

    def _seam_ghost(self, radius: float) -> float:
        """How far (degrees of azimuth) the R 2.25 projection misplaces a seam person who is really
        at `radius`. This is the ghost the stitch shows."""
        rig: Rig = _rig()
        true_column: float | None = azimuth_to_camera_x(
            90.0, 0, CAM_FOV, TARGET_FOV, RING_RADIUS, radius
        )
        assert true_column is not None
        drawn: float = self._forward(rig, true_column, 0, FOCUS_RADIUS)
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

    def test_the_two_angles_carry_the_whole_row_model(self) -> None:
        """Why the tracker publishes only `angle_bottom`/`angle_top`: `row_model` rebuilds
        (horizon_row, focal_rows) from them exactly, for any sensor mode, tilt and frame height —
        horizon inside the frame or below it. If this ever fails, the dropped fields lost something."""
        for src in ((1280, 720), (1280, 800)):
            for tilt in (0.0, 12.0, 15.0, 16.0, 30.0):
                for rows in (src[1], 960, 1152):
                    with self.subTest(src=src, tilt=tilt, rows=rows):
                        w = _window(rows, tilt, src)
                        horizon_row, focal_rows = row_model(w.elevation_bottom, w.elevation_top)
                        self.assertAlmostEqual(horizon_row, w.horizon_px / (rows - 1), places=12)
                        self.assertAlmostEqual(focal_rows, w.focal / (rows - 1), places=12)
                        self.assertAlmostEqual(row_from_elevation(w.elevation_top, horizon_row, focal_rows),
                                               0.0, places=12)
                        self.assertAlmostEqual(row_from_elevation(w.elevation_bottom, horizon_row, focal_rows),
                                               1.0, places=12)

    def test_a_frame_with_no_span_does_not_divide_by_zero(self) -> None:
        horizon_row, focal_rows = row_model(10.0, 10.0)
        self.assertGreater(focal_rows, 0.0)
        self.assertTrue(math.isfinite(horizon_row))


class TestNoRingIsLinear(unittest.TestCase):
    def test_projection_collapses_to_the_offset(self) -> None:
        for cam_id in range(NUM_CAMERAS):
            for azimuth in (cam_id * TARGET_FOV + d for d in (0.0, 30.0, 89.0)):
                expected: float = (azimuth - TARGET_FOV * cam_id + FIELD_OFFSET) / CAM_FOV
                back: float | None = azimuth_to_camera_x(
                    azimuth, cam_id, CAM_FOV, TARGET_FOV, 0.0, FOCUS_RADIUS
                )
                assert back is not None
                self.assertAlmostEqual(back, expected, places=12)


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
        # With the cameras at the centre there is no triangle left: the projection collapses to
        # `target_fov * cam_id + local - (cam_fov - target_fov) / 2`, which is how the azimuth frame
        # is defined (CALIBRATION.md, *One frame: azimuth*).
        for cam_id in range(NUM_CAMERAS):
            for local in (0.0, 18.5, 63.5, 108.5, CAM_FOV):
                self.assertAlmostEqual(
                    camera_local_to_azimuth(local, cam_id, CAM_FOV, TARGET_FOV, 0.0,
                                            FOCUS_RADIUS),
                    (TARGET_FOV * cam_id + local - FIELD_OFFSET) % 360.0, places=9)

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


class TestCameraAzimuth(unittest.TestCase):
    def test_axes_sit_between_the_seams(self) -> None:
        self.assertEqual(
            [camera_azimuth(i, TARGET_FOV) for i in range(NUM_CAMERAS)], [45.0, 135.0, 225.0, 315.0]
        )

    def test_agrees_with_the_forward_model_at_frame_centre(self) -> None:
        # The frame centre is the camera's axis at any ring and any depth — the one bearing the
        # parallax triangle leaves alone.
        rig: Rig = _rig()
        for cam_id in range(NUM_CAMERAS):
            _local, centre, _d = rig.calc_angle(Rect(0.5, 0.0, 0.0, 0.0), cam_id)
            self.assertAlmostEqual(centre, camera_azimuth(cam_id, TARGET_FOV), places=9)

    def test_three_and_six_cameras(self) -> None:
        self.assertAlmostEqual(camera_azimuth(0, 120.0), 60.0, places=12)
        self.assertAlmostEqual(camera_azimuth(5, 60.0), 330.0, places=12)


class TestWrap180(unittest.TestCase):
    def test_folds_into_range(self) -> None:
        for angle, expected in ((0.0, 0.0), (180.0, -180.0), (181.0, -179.0),
                                (359.0, -1.0), (-1.0, -1.0), (720.0 + 45.0, 45.0)):
            self.assertAlmostEqual(wrap180(angle), expected, places=12)


if __name__ == '__main__':
    unittest.main()
