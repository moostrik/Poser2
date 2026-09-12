"""The panorama's marks: where an observation lands on the strip, and how wide its rule is.

`marks.py` is arithmetic with no GL in it, which is why it can be tested at all — the renderers
around it cannot be, without a context. What is worth pinning here is the part a reader of the
display has to be able to trust: **two fields of one colour that overlap are two observations the
tracker will join.** That is a claim about two separate pieces of code agreeing, so it is asserted
against `Tracker`'s own gate rather than against a second copy of the same comparison.
"""

import math
import unittest

from modules.render.layers.panorama.marks import Mark, StripGeometry, build_marks
from modules.tracker import PanoramicAnnotation, PanoramicTracker, PanoramicTrackerSettings, \
    Tracklet, TrackingStatus, row_from_elevation, strip_spans, strip_y
from modules.utils import Rect


# The White Space rig: four cameras, 127 degrees each, lenses 0.36 m out from the centre.
CAM_FOV: float = 127.0
TARGET_FOV: float = 90.0
RING_RADIUS: float = 0.36
PARALLAX_DIAMETER: float = 4.2      # the zone's harmonic mean, what the tracker corrects at
LINK_ANGLE: float = 18.0
REACQUIRE_ANGLE: float = 5.0

GEOMETRY: StripGeometry = StripGeometry(
    cam_fov=CAM_FOV, target_fov=TARGET_FOV, ring_radius=RING_RADIUS,
    parallax_diameter=PARALLAX_DIAMETER,
    row_model=(0.78, 0.58), elevation_window=(40.0, -30.0),
    link_angle=LINK_ANGLE, reacquire_angle=REACQUIRE_ANGLE,
)

WHITE: list[tuple[float, float, float, float]] = [(1.0, 1.0, 1.0, 1.0)]


def observation(cam_id: int, local_angle: float, world_angle: float, *, overlap: bool,
                distance: float = 3.0, top: float = 0.2, height: float = 0.5,
                world_id: int = 0, status: TrackingStatus = TrackingStatus.TRACKED) -> Tracklet:
    """One camera's annotated view of one person, as the tracker would have stored it.

    `local_angle` and `world_angle` are passed independently on purpose: the tracker derives the
    second from the first through the parallax correction, and a mark must use each where it
    belongs rather than re-deriving either.
    """
    roi = Rect(x=local_angle / CAM_FOV - 0.025, y=top, width=0.05, height=height)
    return Tracklet(
        cam_id=cam_id, status=status, roi=roi, external_id=1, external_age_in_frames=10,
        annotation=PanoramicAnnotation(local_angle=local_angle, world_angle=world_angle,
                                       overlap=overlap, distance=distance, height=1.8),
    )


def mark(tracklet: Tracklet, geometry: StripGeometry = GEOMETRY) -> Mark:
    marks: list[Mark] = build_marks([tracklet], set(), WHITE, geometry)
    assert len(marks) == 1
    return marks[0]


def span(m: Mark) -> tuple[float, float]:
    """A mark's tolerance field in degrees, as (low, high) — high may pass 360 when it wraps."""
    return (m.tolerance_x * 360.0, (m.tolerance_x + m.tolerance_w) * 360.0)


class TestPlacement(unittest.TestCase):

    def test_x_is_the_fused_world_angle(self) -> None:
        # Not the local angle, and not the picture: the number the rest of the app receives.
        self.assertAlmostEqual(mark(observation(0, 120.0, 96.0, overlap=True)).x,
                               96.0 / 360.0, places=9)

    def test_rows_are_not_clamped_to_the_strip(self) -> None:
        # The device extrapolates a partly visible person, and that is real information about how
        # close they are. The renderer clips when it draws; a mark must not throw it away first.
        m: Mark = mark(observation(0, 63.5, 45.0, overlap=False, top=-0.4, height=1.8))
        self.assertLess(m.top_y, 0.0)
        self.assertGreater(m.bottom_y, 1.0)

    def test_a_removed_observation_makes_no_mark(self) -> None:
        removed = observation(0, 63.5, 45.0, overlap=False, status=TrackingStatus.REMOVED)
        self.assertEqual(build_marks([removed], set(), WHITE, GEOMETRY), [])


class TestToleranceField(unittest.TestCase):
    """The width says which rule owns this part of the ring, and it is the tolerance wide rather
    than either side of the line, so that two fields overlapping *is* the gate."""

    def test_in_an_overlap_it_is_the_link_angle_centred_on_the_mark(self) -> None:
        m: Mark = mark(observation(0, 120.0, 96.0, overlap=True))
        low, high = span(m)
        self.assertAlmostEqual(high - low, LINK_ANGLE, places=9)
        self.assertAlmostEqual((low + high) / 2.0, 96.0, places=9)

    def test_outside_an_overlap_it_is_the_reacquire_angle(self) -> None:
        # A local-angle rule, converted to azimuth through the SAME cylinder the tracker corrects
        # the azimuth at. The centre is further from the cylinder than the camera is, so the field
        # measures less than the rule: 5 deg of local angle is about 4.1 deg of azimuth at Ø 4.2
        # on the camera's own axis.
        m: Mark = mark(observation(0, 63.5, 45.0, overlap=False))
        low, high = span(m)
        radius: float = PARALLAX_DIAMETER / 2.0
        expected: float = REACQUIRE_ANGLE * (radius - RING_RADIUS) / radius
        self.assertAlmostEqual(high - low, expected, delta=0.05)
        self.assertAlmostEqual((low + high) / 2.0, 45.0, delta=0.01)
        self.assertLess(high - low, REACQUIRE_ANGLE)    # never wider than the rule itself

    def test_the_field_no_longer_depends_on_the_measured_distance(self) -> None:
        """The point of moving the azimuth onto a fixed depth. The field used to convert through
        each observation's own `estimate_distance`, so it breathed as that biased number moved and
        two observations of one person converted by different factors. Now it is the same width
        whatever the box bottom says, which is what makes the pair test exact rather than
        approximate."""
        widths = {mark(observation(0, 63.5, 45.0, overlap=False, distance=d)).tolerance_w
                  for d in (0.6, 2.0, 3.0, 8.0, 40.0)}
        self.assertEqual(len(widths), 1, f'field width still varies with distance: {widths}')

    def test_the_reacquire_field_is_clamped_to_the_camera_field(self) -> None:
        # Past a field edge there are no pixels for a returning person to come back in through,
        # so the window stops there rather than claiming ring the camera cannot see.
        at_edge: Mark = mark(observation(0, CAM_FOV, 100.2, overlap=False))
        inside: Mark = mark(observation(0, CAM_FOV - REACQUIRE_ANGLE, 96.0, overlap=False))
        self.assertLess(at_edge.tolerance_w, inside.tolerance_w)
        self.assertAlmostEqual(at_edge.tolerance_w, inside.tolerance_w / 2.0, delta=0.001)

    def test_a_zero_tolerance_draws_nothing(self) -> None:
        blank: StripGeometry = StripGeometry(
            cam_fov=CAM_FOV, target_fov=TARGET_FOV, ring_radius=RING_RADIUS,
            parallax_diameter=PARALLAX_DIAMETER,
            row_model=(0.78, 0.58), elevation_window=(40.0, -30.0),
            link_angle=0.0, reacquire_angle=0.0)
        for overlap in (True, False):
            with self.subTest(overlap=overlap):
                m: Mark = mark(observation(0, 63.5, 45.0, overlap=overlap), blank)
                self.assertAlmostEqual(m.tolerance_w, 0.0, places=12)
                self.assertEqual(strip_spans(m.tolerance_x, m.tolerance_w), [])


class TestFieldsOverlapExactlyWhenTheTrackerLinks(unittest.TestCase):
    """The one property a reader of the strip acts on, checked against the tracker's own gate.

    Two views of a seam person are joined when their world angles differ by no more than
    `seam.link_angle`. Drawn as two fields each that wide, they touch at exactly that difference —
    so `fields overlap` and `the tracker will link` are the same statement, which is the whole
    reason the field is the tolerance wide and not either side of the line.
    """

    def setUp(self) -> None:
        self.config = PanoramicTrackerSettings(fov=CAM_FOV)
        self.config.seam.link_angle = LINK_ANGLE
        self.config.rig.camera_diameter = RING_RADIUS * 2.0
        self.tracker = PanoramicTracker(self.config, num_players=8, num_cameras=4)

    def fields_overlap(self, a: Mark, b: Mark) -> bool:
        """Do the two drawn fields share any strip, wrap included?"""
        for ax, aw in strip_spans(a.tolerance_x, a.tolerance_w):
            for bx, bw in strip_spans(b.tolerance_x, b.tolerance_w):
                if ax < bx + bw and bx < ax + aw:
                    return True
        return False

    def test_the_drawing_and_the_gate_agree(self) -> None:
        for difference in (0.0, 9.0, 17.5, 18.5, 30.0, 35.5, 40.0):
            with self.subTest(difference=difference):
                a = observation(0, 120.0, 90.0, overlap=True)
                b = observation(1, 8.0, 90.0 + difference, overlap=True)
                links: bool = self.tracker._observations_match(a, b)
                self.assertEqual(self.fields_overlap(mark(a), mark(b)), links,
                                 f'{difference} deg apart: drawn and gated disagree')
                self.assertEqual(links, difference <= LINK_ANGLE)   # and the gate is the gate

    def test_two_people_at_a_seam_do_not_overlap(self) -> None:
        # The rig case: a metre apart at Ø 4.5 is about 25 deg, past the 18 deg gate.
        a, b = observation(0, 120.0, 90.0, overlap=True), observation(1, 8.0, 115.0, overlap=True)
        self.assertFalse(self.tracker._observations_match(a, b))
        self.assertFalse(self.fields_overlap(mark(a), mark(b)))


class TestTheWrap(unittest.TestCase):
    """A field on the azimuth-0 seam is exactly where two cameras meet, so it must not be the one
    that gets silently clipped."""

    def test_a_field_across_zero_splits_into_two_spans_that_sum(self) -> None:
        m: Mark = mark(observation(3, 120.0, 356.0, overlap=True))
        spans = strip_spans(m.tolerance_x, m.tolerance_w)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(sum(w for _x, w in spans) * 360.0, LINK_ANGLE, places=9)

    def test_a_field_just_past_zero_starts_negative_and_wraps_in(self) -> None:
        m: Mark = mark(observation(0, 8.0, 4.0, overlap=True))
        self.assertGreater(m.tolerance_x, 0.9)                      # left edge came back round
        spans = strip_spans(m.tolerance_x, m.tolerance_w)
        self.assertAlmostEqual(sum(w for _x, w in spans) * 360.0, LINK_ANGLE, places=9)


class TestZoneLines(unittest.TestCase):
    """The tracked zone as two rows on the strip — the arithmetic `GridRenderer._zone_lines` does.

    A floor circle of constant radius subtends a constant depression at the rig centre, so each
    edge of the zone is one straight row at every azimuth. They are the only lines on the strip
    measured in metres, which makes them the only ones a tape on the floor can check.
    """

    CAMERA_HEIGHT: float = 0.5
    # The studio strip: P720 up 15 on 960 rows, re-projected to the centre for Ø 4.5.
    WINDOW: tuple[float, float] = (47.38, -17.13)

    def elevation(self, diameter: float) -> float:
        return -math.degrees(math.atan(self.CAMERA_HEIGHT / (diameter / 2.0)))

    def test_a_wider_zone_edge_sits_closer_to_the_horizon(self) -> None:
        near, far = self.elevation(3.0), self.elevation(7.0)
        self.assertAlmostEqual(near, -18.43, delta=0.01)
        self.assertAlmostEqual(far, -8.13, delta=0.01)
        self.assertLess(near, far)                                  # both below the horizon
        self.assertLess(strip_y(far, self.WINDOW), strip_y(near, self.WINDOW))

    def test_the_far_edge_is_inside_the_studio_window_and_the_near_one_is_not(self) -> None:
        """Which is why the zone is drawn as a clipped field rather than two lines: at this preset
        the near edge really is below what the strip can show, and a fill running off the bottom
        says "continues past here" where a line pinned to the boundary would have claimed an
        elevation that is not its own. The strip shows less than the frames do —
        `elevation_window` takes the band at its tightest column so no column fades to black."""
        top, bottom = self.WINDOW
        self.assertLess(self.elevation(3.0), bottom)                 # Ø 3 is 1.3 deg below
        self.assertGreater(self.elevation(7.0), bottom)              # Ø 7 is comfortably inside
        self.assertLess(self.elevation(7.0), top)

    def test_an_edge_beyond_the_horizon_cannot_happen(self) -> None:
        # The floor is always below the lens, so a zone edge is always a depression. Only the
        # bottom of the window can ever clip one.
        for diameter in (0.5, 3.0, 50.0):
            self.assertLess(self.elevation(diameter), 0.0)

    def test_a_mark_inside_the_zone_ends_between_the_two_rows(self) -> None:
        """What the lines are for: a mark's line ends at the foot row, so a person inside the zone
        has that end between the two. Checked through the marks' own row model rather than a second
        copy of it."""
        geometry = StripGeometry(
            cam_fov=CAM_FOV, target_fov=TARGET_FOV, ring_radius=0.0,
            parallax_diameter=PARALLAX_DIAMETER,
            row_model=GEOMETRY.row_model, elevation_window=self.WINDOW,
            link_angle=LINK_ANGLE, reacquire_angle=REACQUIRE_ANGLE)
        near_y = strip_y(self.elevation(3.0), self.WINDOW)
        far_y = strip_y(self.elevation(7.0), self.WINDOW)
        horizon_row, focal_rows = geometry.row_model
        for diameter in (4.0, 4.5, 6.0):
            with self.subTest(diameter=diameter):
                # Feet on the floor at this radius, with the camera at the centre (ring 0, so the
                # camera distance and the centre radius are the same number).
                radius: float = diameter / 2.0
                feet: float = row_from_elevation(self.elevation(diameter), horizon_row, focal_rows)
                t = observation(0, 63.5, 45.0, overlap=False, distance=radius,
                                top=feet - 0.2, height=0.2)
                m: Mark = mark(t, geometry)
                self.assertLess(far_y, m.bottom_y)
                self.assertLess(m.bottom_y, near_y)


if __name__ == "__main__":
    unittest.main()
