"""The panorama's marks: where an observation lands on the strip, and how wide its rule is.

`marks.py` is arithmetic with no GL in it, which is why it can be tested at all — the renderers
around it cannot be, without a context. What is worth pinning here is the part a reader of the
display has to be able to trust: **two fields of one colour that overlap are two observations the
tracker will join.** That is a claim about two separate pieces of code agreeing, so it is asserted
against `Tracker`'s own gate rather than against a second copy of the same comparison.
"""

import math
import unittest
from dataclasses import replace

from modules.render.layers.panorama.marks import Mark, MarkContext, build_marks
from modules.render.layers.panorama.strip import strip_spans, strip_y
from modules.tracker import PanoramicAnnotation, PanoramicTracker, PanoramicTrackerSettings, Rejection, \
    Tracklet, TrackingStatus, row_from_elevation
from modules.utils import Rect


# The White Space rig: four cameras, 127 degrees each, lenses 0.36 m out from the centre.
CAM_FOV: float = 127.0
TARGET_FOV: float = 90.0
RING_RADIUS: float = 0.36
PARALLAX_RADIUS: float = 2.1        # the zone's harmonic mean, what the tracker corrects at
CAMERA_HEIGHT: float = 0.5
LINK_ANGLE: float = 18.0
REACQUIRE_ANGLE: float = 5.0
ZONE_MAX_RADIUS: float = 3.5
LOST_TIMEOUT: float = 2.0
NOW: float = 1_000_000.0            # a fixed clock, so a LOST mark's fade is deterministic

CONTEXT: MarkContext = MarkContext(
    cam_fov=CAM_FOV, target_fov=TARGET_FOV, ring_radius=RING_RADIUS,
    parallax_radius=PARALLAX_RADIUS, camera_height=CAMERA_HEIGHT,
    row_model=(0.78, 0.58), elevation_window=(40.0, -30.0),
    link_angle=LINK_ANGLE, reacquire_angle=REACQUIRE_ANGLE,
    zone_max_radius=ZONE_MAX_RADIUS, lost_timeout=LOST_TIMEOUT, now=NOW,
)

WHITE: list[tuple[float, float, float, float]] = [(1.0, 1.0, 1.0, 1.0)]
RED: list[tuple[float, float, float, float]] = [(1.0, 0.0, 0.0, 1.0)]
GREY: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 0.8)


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


def mark(tracklet: Tracklet, context: MarkContext = CONTEXT) -> Mark:
    marks: list[Mark] = build_marks([tracklet], set(), WHITE, GREY, context)
    assert len(marks) == 1
    return marks[0]


def span(m: Mark) -> tuple[float, float]:
    """A mark's field in degrees, as (low, high) — high may pass 360 when it wraps."""
    return (m.field_x * 360.0, (m.field_x + m.field_w) * 360.0)


class TestPlacement(unittest.TestCase):

    def test_x_is_the_world_angle(self) -> None:
        # Not the local angle, and not the picture: this view's azimuth, as the tracker derived it.
        self.assertAlmostEqual(mark(observation(0, 120.0, 96.0, overlap=True)).x,
                               96.0 / 360.0, places=9)

    def test_the_head_row_is_not_clamped_to_the_strip(self) -> None:
        # The device extrapolates a partly visible person, and that is real information about how
        # close they are. The renderer clips when it draws; a mark must not throw it away first.
        m: Mark = mark(observation(0, 63.5, 45.0, overlap=False, top=-0.4, height=1.8))
        self.assertLess(m.top_y, 0.0)

    def test_the_foot_row_follows_the_distance_and_not_the_box(self) -> None:
        """The foot row is the FLOOR at the reported radius, not the box's bottom pixel — which is
        what makes it exact against the zone band, and what keeps `foot_offset` out of this file:
        the correction arrives already applied, inside `distance`."""
        rows = {mark(observation(0, 63.5, 45.0, overlap=False, distance=3.0,
                                 top=0.1, height=h)).bottom_y for h in (0.2, 0.5, 1.4)}
        self.assertEqual(len(rows), 1, f'foot row still moves with the box: {rows}')
        near: Mark = mark(observation(0, 63.5, 45.0, overlap=False, distance=1.5))
        far: Mark = mark(observation(0, 63.5, 45.0, overlap=False, distance=6.0))
        self.assertGreater(near.bottom_y, far.bottom_y)     # nearer feet are lower on the strip

    def test_someone_past_the_far_edge_reads_where_they_are(self) -> None:
        """The distance is unclamped, so a person at R 5 is labelled R5.0 and their foot tick sits
        above the zone's R 3.5 row — which on the strip is what "no longer seen" looks like —
        instead of being piled up at the edge with everyone else beyond it."""
        on_axis: float = CAM_FOV / 2.0
        m: Mark = mark(observation(0, on_axis, 45.0, overlap=False, distance=5.0 - RING_RADIUS))
        self.assertIn('R5.0m', m.label)
        far_edge_y: float = strip_y(-math.degrees(math.atan(CAMERA_HEIGHT / 3.5)),
                                    CONTEXT.elevation_window)
        self.assertLess(m.bottom_y, far_edge_y)                   # higher on the strip = further

    def test_feet_not_on_the_floor_still_make_a_mark_but_no_foot_tick(self) -> None:
        # `inf`: no distance to put the rows through, so the line is placed on the parallax cylinder
        # instead — it is still drawn, so it cannot vanish — and there is no floor reading to tick.
        m: Mark = mark(observation(0, 63.5, 45.0, overlap=False, distance=math.inf))
        self.assertFalse(m.has_foot)
        self.assertTrue(math.isfinite(m.top_y) and math.isfinite(m.bottom_y))
        self.assertLess(m.top_y, m.bottom_y)
        self.assertIn('R-', m.label)

    def test_a_removed_observation_makes_no_mark(self) -> None:
        removed = observation(0, 63.5, 45.0, overlap=False, status=TrackingStatus.REMOVED)
        self.assertEqual(build_marks([removed], set(), WHITE, GREY, CONTEXT), [])


def rejected(reason: Rejection, **kwargs) -> Tracklet:
    """A detection a filter rejected, as the tracker publishes it: no world, with its rejection."""
    t: Tracklet = observation(0, 63.5, 45.0, overlap=False, **kwargs)
    assert isinstance(t.annotation, PanoramicAnnotation)
    return replace(t, id=-1, annotation=replace(t.annotation, rejected=reason))


class TestRejectedDetections(unittest.TestCase):
    """A detection the tracker did not count is still a mark: a grey line, labelled with why."""

    def test_it_is_a_grey_line_that_joins_nothing(self) -> None:
        m: Mark = mark(rejected(Rejection.SMALL))
        self.assertTrue(m.rejected)
        self.assertEqual(m.color, GREY)
        self.assertEqual(m.field_color[3], 0.0)                     # no field
        self.assertAlmostEqual(m.field_w, 0.0, places=12)
        self.assertEqual(strip_spans(m.field_x, m.field_w), [])

    def test_each_rejection_labels_itself(self) -> None:
        for reason, text in ((Rejection.YOUNG, 'young'), (Rejection.SMALL, 'small'),
                             (Rejection.DEAD_ZONE, 'dead zone'), (Rejection.PAST_EDGE, 'past R3.5'),
                             (Rejection.NO_ID, 'no id')):
            with self.subTest(reason=reason):
                self.assertEqual(mark(rejected(reason)).label, text)

    def test_rejected_marks_are_drawn_first(self) -> None:
        marks = build_marks([observation(0, 63.5, 45.0, overlap=False), rejected(Rejection.YOUNG)],
                            set(), WHITE, GREY, CONTEXT)
        self.assertEqual([m.rejected for m in marks], [True, False])

    def test_a_tracked_person_past_the_edge_keeps_their_own_mark(self) -> None:
        # Rejected but holding a world: its own mark, not a grey line, and the label says why it is fading.
        t: Tracklet = observation(0, 63.5, 45.0, overlap=False, status=TrackingStatus.LOST)
        assert isinstance(t.annotation, PanoramicAnnotation)
        t = replace(t, id=0, last_active=NOW, annotation=replace(t.annotation, rejected=Rejection.PAST_EDGE))
        m: Mark = build_marks([t], set(), RED, GREY, CONTEXT)[0]
        self.assertFalse(m.rejected)
        self.assertTrue(m.label.endswith(' past R3.5'))
        self.assertTrue(m.label.startswith('#0 c0'))


class TestLostFade(unittest.TestCase):
    """A LOST observation's line fades from its world colour to grey over `lost_timeout`, and its
    field fades out, keeping its colour."""

    def lost(self, seconds_lost: float, status: TrackingStatus = TrackingStatus.LOST) -> Mark:
        t: Tracklet = replace(observation(0, 63.5, 45.0, overlap=False, status=status),
                              id=0, last_active=NOW - seconds_lost)
        return build_marks([t], set(), RED, GREY, CONTEXT)[0]

    def colour(self, seconds_lost: float, status: TrackingStatus = TrackingStatus.LOST):
        return self.lost(seconds_lost, status).color

    def test_the_field_keeps_its_colour_and_fades_out(self) -> None:
        self.assertEqual(self.lost(0.0).field_color, (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(self.lost(LOST_TIMEOUT / 2.0).field_color, (1.0, 0.0, 0.0, 0.5))
        self.assertEqual(self.lost(LOST_TIMEOUT).field_color, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.lost(LOST_TIMEOUT, TrackingStatus.TRACKED).field_color, (1.0, 0.0, 0.0, 1.0))

    def test_a_lost_field_is_outlined_even_as_primary(self) -> None:
        t: Tracklet = replace(observation(0, 63.5, 45.0, overlap=False, status=TrackingStatus.LOST),
                              id=0, last_active=NOW)
        self.assertTrue(build_marks([t], {t.obs_id}, RED, GREY, CONTEXT)[0].field_outline)

    def test_fresh_is_the_world_colour(self) -> None:
        self.assertEqual(self.colour(0.0), (1.0, 0.0, 0.0, 0.8))

    def test_at_the_timeout_it_is_grey(self) -> None:
        r, g, b, a = self.colour(LOST_TIMEOUT)
        self.assertAlmostEqual(r, GREY[0]); self.assertAlmostEqual(g, GREY[1]); self.assertAlmostEqual(b, GREY[2])
        self.assertEqual(self.colour(LOST_TIMEOUT * 3.0)[:3], self.colour(LOST_TIMEOUT)[:3])   # and stays

    def test_halfway_is_halfway(self) -> None:
        r, g, b, a = self.colour(LOST_TIMEOUT / 2.0)
        self.assertAlmostEqual(r, (1.0 + GREY[0]) / 2.0)
        self.assertAlmostEqual(g, GREY[1] / 2.0)
        self.assertAlmostEqual(a, 0.8)

    def test_a_tracked_observation_does_not_fade(self) -> None:
        # Only LOST fades; an active passive view keeps its colour at the passive alpha however old.
        self.assertEqual(self.colour(LOST_TIMEOUT, status=TrackingStatus.TRACKED), (1.0, 0.0, 0.0, 0.8))


class TestPassiveField(unittest.TestCase):
    """The primary's field is filled; a passive view's is outlined, at full visibility."""

    def test_primary_filled_passive_outlined(self) -> None:
        t: Tracklet = replace(observation(0, 63.5, 45.0, overlap=True), id=0)
        primary: Mark = build_marks([t], {t.obs_id}, RED, GREY, CONTEXT)[0]
        passive: Mark = build_marks([t], set(), RED, GREY, CONTEXT)[0]
        self.assertFalse(primary.field_outline)
        self.assertTrue(passive.field_outline)
        self.assertEqual(primary.field_color, passive.field_color)
        self.assertEqual((primary.color[3], passive.color[3]), (1.0, 0.8))


class TestFieldWidth(unittest.TestCase):
    """The width says which rule owns this part of the ring, and it is the rule's angle wide rather
    than either side of the line, so that two fields overlapping *is* the rule."""

    def test_in_an_overlap_it_is_the_link_angle_centred_on_the_mark(self) -> None:
        m: Mark = mark(observation(0, 120.0, 96.0, overlap=True))
        low, high = span(m)
        self.assertAlmostEqual(high - low, LINK_ANGLE, places=9)
        self.assertAlmostEqual((low + high) / 2.0, 96.0, places=9)

    def test_outside_an_overlap_it_is_the_reacquire_angle(self) -> None:
        # A local-angle rule, converted to azimuth through the SAME cylinder the tracker corrects
        # the azimuth at. The centre is further from the cylinder than the camera is, so the field
        # measures less than the rule: 5 deg of local angle is about 4.1 deg of azimuth at R 2.1
        # on the camera's own axis.
        m: Mark = mark(observation(0, 63.5, 45.0, overlap=False))
        low, high = span(m)
        radius: float = PARALLAX_RADIUS
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
        widths = {mark(observation(0, 63.5, 45.0, overlap=False, distance=d)).field_w
                  for d in (0.6, 2.0, 3.0, 8.0, 40.0)}
        self.assertEqual(len(widths), 1, f'field width still varies with distance: {widths}')

    def test_the_reacquire_field_is_clamped_to_the_camera_field(self) -> None:
        # Past a field edge there are no pixels for a returning person to come back in through,
        # so the window stops there rather than claiming ring the camera cannot see.
        at_edge: Mark = mark(observation(0, CAM_FOV, 100.2, overlap=False))
        inside: Mark = mark(observation(0, CAM_FOV - REACQUIRE_ANGLE, 96.0, overlap=False))
        self.assertLess(at_edge.field_w, inside.field_w)
        self.assertAlmostEqual(at_edge.field_w, inside.field_w / 2.0, delta=0.001)

    def test_a_zero_angle_draws_no_field(self) -> None:
        blank: MarkContext = MarkContext(
            cam_fov=CAM_FOV, target_fov=TARGET_FOV, ring_radius=RING_RADIUS,
            parallax_radius=PARALLAX_RADIUS, camera_height=CAMERA_HEIGHT,
            row_model=(0.78, 0.58), elevation_window=(40.0, -30.0),
            link_angle=0.0, reacquire_angle=0.0,
            zone_max_radius=ZONE_MAX_RADIUS, lost_timeout=LOST_TIMEOUT, now=NOW)
        for overlap in (True, False):
            with self.subTest(overlap=overlap):
                m: Mark = mark(observation(0, 63.5, 45.0, overlap=overlap), blank)
                self.assertAlmostEqual(m.field_w, 0.0, places=12)
                self.assertEqual(strip_spans(m.field_x, m.field_w), [])


class TestFieldsOverlapExactlyWhenTheTrackerLinks(unittest.TestCase):
    """The one property a reader of the strip acts on, checked against the tracker's own gate.

    Two views of a seam person are joined when their world angles differ by no more than
    `seam.link_angle`. Drawn as two fields each that wide, they touch at exactly that difference —
    so `fields overlap` and `the tracker will link` are the same statement, which is the whole
    reason the field is the rule's angle wide and not either side of the line.
    """

    def setUp(self) -> None:
        self.config = PanoramicTrackerSettings(fov=CAM_FOV)
        self.config.seam.link_angle = LINK_ANGLE
        self.config.rig.camera_radius = RING_RADIUS
        self.tracker = PanoramicTracker(self.config, num_players=8, num_cameras=4)

    def fields_overlap(self, a: Mark, b: Mark) -> bool:
        """Do the two drawn fields share any strip, wrap included?"""
        for ax, aw in strip_spans(a.field_x, a.field_w):
            for bx, bw in strip_spans(b.field_x, b.field_w):
                if ax < bx + bw and bx < ax + aw:
                    return True
        return False

    def test_the_drawing_and_the_gate_agree(self) -> None:
        for difference in (0.0, 9.0, 17.5, 18.5, 30.0, 35.5, 40.0):
            with self.subTest(difference=difference):
                a = observation(0, 120.0, 90.0, overlap=True)
                b = observation(1, 8.0, 90.0 + difference, overlap=True)
                links: bool = self.tracker.seams._observations_match(a, b)
                self.assertEqual(self.fields_overlap(mark(a), mark(b)), links,
                                 f'{difference} deg apart: drawn and gated disagree')
                self.assertEqual(links, difference <= LINK_ANGLE)   # and the gate is the gate

    def test_two_people_at_a_seam_do_not_overlap(self) -> None:
        # The rig case: a metre apart at R 2.25 is about 25 deg, past the 18 deg gate.
        a, b = observation(0, 120.0, 90.0, overlap=True), observation(1, 8.0, 115.0, overlap=True)
        self.assertFalse(self.tracker.seams._observations_match(a, b))
        self.assertFalse(self.fields_overlap(mark(a), mark(b)))


class TestTheWrap(unittest.TestCase):
    """A field on the azimuth-0 seam is exactly where two cameras meet, so it must not be the one
    that gets silently clipped."""

    def test_a_field_across_zero_splits_into_two_spans_that_sum(self) -> None:
        m: Mark = mark(observation(3, 120.0, 356.0, overlap=True))
        spans = strip_spans(m.field_x, m.field_w)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(sum(w for _x, w in spans) * 360.0, LINK_ANGLE, places=9)

    def test_a_field_just_past_zero_starts_negative_and_wraps_in(self) -> None:
        m: Mark = mark(observation(0, 8.0, 4.0, overlap=True))
        self.assertGreater(m.field_x, 0.9)                      # left edge came back round
        spans = strip_spans(m.field_x, m.field_w)
        self.assertAlmostEqual(sum(w for _x, w in spans) * 360.0, LINK_ANGLE, places=9)


class TestZoneLines(unittest.TestCase):
    """The tracked zone as two rows on the strip — the arithmetic `GridRenderer._zone_lines` does.

    A floor circle of constant radius subtends a constant depression at the rig centre, so each
    edge of the zone is one straight row at every azimuth. They are the only lines on the strip
    measured in metres, which makes them the only ones a tape on the floor can check.
    """

    CAMERA_HEIGHT: float = 0.5
    # The studio strip: P720 up 15 on 960 rows, re-projected to the centre for R 2.25.
    WINDOW: tuple[float, float] = (47.38, -17.13)

    def elevation(self, radius: float) -> float:
        return -math.degrees(math.atan(self.CAMERA_HEIGHT / radius))

    def test_a_wider_zone_edge_sits_closer_to_the_horizon(self) -> None:
        near, far = self.elevation(1.5), self.elevation(3.5)
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
        self.assertLess(self.elevation(1.5), bottom)                 # R 1.5 is 1.3 deg below
        self.assertGreater(self.elevation(3.5), bottom)              # R 3.5 is comfortably inside
        self.assertLess(self.elevation(3.5), top)

    def test_an_edge_beyond_the_horizon_cannot_happen(self) -> None:
        # The floor is always below the lens, so a zone edge is always a depression. Only the
        # bottom of the window can ever clip one.
        for radius in (0.25, 1.5, 25.0):
            self.assertLess(self.elevation(radius), 0.0)

    def test_a_mark_inside_the_zone_ends_between_the_two_rows(self) -> None:
        """What the lines are for: a mark's line ends at the foot row, so a person inside the zone
        has that end between the two. Checked through the marks' own row model rather than a second
        copy of it."""
        context = MarkContext(
            cam_fov=CAM_FOV, target_fov=TARGET_FOV, ring_radius=0.0,
            parallax_radius=PARALLAX_RADIUS, camera_height=self.CAMERA_HEIGHT,
            row_model=CONTEXT.row_model, elevation_window=self.WINDOW,
            link_angle=LINK_ANGLE, reacquire_angle=REACQUIRE_ANGLE,
            zone_max_radius=ZONE_MAX_RADIUS, lost_timeout=LOST_TIMEOUT, now=NOW)
        near_y = strip_y(self.elevation(1.5), self.WINDOW)
        far_y = strip_y(self.elevation(3.5), self.WINDOW)
        horizon_row, focal_rows = context.row_model
        for radius in (2.0, 2.25, 3.0):
            with self.subTest(radius=radius):
                # Feet on the floor at this radius, with the camera at the centre (ring 0, so the
                # camera distance and the centre radius are the same number).
                feet: float = row_from_elevation(self.elevation(radius), horizon_row, focal_rows)
                t = observation(0, 63.5, 45.0, overlap=False, distance=radius,
                                top=feet - 0.2, height=0.2)
                m: Mark = mark(t, context)
                self.assertLess(far_y, m.bottom_y)
                self.assertLess(m.bottom_y, near_y)

    def test_the_foot_tick_is_exactly_the_zone_lines_own_formula(self) -> None:
        """**The instrument.** Standing on a taped circle, the tick must land on that zone edge —
        so "the tick is on the R 3.5 line" has to mean "the tracker reports this person at R 3.5", to
        the pixel, and not approximately.

        It does because the rows go through the person's OWN distance, where the lens height
        cancels out of the conversion algebraically:

            atan(tan(-atan(h/d)) * d / R) = atan(-h/R)

        the right-hand side being exactly what `GridRenderer._zone_band` draws. Put the rows on the
        parallax cylinder instead — tidier, since the x uses it — and this breaks by 20 px at R 1.5.
        That is what this test exists to catch.
        """
        context = MarkContext(
            cam_fov=CAM_FOV, target_fov=TARGET_FOV, ring_radius=RING_RADIUS,
            parallax_radius=PARALLAX_RADIUS, camera_height=self.CAMERA_HEIGHT,
            row_model=CONTEXT.row_model, elevation_window=self.WINDOW,
            link_angle=LINK_ANGLE, reacquire_angle=REACQUIRE_ANGLE,
            zone_max_radius=ZONE_MAX_RADIUS, lost_timeout=LOST_TIMEOUT, now=NOW)
        horizon_row, focal_rows = context.row_model
        for radius in (1.5, PARALLAX_RADIUS, 2.5, 3.5):
            for local_angle in (63.5, 20.0, 110.0):     # on the axis and well off it
                with self.subTest(radius=radius, local_angle=local_angle):
                    # A person on that circle, as the camera sees them: the bearing off its axis
                    # fixes the camera distance, and the feet then fix the row.
                    bearing: float = math.radians(local_angle - CAM_FOV / 2.0)
                    # Camera `RING_RADIUS` out, aimed radially: solve for its distance to the circle.
                    cam_distance: float = -RING_RADIUS * math.cos(bearing) + math.sqrt(
                        radius ** 2 - (RING_RADIUS * math.sin(bearing)) ** 2)
                    depression: float = math.degrees(math.atan(self.CAMERA_HEIGHT / cam_distance))
                    feet: float = row_from_elevation(-depression, horizon_row, focal_rows)
                    m: Mark = mark(observation(0, local_angle, 45.0, overlap=False,
                                               distance=cam_distance, top=feet - 0.2, height=0.2),
                                   context)
                    # The zone band's own edge, for that radius.
                    zone_y: float = strip_y(
                        -math.degrees(math.atan(self.CAMERA_HEIGHT / radius)), self.WINDOW)
                    self.assertAlmostEqual(m.bottom_y, zone_y, places=9)


if __name__ == "__main__":
    unittest.main()
