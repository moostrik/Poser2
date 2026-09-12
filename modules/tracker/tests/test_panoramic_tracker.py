"""Tests for the panoramic tracker: seam hysteresis, dead-zone handling,
cross-camera linking, same-camera re-acquisition, device id reuse, emission hold,
world id reuse, and ring parallax correction."""

import math
import time
import unittest
from dataclasses import replace

from modules.oak import frame_window, delivered_height
from modules.tracker import (
    PanoramicTracker, PanoramicTrackerSettings, PanoramicAnnotation,
    Tracklet, TrackingStatus, TrackletDict, row_from_elevation,
)
from modules.tracker.panoramic.geometry import Geometry, _MAX_HEIGHT, height_is_measured
from modules.tracker.panoramic.store import TrackletIdPool
from modules.utils import Rect


# fov 110 / target 90 -> `fov_overlap` 10 deg of offset, and a 20 deg band of each camera's field
# that its neighbour also sees. With the default seam settings (dead_zone 5, link_angle 8,
# link_height 15, hysteresis 0.9): no births at local angle <= 5 or >= 105; a second opinion
# exists at <= 20 or >= 90; two cameras' observations are one person within 8 deg of world
# azimuth and 15% of measured height.
FOV = 110.0


def make_tracklet(cam_id: int, ext_id: int, local_angle: float, *,
                  status: TrackingStatus = TrackingStatus.TRACKED,
                  height: float = 0.5, age: int = 10, top: float = 0.1) -> Tracklet:
    width = 0.1
    center_x = local_angle / FOV
    roi = Rect(x=center_x - width / 2.0, y=top, width=width, height=height)
    return Tracklet(cam_id=cam_id, status=status, roi=roi,
                    external_id=ext_id, external_age_in_frames=age)


class PanoramicTrackerCase(unittest.TestCase):
    """Drives the tracker synchronously via _add_tracklet/_update_and_notify;
    the background thread is never started."""

    def setUp(self) -> None:
        self.config = PanoramicTrackerSettings(fov=FOV)
        self.tracker = PanoramicTracker(self.config, num_players=8, num_cameras=4)
        self.emitted: list[TrackletDict] = []
        self.tracker.add_tracklet_callback(self.emitted.append)

    def person(self, cam_id: int, ext_id: int, local_angle: float, *,
               height_m: float = 1.8, distance: float = 3.0,
               status: TrackingStatus = TrackingStatus.TRACKED, age: int = 10,
               feet_off_floor: float = 0.0) -> Tracklet:
        """A tracklet whose box reads back as a `height_m` person `distance` m out.

        Built from the row model the tracker itself published, so the gates are exercised on the
        metres they are written in rather than on hand-picked frame fractions. `feet_off_floor`
        lifts the whole box by that many metres — a jump, which the floor-plane model cannot
        represent and so reads as someone smaller and further away.
        """
        p = self.config.rig
        lift: float = p.focal_rows * feet_off_floor / distance
        bottom: float = p.horizon_row + p.focal_rows * p.camera_height / distance - lift
        box_h: float = p.focal_rows * height_m / distance
        return make_tracklet(cam_id, ext_id, local_angle, status=status, age=age,
                             height=box_h, top=bottom - box_h)

    def submit(self, *tracklets: Tracklet) -> TrackletDict:
        for t in tracklets:
            self.tracker._add_tracklet(t)
        self.tracker._update_and_notify()
        return self.emitted[-1]


class TestDeviceIdReuse(PanoramicTrackerCase):
    """The device tracker hands out the smallest free id, so a departed person's number goes
    straight to the next arrival — while our observation of the departed person is still LOST
    and still anchoring seam crossings. Identity must come from where a person is, never from
    the number the device happens to be using."""

    def test_reused_id_elsewhere_does_not_inherit_the_world(self) -> None:
        self.submit(make_tracklet(0, 1, 50.0))
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 0)
        # The device drops the track: its id 1 is now free.
        self.submit(make_tracklet(0, 1, 50.0, status=TrackingStatus.REMOVED))
        # A different person appears 30 deg away and the device hands them id 1 again.
        out = self.submit(make_tracklet(0, 1, 80.0))
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 1)   # a NEW world, not 0
        self.assertEqual(set(out.keys()), {0, 1})                    # the old one still anchors

    def test_reacquired_in_place_under_the_same_id_keeps_the_world(self) -> None:
        # The other half: the device losing and re-finding the same person must not split them.
        self.submit(make_tracklet(0, 1, 50.0))
        self.submit(make_tracklet(0, 1, 50.0, status=TrackingStatus.REMOVED))
        out = self.submit(make_tracklet(0, 1, 52.0))
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 0)
        self.assertEqual(set(out.keys()), {0})

    def test_reacquired_in_place_under_a_new_id_keeps_the_world(self) -> None:
        # `UNIQUE_ID` on the device would make every re-acquisition look like this.
        self.submit(make_tracklet(0, 1, 50.0))
        self.submit(make_tracklet(0, 1, 50.0, status=TrackingStatus.REMOVED))
        out = self.submit(make_tracklet(0, 7, 51.0))
        self.assertEqual(self.tracker.store.get_world_id(0, 7), 0)
        self.assertEqual(set(out.keys()), {0})

    def test_relink_is_bounded_by_angle_alone(self) -> None:
        # The reason a person was dropped is usually that something changed about their box, so a
        # re-acquisition asks only "is this the same place": `reacquire_angle`, no height gate.
        self.submit(self.person(0, 1, 50.0, height_m=1.8, distance=3.0))
        self.submit(self.person(0, 1, 50.0, status=TrackingStatus.REMOVED))
        out = self.submit(self.person(0, 2, 52.0, height_m=1.2, distance=1.8))
        self.assertEqual(self.tracker.store.get_world_id(0, 2), 0)   # same world
        self.assertEqual(set(out.keys()), {0})

    def test_relink_refused_beyond_the_reacquire_angle(self) -> None:
        self.config.reacquire_angle = 5.0
        self.submit(self.person(0, 1, 50.0))
        self.submit(self.person(0, 1, 50.0, status=TrackingStatus.REMOVED))
        out = self.submit(self.person(0, 2, 60.0))
        self.assertEqual(self.tracker.store.get_world_id(0, 2), 1)   # too far: a new world
        self.assertEqual(set(out.keys()), {0, 1})

    def test_lost_anchor_still_links_across_a_seam_after_its_id_is_reused(self) -> None:
        # The reason a REMOVED observation keeps anchoring at all: the far camera has to be able
        # to pick a person up after the near camera has given up on them.
        self.submit(make_tracklet(0, 1, 98.0))                       # seam person, world 0
        self.submit(make_tracklet(0, 1, 98.0, status=TrackingStatus.REMOVED))
        self.submit(make_tracklet(0, 1, 40.0))                       # id 1 reused mid-field
        out = self.submit(make_tracklet(1, 1, 8.0))                  # cam1 sees the seam person
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)   # links to the anchor
        self.assertIn(0, out)

    def test_a_new_observation_is_a_new_observation(self) -> None:
        # Host-owned ids: reusing a device number must not reuse our record of it.
        self.submit(make_tracklet(0, 1, 50.0))
        first = self.tracker.store.get_live_observation(0, 1)
        self.submit(make_tracklet(0, 1, 50.0, status=TrackingStatus.REMOVED))
        self.submit(make_tracklet(0, 1, 52.0))
        second = self.tracker.store.get_live_observation(0, 1)
        assert first is not None and second is not None
        self.assertNotEqual(first.obs_id, second.obs_id)


class TestEmitTimeout(PanoramicTrackerCase):
    """`lost_timeout` and `emit_timeout` do two different jobs, so they are two settings: an
    observation must keep anchoring long after the person it describes should stop driving the
    show."""

    def test_a_stale_world_stops_being_emitted_but_keeps_anchoring(self) -> None:
        stale = replace(make_tracklet(0, 1, 50.0), last_active=time.time() - 1.0)
        out = self.submit(stale)
        self.assertEqual(out, {})                                      # 1.0 s > emit_timeout 0.3
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 0)     # 1.0 s < lost_timeout 2.0

    def test_a_fresh_world_is_emitted(self) -> None:
        out = self.submit(make_tracklet(0, 1, 50.0))
        self.assertEqual(set(out.keys()), {0})


class TestObservationChannel(PanoramicTrackerCase):
    """The unfused view the calibration display needs: one entry per camera that can see a
    person, where the primary channel gives one entry per person."""

    def setUp(self) -> None:
        super().setUp()
        self.observed: list[list[Tracklet]] = []
        self.tracker.add_observation_callback(self.observed.append)

    def test_a_seam_person_appears_once_per_camera(self) -> None:
        self.submit(make_tracklet(0, 1, 98.0))
        out = self.submit(make_tracklet(1, 1, 8.0))
        self.assertEqual(set(out.keys()), {0})                        # fused: one person
        obs = self.observed[-1]
        self.assertEqual(len(obs), 2)                                 # unfused: two cameras
        self.assertEqual({o.cam_id for o in obs}, {0, 1})
        self.assertEqual({o.id for o in obs}, {0})                    # both carry the same world
        for o in obs:
            assert isinstance(o.annotation, PanoramicAnnotation)
            self.assertGreater(o.annotation.distance, 0.0)            # the label the view shows

    def test_each_camera_keeps_its_own_opinion(self) -> None:
        # What makes the display a calibration tool: each camera's own reading survives the
        # fusion instead of being replaced by the winner's. Here the two agree, because the
        # geometry is consistent — cam0's local 98 and cam1's local 8 are the same azimuth 88.
        # A disagreement is precisely what a wrong `fov`, `tilt` or `ring_radius` produces, and
        # what the stitched view draws as a ghost.
        self.submit(make_tracklet(0, 1, 98.0))
        self.submit(make_tracklet(1, 1, 8.0))
        by_cam = {o.cam_id: o for o in self.observed[-1]}
        self.assertEqual(set(by_cam), {0, 1})
        for cam_id, expected_local in ((0, 98.0), (1, 8.0)):
            ann = by_cam[cam_id].annotation
            assert isinstance(ann, PanoramicAnnotation)
            self.assertAlmostEqual(ann.local_angle, expected_local, places=5)
            self.assertAlmostEqual(ann.world_angle, 88.0, places=5)


class TestPrimaryHysteresis(PanoramicTrackerCase):

    def test_primary_held_through_lost_flicker(self) -> None:
        # Person at the cam0/cam1 seam: cam0 edge distance 12, cam1 edge distance 8
        out = self.submit(make_tracklet(0, 1, 98.0))
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(out[0].cam_id, 0)

        out = self.submit(make_tracklet(1, 1, 8.0))
        self.assertEqual(set(out.keys()), {0})  # linked into the same world
        self.assertEqual(out[0].cam_id, 0)      # cam0 stays primary

        # A transient LOST on the primary must not hand off to the other camera
        out = self.submit(make_tracklet(0, 1, 98.0, status=TrackingStatus.LOST))
        self.assertEqual(out[0].cam_id, 0)

        out = self.submit(make_tracklet(0, 1, 98.0))
        self.assertEqual(out[0].cam_id, 0)

    def test_crossing_hands_off_once_hysteresis_cleared(self) -> None:
        self.submit(make_tracklet(0, 1, 98.0))
        out = self.submit(make_tracklet(1, 1, 8.0))
        self.assertEqual(out[0].cam_id, 0)

        # Walk towards cam1: cam0 edge distance 6, cam1 edge distance 14;
        # 14 >= 6 / 0.9 clears the hysteresis ratio.
        out = self.submit(make_tracklet(0, 1, 104.0), make_tracklet(1, 1, 14.0))
        self.assertEqual(set(out.keys()), {0})  # same world id across the seam
        self.assertEqual(out[0].cam_id, 1)

    def test_competitor_below_hysteresis_does_not_take_over(self) -> None:
        self.submit(make_tracklet(0, 1, 98.0))
        out = self.submit(make_tracklet(1, 1, 8.0))
        # cam1 edge distance 8 < cam0 edge distance 12 / 0.9
        self.assertEqual(out[0].cam_id, 0)


class TestDeadZone(PanoramicTrackerCase):

    def test_existing_observation_refreshed_inside_dead_zone(self) -> None:
        self.submit(make_tracklet(0, 1, 98.0))
        out = self.submit(make_tracklet(0, 1, 107.0))  # inside the dead zone
        self.assertEqual(out[0].cam_id, 0)
        annotation = out[0].annotation
        self.assertIsInstance(annotation, PanoramicAnnotation)
        assert isinstance(annotation, PanoramicAnnotation)
        self.assertAlmostEqual(annotation.local_angle, 107.0, places=5)

    def test_new_observation_rejected_inside_dead_zone(self) -> None:
        out = self.submit(make_tracklet(0, 7, 107.0))
        self.assertEqual(out, {})


class TestCrossCameraLinking(PanoramicTrackerCase):

    def test_links_to_lost_anchor(self) -> None:
        self.submit(make_tracklet(0, 1, 98.0))
        self.submit(make_tracklet(0, 1, 98.0, status=TrackingStatus.LOST))
        out = self.submit(make_tracklet(1, 1, 8.0))
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)

    def test_one_person_seen_from_two_distances_passes_the_height_gate(self) -> None:
        # The point of gating on metres. Two cameras at a seam are at genuinely different
        # distances from the same person, so their box heights in pixels differ by tens of
        # percent while the measured heights agree exactly.
        near = self.person(0, 1, 98.0, height_m=1.8, distance=2.0)
        far = self.person(1, 1, 8.0, height_m=1.8, distance=3.5)
        self.assertGreater(near.roi.height, far.roi.height * 1.5)   # the pixels disagree a lot
        self.submit(near)
        out = self.submit(far)
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)

    def test_height_gate_rejects_a_genuinely_different_person(self) -> None:
        # Same azimuth, 1.2 m against 1.8 m: 33% apart, past the 15% gate.
        self.submit(self.person(0, 1, 98.0, height_m=1.8, distance=2.5))
        out = self.submit(self.person(1, 1, 8.0, height_m=1.2, distance=2.5))
        self.assertEqual(set(out.keys()), {0, 1})  # link refused -> new world

    def test_a_jumper_links_on_azimuth_alone(self) -> None:
        # Feet off the floor breaks the one assumption the height measurement rests on, so the
        # reading is not a measurement and must not be allowed to veto a link the azimuth
        # supports — a person is never harder to re-find than mid-jump.
        self.submit(self.person(0, 1, 98.0, height_m=1.8, distance=2.5))
        jumper = self.person(1, 1, 8.0, height_m=1.8, distance=2.5, feet_off_floor=0.45)
        out = self.submit(jumper)
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)
        annotation = self.tracker.store.get_live_observation(1, 1)
        assert annotation is not None and isinstance(annotation.annotation, PanoramicAnnotation)
        # It really is a bad reading, not a lucky one that happened to fall inside the gate.
        self.assertFalse(height_is_measured(annotation.annotation.height)
                         and abs(annotation.annotation.height - 1.8) < 0.15 * 1.8)

    def test_two_people_inside_the_overlap_stay_two_worlds(self) -> None:
        # 12 deg of world azimuth apart, both inside the cam0/cam1 overlap band (20 deg in from
        # each field edge here): past `link_angle` 8, so the two cameras' views must not be
        # fused, and the collapse net must not repair it either.
        self.config.seam.link_angle = 8.0
        self.submit(self.person(0, 1, 96.0))         # world angle 86
        out = self.submit(self.person(1, 2, 18.0))   # world angle 98
        self.assertEqual(set(out.keys()), {0, 1})
        self.assertEqual(self.tracker.store.get_world_id(1, 2), 1)

    def test_two_people_fuse_at_a_loose_link_angle(self) -> None:
        # The other half of the same fixture: the gate, not the geometry, is what kept them
        # apart — which is what makes the preset's 20.5 worth shrinking on the rig.
        self.config.seam.link_angle = 20.5
        self.submit(self.person(0, 1, 96.0))
        out = self.submit(self.person(1, 2, 18.0))
        self.assertEqual(set(out.keys()), {0})

    def test_ambiguous_link_picks_nearest_angle(self) -> None:
        self.submit(make_tracklet(0, 1, 98.0))   # world 0 at world angle 88
        self.submit(make_tracklet(0, 2, 102.0))  # world 1 at world angle 92
        # New cam1 observation at world angle 90.5 matches both worlds within
        # reach; it must link to the nearest (world 1), and the collapse net
        # must not merge world 0 into it afterwards (its mutual nearest match
        # is its own-world partner, not world 0's observation).
        out = self.submit(make_tracklet(1, 1, 10.5))
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 1)
        self.assertEqual(set(out.keys()), {0, 1})

    def test_collapse_repairs_simultaneous_arrivals(self) -> None:
        # Two observations of the same person arrive in the same batch: linking
        # cannot happen at ingest for the second one only if it arrives first,
        # so force the split by submitting both as brand-new in one tick.
        self.tracker._add_tracklet(self.person(0, 1, 98.0, height_m=1.2, distance=2.5))
        self.tracker._add_tracklet(self.person(1, 1, 8.0, height_m=1.8, distance=2.5))
        # 33% apart in height kept them apart; now both read the same and the
        # per-tick collapse merges the younger world into the older.
        self.tracker._add_tracklet(self.person(0, 1, 98.0, height_m=1.8, distance=2.5))
        self.tracker._update_and_notify()
        out = self.emitted[-1]
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)


class TestTrackletIdPool(unittest.TestCase):

    def test_fifo_reuse(self) -> None:
        pool = TrackletIdPool(4)
        self.assertEqual([pool.acquire() for _ in range(3)], [0, 1, 2])
        pool.release(0)
        pool.release(1)
        # Freed ids go to the back: 3 is handed out before 0 and 1 come around
        self.assertEqual([pool.acquire() for _ in range(3)], [3, 0, 1])

    def test_release_of_free_id_raises(self) -> None:
        pool = TrackletIdPool(2)
        with self.assertRaises(Exception):
            pool.release(0)

    def test_exhaustion_raises(self) -> None:
        pool = TrackletIdPool(1)
        pool.acquire()
        with self.assertRaises(Exception):
            pool.acquire()

    def test_availability_tracking(self) -> None:
        pool = TrackletIdPool(2)
        self.assertTrue(pool.is_available(0))
        acquired = pool.acquire()
        self.assertFalse(pool.is_available(acquired))
        self.assertEqual(pool.size(), 1)
        pool.release(acquired)
        self.assertTrue(pool.is_available(acquired))


# Rig geometry for the parallax tests: 4 cameras, fov 127, lens 0.36 m out and 0.5 m up.
PARALLAX_FOV = 127.0
TARGET_FOV = 90.0
RING_RADIUS = 0.36
CAMERA_HEIGHT = 0.5
# The tracked zone, deliberately WIDER than the studio preset's Ø 3 – Ø 7. The distance clamp is
# derived from it (`Geometry.set_zone`), and at Ø 3 the near bound lands on 1.14 m — exactly the
# frame's own nearest readable row at this tilt, which would make
# `test_the_bottom_row_is_the_nearest_readable_distance` pass for the wrong reason. Ø 2 keeps the
# clamp clear of the frame's limit, so each test measures the one thing it names.
ZONE_MIN_DIAMETER = 2.0
ZONE_MAX_DIAMETER = 8.0
# The delivered frame: P800 aimed up 16 on the sensor's full reach (1152 rows), the ideal lens.
# Rows are tangents of elevation below the horizon row (`FrameWindow`), so a depression angle
# is a row through `row_from_elevation`, never `0.5 + angle / vfov`.
ROWS = 1152
WINDOW = frame_window((1280, 800), (1280, ROWS), 1280, PARALLAX_FOV, 16.0)
HORIZON_ROW = WINDOW.horizon_px / (ROWS - 1)
FOCAL_ROWS = WINDOW.focal / (ROWS - 1)


def feet_row(distance: float) -> float:
    """The normalised row where the feet of someone `distance` m out meet the floor."""
    depression = math.degrees(math.atan(CAMERA_HEIGHT / distance))
    return row_from_elevation(-depression, HORIZON_ROW, FOCAL_ROWS)


def head_row(height_m: float, distance: float) -> float:
    """The normalised row of the top of someone `height_m` tall, `distance` m out."""
    elevation = math.degrees(math.atan((height_m - CAMERA_HEIGHT) / distance))
    return row_from_elevation(elevation, HORIZON_ROW, FOCAL_ROWS)


def synth_observation(cam_id: int, world_azimuth: float, radius: float,
                      person_height: float | None = None) -> tuple[Rect, float]:
    """Build the ROI a camera on the ring would report for a person standing at
    ``world_azimuth`` degrees, ``radius`` m from the rig centre. Returns the ROI and the true
    camera->person distance.

    Inverts the tracker's own projection so the parallax correction can be checked against
    ground truth. The frame is cylindrical and level, so the column is the bearing, linearly,
    and the row is the tangent of the elevation — and the box *bottom* is where the feet meet
    the floor, which is the only thing the distance estimate reads.

    ``person_height`` puts the box *top* on the head of a person that tall, which is what the
    height estimate reads; without it the box keeps a fixed fraction of the frame, which is
    what the parallax tests want.
    """
    fov_overlap = (PARALLAX_FOV - TARGET_FOV) / 2.0
    facing = TARGET_FOV * cam_id + PARALLAX_FOV / 2.0 - fov_overlap  # world angle the camera faces
    cx = RING_RADIUS * math.cos(math.radians(facing))
    cy = RING_RADIUS * math.sin(math.radians(facing))
    px = radius * math.cos(math.radians(world_azimuth))
    py = radius * math.sin(math.radians(world_azimuth))
    dx, dy = px - cx, py - cy
    distance = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dy, dx))
    theta = (bearing - facing + 180.0) % 360.0 - 180.0  # offset from camera facing
    local_angle = theta + PARALLAX_FOV / 2.0
    # Feet on the floor, CAMERA_HEIGHT below the lens: a depression angle, hence a row.
    bottom = feet_row(distance)
    width = 0.05
    center_x = local_angle / PARALLAX_FOV
    height = 0.4 if person_height is None else bottom - head_row(person_height, distance)
    return Rect(x=center_x - width / 2.0, y=bottom - height, width=width, height=height), distance


class TestOverlapBand(unittest.TestCase):
    """The band `angle_in_overlap` tests, derived at the tracked zone's far edge.

    It used to be `cam_fov - target_fov`, the band two cameras share at *infinite* distance, which
    flagged 58% of every camera's field and widened an observation's drawn tolerance 12° of azimuth
    before the person was anywhere near a seam. Deriving it at the zone's outer circle is the
    tightest choice that still never under-reports anywhere people are tracked — and under-reporting
    is the failure that matters, because it splits one person into two worlds at a seam.
    """

    def make_geometry(self, camera_diameter: float = RING_RADIUS * 2.0,
                      zone: tuple[float, float] = (3.0, 7.0)) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_camera_diameter(camera_diameter)
        g.set_zone(*zone)
        return g

    def test_the_rig_band(self) -> None:
        # Ø 0.72 ring, Ø 3 – Ø 7 zone, 127 deg fields on 90 deg sectors.
        g = self.make_geometry()
        self.assertAlmostEqual(g.overlap_band, 28.3, delta=0.05)     # local angle: the threshold
        self.assertAlmostEqual(g.overlap_world, 26.4, delta=0.05)    # azimuth: what gets drawn
        self.assertLess(g.overlap_world, g.overlap_band)             # the map compresses

    def test_no_ring_is_the_bare_field_at_any_zone(self) -> None:
        # With the cameras at the centre there is no depth question left to ask, so the band is
        # exact and the whole derivation collapses. This is why the FOV-110 fixtures above, which
        # leave `camera_diameter` at 0, are untouched by any of this.
        for zone in ((3.0, 7.0), (2.0, 100.0), (1.0, 1.5)):
            with self.subTest(zone=zone):
                g = self.make_geometry(camera_diameter=0.0, zone=zone)
                self.assertAlmostEqual(g.overlap_band, PARALLAX_FOV - TARGET_FOV, places=9)
                self.assertAlmostEqual(g.overlap_world, PARALLAX_FOV - TARGET_FOV, places=9)

    def test_it_widens_with_the_zone_and_approaches_the_bare_field(self) -> None:
        bands = [self.make_geometry(zone=(1.0, d)).overlap_band for d in (3.0, 4.5, 7.0, 40.0)]
        self.assertEqual(bands, sorted(bands))
        self.assertLess(bands[-1], PARALLAX_FOV - TARGET_FOV)        # never reaches infinity
        self.assertGreater(bands[-1], 33.0)                          # but gets close

    def test_nothing_is_shared_once_the_sectors_stop_meeting(self) -> None:
        # Below about Ø 2.0 a camera pushed 0.36 m outward no longer reaches its neighbour's
        # sector at all — the reason the zone has a floor.
        g = self.make_geometry(zone=(1.0, 2.0))
        self.assertEqual(g.overlap_band, 0.0)
        self.assertEqual(g.overlap_world, 0.0)
        self.assertFalse(g.angle_in_overlap(PARALLAX_FOV / 2.0))

    def test_an_observation_between_the_two_bands_is_no_longer_flagged(self) -> None:
        """The behaviour change, stated as the case that moved. A column 30 deg in from the field
        edge was inside the old infinite-distance band (37) and is outside the Ø 7 one (28.3), so
        the tracker no longer believes a second camera can see it — and it cannot: at Ø 7 the
        shared band really is 26.4 deg of azimuth."""
        g = self.make_geometry()
        local: float = PARALLAX_FOV - 30.0
        self.assertTrue(local > PARALLAX_FOV - (PARALLAX_FOV - TARGET_FOV))   # inside the old band
        self.assertFalse(g.angle_in_overlap(local))                           # outside the new one
        self.assertTrue(g.angle_in_overlap(PARALLAX_FOV - 20.0))              # still flagged nearer

    def test_the_zone_drives_the_distance_clamp(self) -> None:
        # The on-axis extremes, so the clamp follows the ring instead of being hand-computed for
        # one: a camera is pushed toward the circle it faces and away from the one behind it.
        g = self.make_geometry()
        self.assertAlmostEqual(g._min_distance, 1.5 - RING_RADIUS, places=9)
        self.assertAlmostEqual(g._max_distance, 3.5 + RING_RADIUS, places=9)

    def test_a_max_below_the_min_collapses_rather_than_inverting(self) -> None:
        g = self.make_geometry(zone=(3.0, 2.0))
        self.assertAlmostEqual(g._min_radius, 1.5, places=9)
        self.assertAlmostEqual(g._max_radius, 1.5, places=9)
        self.assertGreaterEqual(g._max_distance, g._min_distance)


class TestGeometryParallax(unittest.TestCase):

    def make_geometry(self, ring_radius: float = RING_RADIUS) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_camera_diameter(ring_radius * 2.0)
        g.set_camera_height(CAMERA_HEIGHT)
        g.set_zone(ZONE_MIN_DIAMETER, ZONE_MAX_DIAMETER)
        g.set_window(WINDOW, ROWS)
        return g

    def test_recovers_true_azimuth_from_both_sides_of_seam(self) -> None:
        g = self.make_geometry()
        # A person at the cam0/cam1 seam (world azimuth 90) at 2 m: both cameras
        # must report the same true azimuth once parallax is corrected.
        for cam_id in (0, 1):
            roi, _distance = synth_observation(cam_id, world_azimuth=90.0, radius=2.0)
            _local, world, _dist = g.calc_angle(roi, cam_id)
            self.assertAlmostEqual(world, 90.0, delta=0.05,
                                   msg=f"cam {cam_id} did not recover 90 deg")

    def test_recovers_true_azimuth_across_distances(self) -> None:
        g = self.make_geometry()
        for radius in (2.0, 3.0, 4.0):
            roi, _distance = synth_observation(0, world_azimuth=90.0, radius=radius)
            _local, world, _dist = g.calc_angle(roi, 0)
            self.assertAlmostEqual(world, 90.0, delta=0.05)

    def test_uncorrected_model_disagrees_at_seam(self) -> None:
        # Sanity check that the correction is actually doing something: with
        # parallax disabled the two cameras disagree by several degrees.
        g = self.make_geometry(ring_radius=0.0)
        roi0, _ = synth_observation(0, world_azimuth=90.0, radius=2.0)
        roi1, _ = synth_observation(1, world_azimuth=90.0, radius=2.0)
        _l0, world0, _d0 = g.calc_angle(roi0, 0)
        _l1, world1, _d1 = g.calc_angle(roi1, 1)
        self.assertGreater(abs(world0 - world1), 5.0)

    def test_disabled_matches_raw_model(self) -> None:
        g = self.make_geometry(ring_radius=0.0)
        roi = Rect(x=0.6, y=0.1, width=0.05, height=0.5)
        local, world, _dist = g.calc_angle(roi, 2)
        fov_overlap = (PARALLAX_FOV - TARGET_FOV) / 2.0
        expected = (TARGET_FOV * 2 + local - fov_overlap) % 360.0
        self.assertAlmostEqual(world, expected, places=6)

    def test_estimate_distance_reads_the_feet_not_the_height(self) -> None:
        # The whole point of the floor-plane model: arms up, legs pulled up and bending over all
        # change a box's height and none of them move the feet, so the estimate must not care.
        g = self.make_geometry()
        bottom = feet_row(2.0)
        base = g.estimate_distance(Rect(y=bottom - 0.4, height=0.4))
        self.assertAlmostEqual(base, 2.0, places=6)
        for height in (0.1, 0.25, 0.6):
            with self.subTest(height=height):
                self.assertAlmostEqual(g.estimate_distance(Rect(y=bottom - height, height=height)),
                                       base, places=9)

    def test_estimate_distance_matches_the_floor_geometry(self) -> None:
        g = self.make_geometry()
        for want in (1.5, 2.0, 3.0, 3.9):   # all inside the zone's derived clamp
            with self.subTest(distance=want):
                got = g.estimate_distance(Rect(y=feet_row(want) - 0.4, width=0.05, height=0.4))
                self.assertAlmostEqual(got, want, places=6)

    def test_the_bottom_row_is_the_nearest_readable_distance(self) -> None:
        # The window is pinned at the sensor's bottom reach, so feet on the last row are as close
        # as the frame can read: camera_height / tan(-elevation_bottom), 1.14 m at P800 and tilt 16.
        g = self.make_geometry()
        nearest = CAMERA_HEIGHT / math.tan(math.radians(-WINDOW.elevation_bottom))
        self.assertAlmostEqual(g.estimate_distance(Rect(y=0.6, height=0.4)), nearest, places=6)
        self.assertAlmostEqual(nearest, 1.14, delta=0.01)

    def test_estimate_distance_clamps_degenerate_box(self) -> None:
        g = self.make_geometry()
        # Feet at or above the horizon: nobody standing on this floor.
        far: float = g._max_distance          # derived from the zone, not a literal
        self.assertEqual(g.estimate_distance(Rect(y=HORIZON_ROW - 0.5, height=0.5)), far)
        self.assertEqual(g.estimate_distance(Rect(y=0.0, height=0.5)), far)
        d = g.estimate_distance(Rect(y=0.55, height=0.4))
        self.assertGreaterEqual(d, g._min_distance)
        self.assertLessEqual(d, far)

    def test_estimate_distance_uses_boxes_that_leave_the_frame(self) -> None:
        # The device extrapolates the extent of a partly-visible person and nothing clamps it on
        # the way in, so `bottom` past 1.0 is real and must not be thrown away. It is monotonic
        # across the frame edge — closer feet, closer person — and bounded by the clamp.
        g = self.make_geometry()
        near_edge = g.estimate_distance(Rect(y=0.55, height=0.44))     # bottom 0.99, in frame
        at_edge = g.estimate_distance(Rect(y=0.6, height=0.4))         # bottom 1.00, at the edge
        past_edge = g.estimate_distance(Rect(y=0.7, height=0.4))       # bottom 1.10, extrapolated
        self.assertGreater(near_edge, at_edge)
        self.assertGreaterEqual(at_edge, past_edge)
        self.assertGreaterEqual(past_edge, g._min_distance)
        # Absurd extrapolation (feet "behind" the camera) still lands inside the band.
        self.assertEqual(g.estimate_distance(Rect(y=1.5, height=0.5)), g._min_distance)

    def test_a_box_may_start_above_the_frame(self) -> None:
        # A tall person close in: head extrapolated above the top. Only the bottom is read, so
        # the estimate is unaffected by how far above 0 the box starts.
        g = self.make_geometry()
        cut = g.estimate_distance(Rect(y=-0.3, height=1.2))            # bottom 0.9
        whole = g.estimate_distance(Rect(y=0.5, height=0.4))           # bottom 0.9
        self.assertAlmostEqual(cut, whole, places=9)
        self.assertLess(cut, g._max_distance)                           # a real reading, not the clamp


class TestSeamBirths(unittest.TestCase):
    """What `seam.dead_zone` costs, on the real rig's geometry rather than a round fov.

    The zone exists so that nobody is created twice on a seam: a brand-new observation inside it
    is refused, and the person is picked up by whichever camera has them well inside its field.
    Close in, both cameras have them near an edge and neither will start them — a deliberate
    choice, since half a person tracked from one edge is worse than no person — and this is the
    test that pins where that boundary actually falls.
    """

    def setUp(self) -> None:
        self.config = PanoramicTrackerSettings(fov=PARALLAX_FOV)
        self.config.rig.camera_diameter = RING_RADIUS * 2.0
        self.config.rig.camera_height = CAMERA_HEIGHT
        self.config.seam.dead_zone = 6.5
        self.tracker = PanoramicTracker(self.config, num_players=8, num_cameras=4)
        # The fixture's own frame, as in `TestGeometryParallax`: P800 up 16 on the full reach.
        self.tracker.geometry.set_window(WINDOW, ROWS)
        self.emitted: list[TrackletDict] = []
        self.tracker.add_tracklet_callback(self.emitted.append)

    def arrive(self, radius: float) -> TrackletDict:
        """Both cameras' first sight of a person standing on the cam0/cam1 seam."""
        for cam_id in (0, 1):
            roi, _distance = synth_observation(cam_id, world_azimuth=90.0, radius=radius)
            self.tracker._add_tracklet(
                Tracklet(cam_id=cam_id, status=TrackingStatus.TRACKED, roi=roi,
                         external_id=1, external_age_in_frames=10))
        self.tracker._update_and_notify()
        return self.emitted[-1]

    def test_no_birth_on_a_seam_inside_the_dead_zone(self) -> None:
        # Ø 2.7, the inner edge of the play zone: 5.4 deg from both field edges, inside 6.5.
        self.assertEqual(self.arrive(1.35), {})

    def test_one_world_born_once_outside_it(self) -> None:
        # Ø 3.5: 8.8 deg from both edges. The first camera starts the person and the second
        # links into the same world rather than creating a second one.
        out = self.arrive(1.75)
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 0)
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)


class TestGeometryHeight(unittest.TestCase):
    """A person's height is a pure pixel ratio on cylindrical rows: camera_height times the box
    height over the rows from the horizon down to the feet. No focal, no field, no distance."""

    def make_geometry(self) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_camera_height(CAMERA_HEIGHT)
        g.set_zone(ZONE_MIN_DIAMETER, ZONE_MAX_DIAMETER)
        g.set_window(WINDOW, ROWS)
        return g

    def box(self, height_m: float, distance: float) -> Rect:
        bottom: float = feet_row(distance)
        return Rect(x=0.5, y=head_row(height_m, distance), width=0.05,
                    height=bottom - head_row(height_m, distance))

    def test_recovers_the_height_at_every_distance(self) -> None:
        g = self.make_geometry()
        for want in (1.5, 1.8, 2.0):
            for distance in (1.5, 2.0, 3.0, 5.0, 7.0):
                with self.subTest(height=want, distance=distance):
                    self.assertAlmostEqual(g.estimate_height(self.box(want, distance)), want, places=6)

    def test_is_independent_of_distance(self) -> None:
        # The pixel box shrinks with distance; the metres must not move. This is what a gate on
        # pixel box heights could not do, and why `seam.link_height` is a percentage of these.
        g = self.make_geometry()
        near: Rect = self.box(1.8, 2.0)
        far: Rect = self.box(1.8, 6.0)
        self.assertGreater(near.height, far.height * 2.0)           # the pixels differ a lot
        self.assertAlmostEqual(g.estimate_height(near), g.estimate_height(far), places=6)

    def test_reads_reach_when_the_arms_go_up(self) -> None:
        # The box top is the highest pixel, so this is overhead reach, not stature — the number
        # the tilt table is chosen around.
        g = self.make_geometry()
        for distance in (2.0, 4.0):
            with self.subTest(distance=distance):
                self.assertAlmostEqual(g.estimate_height(self.box(1.8, distance)), 1.8, places=6)
                self.assertAlmostEqual(g.estimate_height(self.box(2.2, distance)), 2.2, places=6)

    def test_zero_when_the_feet_are_not_on_this_floor(self) -> None:
        g = self.make_geometry()
        self.assertEqual(g.estimate_height(Rect(y=0.0, height=HORIZON_ROW)), 0.0)   # feet on the horizon
        self.assertEqual(g.estimate_height(Rect(y=0.0, height=0.2)), 0.0)           # and above it

    def test_capped_for_a_box_that_barely_clears_the_horizon(self) -> None:
        # Feet a pixel below the horizon divide by almost nothing; the cap is what bounds it, and
        # it sits above any height a person can measure so it only ever catches a mangled box.
        g = self.make_geometry()
        one_px: float = 1.0 / (ROWS - 1)
        self.assertEqual(g.estimate_height(Rect(y=0.1, height=HORIZON_ROW - 0.1 + one_px)), _MAX_HEIGHT)
        self.assertGreater(_MAX_HEIGHT, 2.4)                # a tall person with both arms up

    def test_the_feet_on_the_bottom_row_still_measure(self) -> None:
        # The nearest the frame can read, which is where the distance estimate saturates too.
        g = self.make_geometry()
        nearest: float = CAMERA_HEIGHT / math.tan(math.radians(-WINDOW.elevation_bottom))
        self.assertAlmostEqual(g.estimate_height(self.box(1.8, nearest)), 1.8, places=6)

    def test_two_cameras_at_a_seam_agree_in_metres(self) -> None:
        """The point of measuring in metres. A person in an overlap but off the seam centre is at
        genuinely different distances from the two cameras, so their pixel box heights differ by
        up to 0.16 of the frame — which a gate on frame fractions could not tell from two
        different people — while the measured heights agree exactly."""
        g = self.make_geometry()
        for azimuth in (80.0, 75.0, 72.0):
            for radius in (1.35, 2.25, 3.5):
                with self.subTest(azimuth=azimuth, radius=radius):
                    roi0, d0 = synth_observation(0, azimuth, radius, person_height=1.8)
                    roi1, d1 = synth_observation(1, azimuth, radius, person_height=1.8)
                    self.assertNotAlmostEqual(d0, d1, places=2)          # different distances
                    self.assertAlmostEqual(g.estimate_height(roi0), 1.8, places=6)
                    self.assertAlmostEqual(g.estimate_height(roi1), 1.8, places=6)

    def test_the_annotation_carries_it(self) -> None:
        g = self.make_geometry()
        roi, _distance = synth_observation(0, 45.0, 2.25, person_height=1.8)
        *_rest, distance, height = g.get_angles_and_overlap(roi, 0)
        self.assertAlmostEqual(height, 1.8, places=6)
        self.assertGreater(distance, 0.6)


class TestInitialGeometrySync(unittest.TestCase):

    def test_config_applied_to_geometry_at_construction(self) -> None:
        # bind() does not fire with the initial value and presets load before
        # the tracker exists, so construction must push config into geometry.
        config = PanoramicTrackerSettings(fov=PARALLAX_FOV)
        config.rig.camera_diameter = RING_RADIUS * 2.0
        config.rig.camera_height = CAMERA_HEIGHT
        tracker = PanoramicTracker(config, num_players=4, num_cameras=4)
        self.assertEqual(tracker.geometry.cam_fov, PARALLAX_FOV)
        self.assertEqual(tracker.geometry._ring_radius, RING_RADIUS)
        self.assertEqual(tracker.geometry._camera_height, CAMERA_HEIGHT)
        # Each camera owns 360/num_cameras of the ring, not a hardcoded 90
        self.assertAlmostEqual(tracker.geometry.target_fov, 90.0, places=9)
        self.assertAlmostEqual(
            PanoramicTracker(config, num_players=4, num_cameras=3).geometry.target_fov,
            120.0, places=9)
        # The row model is derived from the shared camera fields with the warp's own functions —
        # frame_height 0 resolving to the sensor's full reach at this tilt — and published as
        # read-only fields for the panorama.
        rows = delivered_height(False, config.resolution, PARALLAX_FOV, config.tilt,
                                config.lens_fov, (config.lens_centre_x, config.lens_centre_y),
                                config.frame_height)
        window = frame_window((1280, 800), (1280, rows), 1280, PARALLAX_FOV, config.tilt,
                              config.lens_fov, (config.lens_centre_x, config.lens_centre_y))
        self.assertAlmostEqual(tracker.geometry._horizon_px, window.horizon_px, places=9)
        self.assertAlmostEqual(tracker.geometry._focal, window.focal, places=9)
        self.assertEqual(tracker.geometry._rows, rows)
        self.assertAlmostEqual(config.rig.horizon_row, window.horizon_px / (rows - 1), places=9)
        self.assertAlmostEqual(config.rig.focal_rows, window.focal / (rows - 1), places=9)
        self.assertAlmostEqual(config.rig.vfov, window.elevation_top - window.elevation_bottom, places=9)
        self.assertAlmostEqual(config.rig.elevation_bottom, window.elevation_bottom, places=9)
        self.assertAlmostEqual(config.rig.elevation_top, window.elevation_top, places=9)


if __name__ == "__main__":
    unittest.main()
