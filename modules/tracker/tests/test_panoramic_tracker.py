"""Tests for the panoramic tracker: seam hysteresis, dead-zone handling,
cross-camera linking, same-camera re-acquisition, device id reuse, emission of lost worlds,
world id reuse, and ring parallax correction."""

import math
import time
import unittest
from dataclasses import replace

from modules.oak import CameraResolution, delivered_height, frame_coverage, frame_window
from modules.tracker import (
    PanoramicTracker, PanoramicTrackerSettings, PanoramicAnnotation, Rejection,
    Tracklet, TrackingStatus, TrackletDict, camera_local_to_azimuth, row_from_elevation, row_model,
)
from modules.tracker.panoramic.geometry import Geometry, _MAX_HEIGHT, height_is_measured
from modules.tracker.panoramic.tracker import HANDS_HEIGHT
from modules.tracker.panoramic.store import TrackletIdPool
from modules.utils import Rect


# fov 110 / target 90 -> `fov_overlap` 10 deg of offset, and a 20 deg band of each camera's field
# that its neighbour also sees. With the default seam settings (dead_zone 5, link_angle 8,
# link_height 0.15, hysteresis 0.9): no births at local angle <= 5 or >= 105; a second opinion
# exists at <= 20 or >= 90; two cameras' observations are one person within 8 deg of world
# azimuth and 0.15 of the larger measured height.
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
        # These tests are about identity — hysteresis, linking, id reuse — not about where on the
        # floor anyone stands, and `make_tracklet`'s boxes put the feet wherever their `top` and
        # `height` land. Open the far edge as wide as it goes so the floor position never decides
        # them; the far-edge filter has its own tests (`TestFarEdge`). With no ring, the zone moves
        # nothing else here: the overlap band is the bare field at any zone.
        self.config.rig.zone_max_radius = 15.0
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
        horizon_row, focal_rows = row_model(p.angle_bottom, p.angle_top)
        lift: float = focal_rows * feet_off_floor / distance
        bottom: float = horizon_row + focal_rows * p.camera_height / distance - lift
        box_h: float = focal_rows * height_m / distance
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


class TestEmitsWhatItRemembers(PanoramicTrackerCase):
    """The tracker emits every world it still remembers, stale or not, until `lost_timeout`: how old
    a box may be before it stops counting is the consumer's call (pose's `detection_timeout`)."""

    def test_a_stale_world_is_still_emitted_and_keeps_anchoring(self) -> None:
        stale = replace(make_tracklet(0, 1, 50.0), last_active=time.time() - 1.0)
        out = self.submit(stale)
        self.assertEqual(set(out.keys()), {0})                         # 1.0 s < lost_timeout 2.0
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 0)

    def test_a_forgotten_world_is_not_emitted(self) -> None:
        # Last seen 1.5 s ago: remembered at 2.0. Lowering `lost_timeout` below that stands in for
        # time passing; the tick that retires the world must not emit it one last time.
        out = self.submit(replace(make_tracklet(0, 1, 50.0), last_active=time.time() - 1.5))
        self.assertEqual(set(out.keys()), {0})
        self.config.lost_timeout = 1.2
        self.tracker._update_and_notify()
        self.assertEqual(self.emitted[-1], {})

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
        # Same azimuth, 1.2 m against 1.8 m: 0.33 apart, past the 0.15 gate.
        self.submit(self.person(0, 1, 98.0, height_m=1.8, distance=2.5))
        out = self.submit(self.person(1, 1, 8.0, height_m=1.2, distance=2.5))
        self.assertEqual(set(out.keys()), {0, 1})  # link refused -> new world

    def test_the_height_gate_is_a_fraction_of_the_larger_height(self) -> None:
        # Normalised like `height_filter`: 0.15 means 15 % of the larger height. Either side of it,
        # so a gate read as a percentage (0.15 %) or scaled twice would fail one of the two.
        self.config.seam.link_height = 0.15
        self.assertTrue(self.tracker._heights_match(1.8, 1.8 * 0.86))     # 0.14 apart: one person
        self.assertFalse(self.tracker._heights_match(1.8, 1.8 * 0.84))    # 0.16 apart: two

    def test_a_jumper_links_on_azimuth_alone(self) -> None:
        # Feet off the floor breaks the one assumption the height measurement rests on, so the
        # reading is not a measurement and must not be allowed to veto a link the azimuth
        # supports — a person is never harder to re-find than mid-jump. A 0.3 m jump: enough to
        # wreck the height, while the feet still read inside this fixture's open far edge.
        self.submit(self.person(0, 1, 98.0, height_m=1.8, distance=2.5))
        jumper = self.person(1, 1, 8.0, height_m=1.8, distance=2.5, feet_off_floor=0.3)
        out = self.submit(jumper)
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)
        annotation = self.tracker.store.get_live_observation(1, 1)
        assert annotation is not None and isinstance(annotation.annotation, PanoramicAnnotation)
        # It really is a bad reading, not a lucky one that happened to fall inside the gate.
        self.assertFalse(height_is_measured(annotation.annotation.height)
                         and abs(annotation.annotation.height - 1.8) < 0.15 * 1.8)

    def test_a_view_whose_feet_read_past_the_far_edge_links_once_they_land(self) -> None:
        """The far-edge filter's trade at a seam. Feet 0.45 m up, with the lens at 0.5 m, sit just
        below eye level and read about 21 m out — past even this fixture's open edge — so the second
        camera's view is not counted mid-jump. Nobody is lost: the first camera still carries the
        person, and the view links the moment the feet are back on the floor."""
        self.submit(self.person(0, 1, 98.0, height_m=1.8, distance=2.5))
        out = self.submit(self.person(1, 1, 8.0, height_m=1.8, distance=2.5, feet_off_floor=0.45))
        self.assertEqual(set(out.keys()), {0})                         # still emitted, from cam 0
        self.assertIsNone(self.tracker.store.get_world_id(1, 1))       # the view is not seen
        self.submit(self.person(1, 1, 8.0, height_m=1.8, distance=2.5))
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)     # landed: linked

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
# The tracked zone, deliberately WIDER than the studio preset's R 1.5 – R 3.5. It was chosen when
# the distance was clamped to the zone, to keep that clamp clear of the frame's own nearest
# readable row; the clamp is gone, and the geometry tests below only ever read distances, so the
# zone now matters to them only through the overlap band and the parallax depth.
ZONE_MIN_RADIUS = 1.0
ZONE_MAX_RADIUS = 4.0
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

    def make_geometry(self, camera_radius: float = RING_RADIUS,
                      zone: tuple[float, float] = (1.5, 3.5)) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_camera_radius(camera_radius)
        g.set_zone(*zone)
        return g

    def test_the_rig_band(self) -> None:
        # R 0.36 ring, R 1.5 – R 3.5 zone, 127 deg fields on 90 deg sectors.
        g = self.make_geometry()
        self.assertAlmostEqual(g.overlap_band, 28.3, delta=0.05)     # local angle: the threshold
        # The same threshold in azimuth, at the depth the marks are drawn on — which is what the
        # panorama's overlap lines use, so a mark's tolerance changes width exactly on the line.
        self.assertAlmostEqual(g.overlap_azimuth, 31.0, delta=0.05)
        # Wider than the local band, because a nearer depth pulls a bearing toward the camera's
        # own axis and so away from the seam the band is measured from.
        self.assertGreater(g.overlap_azimuth, g.overlap_band)

    def test_no_ring_is_the_bare_field_at_any_zone(self) -> None:
        # With the cameras at the centre there is no depth question left to ask, so the band is
        # exact and the whole derivation collapses. This is why the FOV-110 fixtures above, which
        # leave `camera_radius` at 0, are untouched by any of this.
        for zone in ((1.5, 3.5), (1.0, 50.0), (0.5, 0.75)):
            with self.subTest(zone=zone):
                g = self.make_geometry(camera_radius=0.0, zone=zone)
                self.assertAlmostEqual(g.overlap_band, PARALLAX_FOV - TARGET_FOV, places=9)
                self.assertAlmostEqual(g.overlap_azimuth, PARALLAX_FOV - TARGET_FOV, places=9)

    def test_it_widens_with_the_zone_and_approaches_the_bare_field(self) -> None:
        bands = [self.make_geometry(zone=(0.5, r)).overlap_band for r in (1.5, 2.25, 3.5, 20.0)]
        self.assertEqual(bands, sorted(bands))
        self.assertLess(bands[-1], PARALLAX_FOV - TARGET_FOV)        # never reaches infinity
        self.assertGreater(bands[-1], 33.0)                          # but gets close

    def test_nothing_is_shared_once_the_sectors_stop_meeting(self) -> None:
        # Below about R 1.0 a camera pushed 0.36 m outward no longer reaches its neighbour's
        # sector at all — the reason the zone has a floor.
        g = self.make_geometry(zone=(0.5, 1.0))
        self.assertEqual(g.overlap_band, 0.0)
        self.assertEqual(g.overlap_azimuth, 0.0)
        self.assertFalse(g.angle_in_overlap(PARALLAX_FOV / 2.0))

    def test_an_observation_between_the_two_bands_is_no_longer_flagged(self) -> None:
        """The behaviour change, stated as the case that moved. A column 30 deg in from the field
        edge was inside the old infinite-distance band (37) and is outside the R 3.5 one (28.3), so
        the tracker no longer believes a second camera can see it — and it cannot: at R 3.5 the
        shared band really is 26.4 deg of azimuth."""
        g = self.make_geometry()
        local: float = PARALLAX_FOV - 30.0
        self.assertTrue(local > PARALLAX_FOV - (PARALLAX_FOV - TARGET_FOV))   # inside the old band
        self.assertFalse(g.angle_in_overlap(local))                           # outside the new one
        self.assertTrue(g.angle_in_overlap(PARALLAX_FOV - 20.0))              # still flagged nearer

    def test_the_drawn_line_is_exactly_where_the_flag_flips(self) -> None:
        """What `overlap_azimuth` exists for. The panorama draws two verticals per seam at
        ±`overlap_azimuth`/2 and switches a mark's tolerance width on `angle_in_overlap`. Those are
        the same threshold, but one is a local angle and the other a strip position, and a local
        angle has no single position on the ring — project it at the wrong depth and the line sits
        where nothing happens. Pinned here because the two live in different files."""
        g = self.make_geometry()
        line: float = g.target_fov - g.overlap_azimuth / 2.0
        # The last local angle still inside the flag, carried to the strip the way a mark is.
        flips: float = camera_local_to_azimuth(g.cam_fov - g.overlap_band, 0, g.cam_fov,
                                               g.target_fov, g._ring_radius, g.parallax_radius)
        self.assertAlmostEqual(line, flips, places=9)
        # And it is genuinely a different place from the far-edge projection it used to be.
        far_edge: float = camera_local_to_azimuth(g.cam_fov - g.overlap_band, 0, g.cam_fov,
                                                  g.target_fov, g._ring_radius,
                                                  g._max_radius)
        self.assertGreater(abs(far_edge - flips), 1.0)

    def test_a_max_below_the_min_collapses_rather_than_inverting(self) -> None:
        g = self.make_geometry(zone=(1.5, 1.0))
        self.assertAlmostEqual(g._min_radius, 1.5, places=9)
        self.assertAlmostEqual(g._max_radius, 1.5, places=9)


class TestGeometryParallax(unittest.TestCase):

    def make_geometry(self, ring_radius: float = RING_RADIUS,
                      zone: tuple[float, float] = (ZONE_MIN_RADIUS, ZONE_MAX_RADIUS)) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_camera_radius(ring_radius)
        g.set_camera_height(CAMERA_HEIGHT)
        g.set_zone(*zone)
        g.set_window(WINDOW, ROWS)
        return g

    def test_recovers_true_azimuth_on_the_corrected_cylinder(self) -> None:
        # The correction assumes ONE depth — `parallax_radius`, derived from the zone — so that
        # is where it is exact. A person standing on it is recovered to the degree from either
        # side of the seam, which is the property the whole fusion rests on.
        g = self.make_geometry()
        radius: float = g.parallax_radius
        for cam_id in (0, 1):
            roi, _distance = synth_observation(cam_id, world_azimuth=90.0, radius=radius)
            _local, world, _dist = g.calc_angle(roi, cam_id)
            self.assertAlmostEqual(world, 90.0, delta=0.05,
                                   msg=f"cam {cam_id} did not recover 90 deg on the cylinder")

    def test_off_the_cylinder_the_error_is_bounded_and_symmetric(self) -> None:
        """Away from the assumed depth each camera errs, and — because they see the person on
        opposite sides of their own axes — in opposite directions, so the *seam disagreement* is
        twice one camera's error. It has to stay inside `link_angle` everywhere in the zone, or a
        crossing splits. Bounded, and zero on the cylinder.

        On the STUDIO zone (R 1.5 – R 3.5), because that is the configuration whose bound is quoted
        in CALIBRATION.md — the surrounding fixture uses a wider R 1 – R 4, and a wider zone
        necessarily has a worse worst case (14.4° at R 1 – R 4, which is the honest cost of
        claiming that much floor)."""
        g = self.make_geometry(zone=(1.5, 3.5))
        self.assertAlmostEqual(g.parallax_radius, 2.1, places=9)
        worst: float = 0.0
        for radius in (1.5, 2.0, g.parallax_radius, 3.0, 3.5):
            with self.subTest(radius=radius):
                reported = []
                for cam_id in (0, 1):
                    roi, _d = synth_observation(cam_id, world_azimuth=90.0, radius=radius)
                    reported.append(g.calc_angle(roi, cam_id)[1])
                gap: float = abs(reported[1] - reported[0])
                worst = max(worst, gap)
                # Each camera is off by half the gap, and they straddle the truth.
                self.assertAlmostEqual((reported[0] + reported[1]) / 2.0, 90.0, delta=0.1)
                if abs(radius - g.parallax_radius) < 1e-6:
                    self.assertLess(gap, 0.1)       # exact on the cylinder
        self.assertLess(worst, 6.8, f'seam disagreement {worst:.2f} deg over R 1.5 – R 3.5')

    def test_the_azimuth_ignores_the_box_bottom(self) -> None:
        """The point of taking the measured distance out. The device's box bottom sits below the
        feet, which is why `estimate_distance` reads short — and while that number fed the parallax
        triangle, the bias moved every person's *bearing*, in opposite directions at a seam. Now
        the same person with the box bottom dragged anywhere reports the same azimuth."""
        g = self.make_geometry()
        for cam_id in (0, 1):
            roi, _d = synth_observation(cam_id, world_azimuth=90.0, radius=2.0)
            angles, distances = [], []
            for delta in (0.0, 0.05, 0.15, 0.4):        # box bottom dragged down the frame
                dragged = replace(roi, height=roi.height + delta)
                _local, world, distance = g.calc_angle(dragged, cam_id)
                angles.append(world)
                distances.append(distance)
            self.assertAlmostEqual(max(angles), min(angles), places=9,
                                   msg=f'cam {cam_id} azimuth still moves with the box bottom')
            # ... while the reported metres do move, which is what makes this a real test.
            self.assertGreater(max(distances) - min(distances), 0.2)

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
        for want in (1.5, 2.0, 3.0, 3.9):
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

    def test_feet_not_on_this_floor_read_infinitely_far(self) -> None:
        g = self.make_geometry()
        # Feet at or above the horizon: nobody standing on this floor.
        self.assertEqual(g.estimate_distance(Rect(y=HORIZON_ROW - 0.5, height=0.5)), math.inf)
        self.assertEqual(g.estimate_distance(Rect(y=0.0, height=0.5)), math.inf)

    def test_estimate_distance_is_not_clamped_to_the_zone(self) -> None:
        # The point of removing the clamp: someone well past the far edge reads where they are, so
        # the far-edge filter can see them and the panorama can show them.
        g = self.make_geometry()
        for want in (5.0, 8.0, 12.0):
            with self.subTest(distance=want):
                got = g.estimate_distance(Rect(y=feet_row(want) - 0.2, width=0.05, height=0.2))
                self.assertAlmostEqual(got, want, places=6)

    def test_estimate_distance_uses_boxes_that_leave_the_frame(self) -> None:
        # The device extrapolates the extent of a partly-visible person and nothing clamps it on
        # the way in, so `bottom` past 1.0 is kept, not thrown away. It is monotonic across the frame
        # edge — closer feet, closer person — and stays a positive distance however far it runs.
        g = self.make_geometry()
        near_edge = g.estimate_distance(Rect(y=0.55, height=0.44))     # bottom 0.99, in frame
        at_edge = g.estimate_distance(Rect(y=0.6, height=0.4))         # bottom 1.00, at the edge
        past_edge = g.estimate_distance(Rect(y=0.7, height=0.4))       # bottom 1.10, extrapolated
        absurd = g.estimate_distance(Rect(y=1.5, height=0.5))          # bottom 2.00
        self.assertGreater(near_edge, at_edge)
        self.assertGreater(at_edge, past_edge)
        self.assertGreater(past_edge, absurd)
        self.assertGreater(absurd, 0.0)

    def test_a_box_may_start_above_the_frame(self) -> None:
        # A tall person close in: head extrapolated above the top. Only the bottom is read, so
        # the estimate is unaffected by how far above 0 the box starts.
        g = self.make_geometry()
        cut = g.estimate_distance(Rect(y=-0.3, height=1.2))            # bottom 0.9
        whole = g.estimate_distance(Rect(y=0.5, height=0.4))           # bottom 0.9
        self.assertAlmostEqual(cut, whole, places=9)
        self.assertTrue(math.isfinite(cut))                            # a real reading


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
        self.config.rig.camera_radius = RING_RADIUS
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
        # R 1.35, the inner edge of the play zone: 5.4 deg from both field edges, inside 6.5.
        self.assertEqual(self.arrive(1.35), {})

    def test_one_world_born_once_outside_it(self) -> None:
        # R 1.75: 8.8 deg from both edges. The first camera starts the person and the second
        # links into the same world rather than creating a second one.
        out = self.arrive(1.75)
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 0)
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)


class RigTrackerCase(unittest.TestCase):
    """A tracker on the rig's geometry (R 0.36 ring, R 1.5 – R 3.5 zone) with the fixture's own
    frame, so a synthesised box puts the feet where the person really stands. No tests of its own."""

    AXIS: float = 45.0            # camera 0's optical axis, in world azimuth

    def setUp(self) -> None:
        self.config = PanoramicTrackerSettings(fov=PARALLAX_FOV)
        self.config.rig.camera_radius = RING_RADIUS
        self.config.rig.camera_height = CAMERA_HEIGHT
        self.config.lost_timeout = 2.0
        self.tracker = PanoramicTracker(self.config, num_players=8, num_cameras=4)
        self.tracker.geometry.set_window(WINDOW, ROWS)
        self.emitted: list[TrackletDict] = []
        self.observed: list[list[Tracklet]] = []
        self.tracker.add_tracklet_callback(self.emitted.append)
        self.tracker.add_observation_callback(self.observed.append)

    def seen(self, radius: float, ext_id: int = 1, seconds_ago: float = 0.0,
             status: TrackingStatus = TrackingStatus.TRACKED, age: int = 10,
             height: float | None = None) -> TrackletDict:
        """Camera 0 sees someone on its own axis at `radius`, last detected `seconds_ago`. `height`
        overrides the box's height (frame fraction), keeping its bottom on the feet."""
        roi, _distance = synth_observation(0, world_azimuth=self.AXIS, radius=radius)
        if height is not None:
            roi = replace(roi, y=roi.y + roi.height - height, height=height)
        self.tracker._add_tracklet(Tracklet(cam_id=0, status=status, roi=roi, external_id=ext_id,
                                            external_age_in_frames=age,
                                            last_active=time.time() - seconds_ago))
        self.tracker._update_and_notify()
        return self.emitted[-1]


class TestFarEdge(RigTrackerCase):
    """Past `zone_max_radius` a person is not seen: handled exactly like a missed detection."""

    def test_the_edge_is_a_radius_not_a_camera_distance(self) -> None:
        """A camera sits `ring_radius` out toward the person, so it reads them nearer than their
        radius: on its axis at R 3.6 it reads 3.24 m, which a camera-distance test against 3.5
        would wrongly accept."""
        g = self.tracker.geometry
        on_axis: float = PARALLAX_FOV / 2.0
        self.assertTrue(g.beyond_zone(on_axis, 3.6 - RING_RADIUS))
        self.assertFalse(g.beyond_zone(on_axis, 3.4 - RING_RADIUS))
        self.assertTrue(g.beyond_zone(on_axis, math.inf))            # feet not on this floor

    def test_a_new_arrival_past_the_edge_is_not_started(self) -> None:
        self.assertEqual(self.seen(4.0), {})
        self.assertIsNone(self.tracker.store.get_world_id(0, 1))
        self.assertEqual(set(self.seen(3.0).keys()), {0})             # inside: started

    def test_a_brief_excursion_changes_nothing_visible(self) -> None:
        # A jump, or feet hidden for a moment: LOST, but still emitted (and posed, inside pose's
        # `detection_timeout`).
        self.seen(3.0)
        out = self.seen(4.0)
        self.assertEqual(set(out.keys()), {0})
        observation = self.tracker.store.get_live_observation(0, 1)
        assert observation is not None
        self.assertEqual(observation.status, TrackingStatus.LOST)

    def test_walking_out_does_not_restart_the_clock(self) -> None:
        """The trap the store's `latest` argument exists for: the observation follows the person
        out (so the panorama does too), but its `last_active` stays where they were last inside."""
        self.seen(3.0, seconds_ago=1.5)
        before = self.tracker.store.get_live_observation(0, 1)
        assert before is not None
        out = self.seen(4.0)
        after = self.tracker.store.get_live_observation(0, 1)
        assert after is not None and isinstance(after.annotation, PanoramicAnnotation)
        self.assertEqual(after.last_active, before.last_active)
        self.assertAlmostEqual(after.annotation.distance, 4.0 - RING_RADIUS, delta=0.01)
        self.assertEqual(set(out.keys()), {0})                         # remembered: 1.5 s < lost_timeout

    def test_someone_who_stays_out_is_forgotten(self) -> None:
        # Last inside 1.5 s ago: alive at that tick (< lost_timeout 2.0). Lowering the timeout
        # below 1.5 before the outside frame stands in for time passing. Without the filter that
        # frame would refresh `last_active` and keep them; with it, the old clock retires them.
        self.seen(3.0, seconds_ago=1.5)
        self.assertEqual(self.tracker.store.all_world_ids(), [0])
        self.config.lost_timeout = 1.2
        self.seen(4.0)
        self.assertEqual(self.tracker.store.all_world_ids(), [])

    def test_stepping_back_in_keeps_the_same_person(self) -> None:
        self.seen(3.0)
        self.seen(4.0)
        out = self.seen(3.2)
        self.assertEqual(set(out.keys()), {0})
        observation = self.tracker.store.get_live_observation(0, 1)
        assert observation is not None
        self.assertEqual(observation.status, TrackingStatus.TRACKED)
        self.assertEqual(self.tracker.store.get_world_id(0, 1), 0)

    def test_past_the_edge_blocks_reacquisition(self) -> None:
        # The device drops the person and re-finds them under a new id, but past the edge: not them
        # yet. Back inside, the same world picks them up.
        self.seen(3.0)
        self.seen(3.0, status=TrackingStatus.REMOVED)
        self.seen(4.0, ext_id=2)
        self.assertIsNone(self.tracker.store.get_world_id(0, 2))
        self.seen(3.0, ext_id=3)
        self.assertEqual(self.tracker.store.get_world_id(0, 3), 0)


class TestFilteredDetections(RigTrackerCase):
    """Every detection a filter drops is published on the observation channel, tagged with why —
    so the panorama can draw it — and never on the primary channel the show reads."""

    def rejected(self) -> dict[Rejection, int]:
        """The reasons on the latest observation channel, counted."""
        counts: dict[Rejection, int] = {}
        for t in self.observed[-1]:
            assert isinstance(t.annotation, PanoramicAnnotation)
            if t.annotation.rejected is not None and t.id < 0:
                counts[t.annotation.rejected] = counts.get(t.annotation.rejected, 0) + 1
        return counts

    def test_each_filter_is_tagged_and_kept_off_the_primary_channel(self) -> None:
        cases = {
            Rejection.YOUNG: dict(radius=3.0, age=self.config.age_filter),
            Rejection.SMALL: dict(radius=3.0, height=self.config.height_filter / 2.0),
            Rejection.PAST_EDGE: dict(radius=4.0),
        }
        for reason, kwargs in cases.items():
            with self.subTest(reason=reason):
                self.setUp()
                out = self.seen(**kwargs)
                self.assertEqual(out, {})                                  # the show sees nobody
                self.assertEqual(self.rejected(), {reason: 1})             # the strip sees why
                self.assertEqual(self.tracker.store.all_world_ids(), [])

    def test_a_new_arrival_in_the_dead_zone_is_tagged(self) -> None:
        # Both cameras' first sight of someone on a seam close in: both inside the dead zone.
        self.config.seam.dead_zone = 6.5
        for cam_id in (0, 1):
            roi, _d = synth_observation(cam_id, world_azimuth=90.0, radius=1.35)
            self.tracker._add_tracklet(Tracklet(cam_id=cam_id, status=TrackingStatus.TRACKED, roi=roi,
                                                external_id=1, external_age_in_frames=10))
        self.tracker._update_and_notify()
        self.assertEqual(self.emitted[-1], {})
        self.assertEqual(self.rejected(), {Rejection.DEAD_ZONE: 2})

    def test_it_leaves_the_channel_once_accepted_or_ended(self) -> None:
        self.seen(3.0, age=self.config.age_filter)              # young
        self.assertEqual(self.rejected(), {Rejection.YOUNG: 1})
        self.seen(3.0)                                           # old enough: a person
        self.assertEqual(self.rejected(), {})
        self.assertEqual(self.tracker.store.all_world_ids(), [0])
        self.seen(4.0, ext_id=2)                                 # someone else, past the edge
        self.assertEqual(self.rejected(), {Rejection.PAST_EDGE: 1})
        self.seen(4.0, ext_id=2, status=TrackingStatus.REMOVED)  # the device lets them go
        self.assertEqual(self.rejected(), {})

    def test_a_camera_that_stops_reporting_does_not_leave_it_behind(self) -> None:
        # Last reported 1.5 s ago: still kept at `lost_timeout` 2.0. Lowering it below that stands in
        # for time passing with no further report from the camera — no LOST, no REMOVED.
        self.seen(4.0, seconds_ago=1.5)
        self.assertEqual(self.rejected(), {Rejection.PAST_EDGE: 1})
        self.config.lost_timeout = 1.2
        self.tracker._update_and_notify()
        self.assertEqual(self.rejected(), {})

    def test_off_means_off(self) -> None:
        self.config.zone_filter = False
        out = self.seen(4.0)
        self.assertEqual(set(out.keys()), {0})                   # tracked like anyone
        self.assertEqual(self.rejected(), {})
        for t in self.observed[-1]:
            assert isinstance(t.annotation, PanoramicAnnotation)
            self.assertIsNone(t.annotation.rejected)

    def test_a_tracked_person_past_the_edge_is_tagged_on_their_own_mark(self) -> None:
        self.seen(3.0)
        self.seen(4.0)
        observation = self.tracker.store.get_live_observation(0, 1)
        assert observation is not None and isinstance(observation.annotation, PanoramicAnnotation)
        self.assertEqual(observation.annotation.rejected, Rejection.PAST_EDGE)
        self.assertEqual(self.rejected(), {})                    # one mark, not a second grey line
        self.seen(3.0)
        observation = self.tracker.store.get_live_observation(0, 1)
        assert observation is not None and isinstance(observation.annotation, PanoramicAnnotation)
        self.assertIsNone(observation.annotation.rejected)       # back inside: no tag


class TestGeometryHeight(unittest.TestCase):
    """A person's height is a pure pixel ratio on cylindrical rows: camera_height times the box
    height over the rows from the horizon down to the feet. No focal, no field, no distance."""

    def make_geometry(self) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_camera_height(CAMERA_HEIGHT)
        g.set_zone(ZONE_MIN_RADIUS, ZONE_MAX_RADIUS)
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
        # pixel box heights could not do, and why `seam.link_height` is a fraction of these.
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


class TestFootOffset(unittest.TestCase):
    """The detector's box bottom sits below the feet by a fixed pad in pixels. `foot_offset`
    subtracts it, in one place (`_foot_px`), shared by the distance and the height."""

    PAD: float = 94.0 / (ROWS - 1)          # the measured studio bias, as a frame fraction

    def make_geometry(self, foot_offset: float = 0.0) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_camera_radius(RING_RADIUS)
        g.set_camera_height(CAMERA_HEIGHT)
        g.set_zone(ZONE_MIN_RADIUS, ZONE_MAX_RADIUS)
        g.set_window(WINDOW, ROWS)
        g.set_foot_offset(foot_offset)
        return g

    def true_box(self, height_m: float, distance: float) -> Rect:
        """The box a perfect detector would report: bottom exactly on the feet."""
        top: float = head_row(height_m, distance)
        return Rect(x=0.5, y=top, width=0.05, height=feet_row(distance) - top)

    def padded_box(self, height_m: float, distance: float) -> Rect:
        """...and what this detector actually reports: the same box, bottom dragged down."""
        box: Rect = self.true_box(height_m, distance)
        return replace(box, height=box.height + self.PAD)

    def test_zero_offset_is_the_identity(self) -> None:
        # The default must change nothing, which is what lets every other test in this file stand.
        plain = self.make_geometry()
        for distance in (1.5, 2.0, 3.0):
            box: Rect = self.true_box(1.8, distance)
            with self.subTest(distance=distance):
                self.assertAlmostEqual(plain.estimate_distance(box), distance, places=6)
                self.assertAlmostEqual(plain.estimate_height(box), 1.8, places=6)

    def test_a_matching_offset_recovers_the_distance_and_the_height(self) -> None:
        # The point of sharing `_foot_px`: one number fixes both readouts at once, at every
        # distance, because it corrects the thing they have in common.
        g = self.make_geometry(self.PAD)
        for distance in (1.5, 2.0, 3.0, 3.8):
            with self.subTest(distance=distance):
                box: Rect = self.padded_box(1.8, distance)
                self.assertAlmostEqual(g.estimate_distance(box), distance, places=6)
                self.assertAlmostEqual(g.estimate_height(box), 1.8, places=6)

    def test_the_uncorrected_signature(self) -> None:
        """Pins the table in CALIBRATION: with the pad present and the offset at 0, both readouts
        are short AND the height *falls* as the person walks away. That drift is the whole
        calibration signal — a proportional error would leave `H` flat but wrong instead."""
        g = self.make_geometry()
        heights: list[float] = []
        for distance in (1.5, 2.5, 3.8):
            box: Rect = self.padded_box(1.8, distance)
            self.assertLess(g.estimate_distance(box), distance)
            heights.append(g.estimate_height(box))
        self.assertTrue(all(h < 1.8 for h in heights))
        self.assertTrue(all(b < a for a, b in zip(heights, heights[1:])),
                        f'H should fall with distance, got {heights}')

    def test_over_correcting_reads_further_and_then_infinitely_far(self) -> None:
        """An over-corrected foot row walks up toward the horizon, so the reading runs away — the
        failure is loud (a person past the far edge, or `inf`), never a negative distance. It is
        also why `foot_offset` must be tuned by `H` staying flat, not overshot: too much makes the
        far-edge filter reject people who are inside."""
        box: Rect = self.true_box(1.8, 3.0)
        gap: float = feet_row(3.0) - HORIZON_ROW                  # the feet's rows below the horizon
        self.assertGreater(self.make_geometry(gap * 0.5).estimate_distance(box), 3.0)
        self.assertEqual(self.make_geometry(gap).estimate_distance(box), math.inf)
        self.assertEqual(self.make_geometry(gap * 1.5).estimate_distance(box), math.inf)

    def test_the_azimuth_ignores_it_entirely(self) -> None:
        # The bearing no longer rides on any box row, so no value of this setting can move it.
        roi, _d = synth_observation(0, world_azimuth=90.0, radius=2.0)
        angles = [self.make_geometry(offset).calc_angle(roi, 0)[1]
                  for offset in (0.0, 0.02, self.PAD, 0.15)]
        self.assertAlmostEqual(max(angles), min(angles), places=12)


class TestReachReadouts(unittest.TestCase):
    """`feet_from` / `hands_from` / `hands_seam`: how near a person can stand and still be
    in frame, published live for the running configuration.

    Checked against a brute-force scan that shares nothing with the implementation but the
    sensor's coverage: the person is placed in plan view and seen from the camera with `atan2`, so a
    mistake in `camera_bearing`, `focus_distance` or the bisection shows up here.
    """

    # A frame TALLER than the sensor's reach: P720 at tilt 15 fills 944 rows on the centre column,
    # so on 960 the sensor, not the rows, is what ends the picture at the top — and the sensor's top
    # edge falls toward the sides. The configuration where the seam reach differs most from the axis.
    SRC = (1280, 720)
    ROWS = 960
    TILT = 15.0
    LENS_FOV = 128.9

    def make_config(self, resolution: CameraResolution = CameraResolution.P720,
                    lens_centre: tuple[float, float] = (0.0, 0.0)) -> PanoramicTrackerSettings:
        config = PanoramicTrackerSettings(fov=PARALLAX_FOV, resolution=resolution,
                                          tilt=self.TILT, frame_height=self.ROWS, lens_fov=self.LENS_FOV,
                                          lens_centre_x=lens_centre[0], lens_centre_y=lens_centre[1])
        config.rig.camera_radius = RING_RADIUS
        config.rig.camera_height = CAMERA_HEIGHT
        return config

    def brute_force(self, height: float, centre_bearing: float, ring: float, lens_height: float,
                    edge: int) -> float:
        """The first radius, in 1 mm steps, at which a point on the line is inside the picture."""
        window = frame_window(self.SRC, (self.SRC[0], self.ROWS), self.SRC[0], PARALLAX_FOV,
                              self.TILT, self.LENS_FOV)
        coverage = frame_coverage(self.SRC, (self.SRC[0], self.ROWS), self.SRC[0], PARALLAX_FOV,
                                  self.TILT, lens_fov=self.LENS_FOV)
        dpp: float = PARALLAX_FOV / self.SRC[0]
        centre: float = (self.SRC[0] - 1) / 2.0
        phi: float = math.radians(centre_bearing)
        for step in range(int(ring * 1000) + 1, 10000):
            radius: float = step / 1000.0
            dx, dy = radius * math.cos(phi) - ring, radius * math.sin(phi)     # camera at (ring, 0)
            distance: float = math.hypot(dx, dy)
            x: float = centre + math.degrees(math.atan2(dy, dx)) / dpp
            if distance <= 0.0 or x < -0.5 or x > self.SRC[0] - 0.5:
                continue
            row: int = int(coverage[int(round(min(max(x, 0.0), self.SRC[0] - 1.0))), edge])
            if row < 0:
                continue
            limit: float = window.elevation(row)
            angle: float = math.degrees(math.atan((height - lens_height) / distance))
            if (angle <= limit) if height >= lens_height else (angle >= limit):
                return radius
        return math.inf

    def test_published_reach_matches_a_brute_force_scan(self) -> None:
        config = self.make_config()
        PanoramicTracker(config, num_players=4, num_cameras=4)
        r = config.rig
        self.assertAlmostEqual(r.feet_from,
                               self.brute_force(0.0, 0.0, RING_RADIUS, CAMERA_HEIGHT, 1), delta=0.0015)
        self.assertAlmostEqual(r.hands_from,
                               self.brute_force(HANDS_HEIGHT, 0.0, RING_RADIUS, CAMERA_HEIGHT, 0),
                               delta=0.0015)
        seam: float = max(self.brute_force(HANDS_HEIGHT, side, RING_RADIUS, CAMERA_HEIGHT, 0)
                          for side in (-45.0, 45.0))
        self.assertAlmostEqual(r.hands_seam, seam, delta=0.0015)

    def test_when_the_sensor_ends_the_picture_the_seam_is_worse(self) -> None:
        """A frame taller than the sensor's reach shows the black arch, so raised hands are judged
        against a top that falls toward the seam: here R 1.73 on the axis but beyond R 1.9 on the
        seam. And the lower tilt-for-resolution leaves feet out of frame at R 1.5."""
        config = self.make_config(CameraResolution.P720)
        PanoramicTracker(config, num_players=4, num_cameras=4)
        r = config.rig
        self.assertAlmostEqual(r.feet_from, 1.65, delta=0.01)
        self.assertAlmostEqual(r.hands_from, 1.73, delta=0.01)
        self.assertGreater(r.hands_seam, r.hands_from + 0.2)

    def test_when_the_rows_end_the_picture_the_seam_barely_differs(self) -> None:
        """The opposite case, and the studio configuration: P800 with the shared lens's centre
        offset reaches 1136 rows at tilt 15, so 960 rows cap the top at the same angle on every
        column. The seam is then only worse by the ring's parallax — centimetres — which is why the
        two hands read-outs are worth having side by side: their gap says which of the two limits
        the frame is running into. Feet are in frame inside the zone's R 1.5 edge here."""
        config = self.make_config(CameraResolution.P800, lens_centre=(-10.5, 10.5))
        PanoramicTracker(config, num_players=4, num_cameras=4)
        r = config.rig
        self.assertLess(r.feet_from, 1.5)
        self.assertGreaterEqual(r.hands_seam, r.hands_from)
        self.assertLess(r.hands_seam - r.hands_from, 0.05)

    def test_the_ring_and_the_lens_height_update_it_live(self) -> None:
        # Both are live settings and both move the reach; neither needs the coverage redone.
        config = self.make_config()
        PanoramicTracker(config, num_players=4, num_cameras=4)
        feet, hands = config.rig.feet_from, config.rig.hands_from
        config.rig.camera_radius = RING_RADIUS + 0.1
        # On the axis the ring only shifts the answer outward by itself.
        self.assertAlmostEqual(config.rig.feet_from, feet + 0.1, delta=1e-6)
        self.assertAlmostEqual(config.rig.hands_from, hands + 0.1, delta=1e-6)
        config.rig.camera_height = CAMERA_HEIGHT + 0.2
        self.assertGreater(config.rig.feet_from, feet + 0.1)      # a higher lens sees feet further out
        self.assertLess(config.rig.hands_from, hands + 0.1)  # ...and raised hands nearer


class TestInitialGeometrySync(unittest.TestCase):

    def test_the_published_radius_is_the_geometrys_own(self) -> None:
        """No conversion between what `Geometry` derives and what the panorama draws with.

        The settings, the geometry, the marks, the shader and a tape on the floor all carry radii
        from the fixture axis, so this is an equality and not an `assertAlmostEqual` with a factor
        in it. That is the whole point of the convention: a halving anywhere in the chain has
        somewhere to show up.
        """
        config = PanoramicTrackerSettings(fov=PARALLAX_FOV)
        config.rig.camera_radius = RING_RADIUS
        for zone in ((1.5, 3.5), (1.0, 4.0), (2.0, 2.0)):
            with self.subTest(zone=zone):
                config.rig.zone_min_radius, config.rig.zone_max_radius = zone
                tracker = PanoramicTracker(config, num_players=4, num_cameras=4)
                self.assertEqual(config.rig.parallax_radius, tracker.geometry.parallax_radius)
                # ...and it really is the zone's harmonic mean, in the zone's own unit.
                lo, hi = zone
                self.assertAlmostEqual(config.rig.parallax_radius,
                                       2.0 * lo * hi / (lo + hi), places=9)
                self.assertGreaterEqual(config.rig.parallax_radius, lo)
                self.assertLessEqual(config.rig.parallax_radius, hi)

    def test_config_applied_to_geometry_at_construction(self) -> None:
        # bind() does not fire with the initial value and presets load before
        # the tracker exists, so construction must push config into geometry.
        config = PanoramicTrackerSettings(fov=PARALLAX_FOV)
        config.rig.camera_radius = RING_RADIUS
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
        # frame_height 0 resolving to the sensor's full reach at this tilt — and published as the
        # frame's shape and two edge angles, from which the row form is rebuilt exactly.
        rows = delivered_height(False, config.resolution, PARALLAX_FOV, config.tilt,
                                config.lens_fov, (config.lens_centre_x, config.lens_centre_y),
                                config.frame_height)
        window = frame_window((1280, 800), (1280, rows), 1280, PARALLAX_FOV, config.tilt,
                              config.lens_fov, (config.lens_centre_x, config.lens_centre_y))
        self.assertAlmostEqual(tracker.geometry._horizon_px, window.horizon_px, places=9)
        self.assertAlmostEqual(tracker.geometry._focal, window.focal, places=9)
        self.assertEqual(tracker.geometry._rows, rows)
        self.assertEqual(config.rig.hfov, PARALLAX_FOV)
        self.assertEqual(config.rig.tilt, config.tilt)
        self.assertAlmostEqual(config.rig.vfov, window.elevation_top - window.elevation_bottom, places=9)
        self.assertAlmostEqual(config.rig.angle_bottom, window.elevation_bottom, places=9)
        self.assertAlmostEqual(config.rig.angle_top, window.elevation_top, places=9)
        horizon_row, focal_rows = row_model(config.rig.angle_bottom, config.rig.angle_top)
        self.assertAlmostEqual(horizon_row, window.horizon_px / (rows - 1), places=9)
        self.assertAlmostEqual(focal_rows, window.focal / (rows - 1), places=9)


if __name__ == "__main__":
    unittest.main()
