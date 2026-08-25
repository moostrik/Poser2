"""Tests for the panoramic tracker: seam hysteresis, dead-zone handling,
cross-camera linking, world id reuse, and ring parallax correction."""

import math
import unittest

from modules.tracker import (
    PanoramicTracker, PanoramicTrackerSettings, PanoramicAnnotation,
    Tracklet, TrackingStatus, TrackletDict,
)
from modules.tracker.panoramic.geometry import Geometry
from modules.tracker.panoramic.store import TrackletIdPool
from modules.utils import Rect


# fov 110 / target 90 -> overlap 10 deg per seam side.
# With the default seam settings (reject 0.5, reach 1.3, hysteresis 0.9):
# dead zone: local angle <= 5 or >= 105; overlap flag: <= 15 or >= 95;
# cross-camera match reach: 13 deg.
FOV = 110.0


def make_tracklet(cam_id: int, ext_id: int, local_angle: float, *,
                  status: TrackingStatus = TrackingStatus.TRACKED,
                  height: float = 0.5, age: int = 10) -> Tracklet:
    width = 0.1
    center_x = local_angle / FOV
    roi = Rect(x=center_x - width / 2.0, y=0.1, width=width, height=height)
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

    def submit(self, *tracklets: Tracklet) -> TrackletDict:
        for t in tracklets:
            self.tracker._add_tracklet(t)
        self.tracker._update_and_notify()
        return self.emitted[-1]


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

    def test_height_gate_uses_setting(self) -> None:
        self.submit(make_tracklet(0, 1, 98.0, height=0.5))
        # 0.12 height difference: the old hardcoded 0.1 gate would refuse this
        # link; the default max_height_diff of 0.15 accepts it.
        out = self.submit(make_tracklet(1, 1, 8.0, height=0.62))
        self.assertEqual(set(out.keys()), {0})
        self.assertEqual(self.tracker.store.get_world_id(1, 1), 0)

    def test_height_gate_rejects_beyond_setting(self) -> None:
        self.config.seam.max_height_diff = 0.05
        self.submit(make_tracklet(0, 1, 98.0, height=0.5))
        out = self.submit(make_tracklet(1, 1, 8.0, height=0.62))
        self.assertEqual(set(out.keys()), {0, 1})  # link refused -> new world

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
        self.tracker._add_tracklet(make_tracklet(0, 1, 98.0, height=0.9))
        self.tracker._add_tracklet(make_tracklet(1, 1, 8.0, height=0.5))
        # Different heights kept them apart; now both drift to the same height
        # and the per-tick collapse merges the younger world into the older.
        self.tracker._add_tracklet(make_tracklet(0, 1, 98.0, height=0.5))
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


# Rig geometry for the parallax tests: 4 cameras, fov 127, ring radius 0.25 m.
PARALLAX_FOV = 127.0
TARGET_FOV = 90.0
RING_RADIUS = 0.25
PERSON_HEIGHT = 1.7
VFOV = 71.6


def synth_observation(cam_id: int, world_azimuth: float, radius: float) -> tuple[Rect, float]:
    """Build the ROI a camera on the ring would report for a person at
    ``world_azimuth`` degrees, ``radius`` m from the rig centre. Returns the ROI
    and the true camera->person distance. Inverts the tracker's own projection
    so the parallax correction can be checked against ground truth."""
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
    angular_height = 2.0 * math.degrees(math.atan((PERSON_HEIGHT / 2.0) / distance))
    roi_height = angular_height / VFOV
    width = 0.05
    center_x = local_angle / PARALLAX_FOV
    return Rect(x=center_x - width / 2.0, y=0.1, width=width, height=roi_height), distance


class TestGeometryParallax(unittest.TestCase):

    def make_geometry(self, ring_radius: float = RING_RADIUS) -> Geometry:
        g = Geometry(num_cameras=4, cam_fov=PARALLAX_FOV, target_fov=TARGET_FOV)
        g.set_ring_radius(ring_radius)
        g.set_person_height(PERSON_HEIGHT)
        g.set_vfov(VFOV)
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

    def test_estimate_distance_clamps_degenerate_box(self) -> None:
        g = self.make_geometry()
        self.assertEqual(g.estimate_distance(Rect(height=0.0)), 10.0)
        d = g.estimate_distance(Rect(height=0.5))
        self.assertGreaterEqual(d, 0.5)
        self.assertLessEqual(d, 10.0)


class TestInitialGeometrySync(unittest.TestCase):

    def test_config_applied_to_geometry_at_construction(self) -> None:
        # bind() does not fire with the initial value and presets load before
        # the tracker exists, so construction must push config into geometry.
        config = PanoramicTrackerSettings(fov=PARALLAX_FOV)
        config.parallax.ring_radius = RING_RADIUS
        config.parallax.vfov = 70.0
        config.distortion.poly.k2 = 0.1
        tracker = PanoramicTracker(config, num_players=4, num_cameras=4)
        self.assertEqual(tracker.geometry.cam_fov, PARALLAX_FOV)
        self.assertEqual(tracker.geometry._ring_radius, RING_RADIUS)
        self.assertEqual(tracker.geometry._vfov, 70.0)
        self.assertEqual(tracker.geometry._poly_k2, 0.1)


if __name__ == "__main__":
    unittest.main()
