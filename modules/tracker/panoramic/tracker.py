# Standard library imports
import logging
import math
import time
from dataclasses import dataclass, replace
from enum import IntEnum, auto
from functools import lru_cache
from itertools import combinations
from queue import Empty, Queue
from threading import Lock, Thread, Event
from typing import Callable

# Third-party imports
import numpy as np

# Local application imports
from modules.oak import DepthTracklet, FrameWindow, degrees_per_pixel, delivered_height, \
    frame_coverage, frame_window, mode_size
from .. import (
    BaseTracker, TrackerAnnotation,
    Tracklet, TrackingStatus, TrackletDict, TrackletDictCallback,
)
from .store import TrackletStore
from .geometry import Geometry, height_is_measured
from .panorama_map import reach_radius
from .settings import RigSettings, TrackerSettings

TrackletListCallback = Callable[[list[Tracklet]], None]

logger = logging.getLogger(__name__)

# The reference person's overhead reach (m): 1.8 m tall, hands at 2.2 m with the arms up —
# CALIBRATION.md, *Tilt — derived from the build*. The height the `hands_*` read-outs are
# quoted at. A constant, not a setting: nothing tunes it, and the fields it feeds are read-only.
HANDS_HEIGHT: float = 2.2


@lru_cache(maxsize=8)
def _coverage(src: tuple[int, int], rows: int, fov: float, tilt: float, lens_fov: float,
              lens_centre: tuple[float, float]) -> np.ndarray:
    """`frame_coverage` for one frame configuration, computed once per process.

    It projects the whole output grid (≈0.1 s) and every input is an init field, so the running app
    pays for it once — but the tests build trackers by the dozen, mostly on the same few frames.
    Read-only, since every caller shares the one array.
    """
    coverage: np.ndarray = frame_coverage(src, (src[0], rows), src[0], fov, tilt,
                                          lens_fov=lens_fov, lens_centre=lens_centre)
    coverage.flags.writeable = False
    return coverage


class Rejection(IntEnum):
    """Why the tracker did not count a detection this frame — one member per filter in `_add_tracklet`.

    Carried on the annotation so the panorama can draw the dropped detection and name the filter,
    rather than it vanishing. `rejection_label` is the one spelling of each.
    """
    YOUNG = auto()       # the device has not held it for `age_filter` frames yet
    SMALL = auto()       # its box is shorter than `height_filter`
    DEAD_ZONE = auto()   # a new arrival inside `seam.dead_zone` of a field edge
    PAST_EDGE = auto()   # past `rig.zone_max_radius`, with `zone_filter` on


def rejection_label(reason: Rejection, zone_max_radius: float) -> str:
    """The short name of a filter, as the panorama tags a box with it."""
    if reason == Rejection.PAST_EDGE:
        return f'past R{zone_max_radius:g}'
    return {Rejection.YOUNG: 'young', Rejection.SMALL: 'small', Rejection.DEAD_ZONE: 'dead zone'}[reason]


@dataclass(frozen=True)
class Annotation(TrackerAnnotation):
    """What one camera's box says about one person, all of it derived in `Geometry`.

    `distance` is from that camera (m); `height` is absolute (m) and needs no re-projection,
    since the same camera sees the person's feet and head. `rejected` is set when a filter
    dropped this detection (or, for a person already tracked, stopped counting it).
    """
    local_angle: float
    world_angle: float
    overlap: bool
    distance: float = 0.0
    height: float = 0.0
    rejected: Rejection | None = None


class Tracker(Thread, BaseTracker):
    """
    Tracks N people across a ring of cameras sharing a 360° field of view.

    Each camera runs its own on-device YOLO tracker and emits per-camera tracklets
    with stable local IDs. This class fuses those streams into world-space
    identities. Per-camera tracklet observations are stored as immutable records
    with host-owned ids (see ``TrackletStore``): their ``cam_id``,
    ``external_id``, ``roi``, and ``annotation`` are never rewritten by fusion.
    A separate world id (drawn from the pool) groups one or more observations
    into a single identity emitted to callbacks.

    - **Cross-camera continuity**: when a brand-new observation arrives inside
      an overlap zone, it is linked into the existing world whose other-camera
      observation matches it best in world angle (recently LOST observations
      still anchor). No merge, no rewrite — the person keeps the same id as
      they cross.
    - **Same-camera continuity**: the device tracker has no appearance model, so
      a person it loses and re-acquires comes back under a *new* id. A new
      observation close to a lost one in the same camera rejoins its world, which
      puts that continuity under our control instead of leaving it to the
      device's habit of reusing numbers.
    - **View selection (primary)**: per world, the tracker emits one primary
      observation each tick. Selection is sticky with hysteresis (governed by
      ``seam.hysteresis``): the current primary stays primary unless a
      competitor's distance from the FOV edge exceeds it by the hysteresis
      ratio. Smooth handoff at seams without flicker.
    - **Late safety net**: if two genuine new arrivals at a seam each got their
      own world, a per-tick scan can collapse them into one via ``merge_worlds``.

    Processing runs in a background thread. Camera data is submitted via
    ``submit_cam_tracklets``. Two output channels: ``add_tracklet_callback``
    delivers one primary per world still remembered, LOST ones included — freshness
    is the consumer's to judge (pose by box age, the show by ``is_active``) — and
    ``add_observation_callback`` delivers every live observation, which is the
    only way to see the two cameras' separate opinions of a person on a seam.
    """

    def __init__(self, config: TrackerSettings, num_players: int, num_cameras: int) -> None:
        super().__init__()

        self._running: bool = False
        self._update_event: Event = Event()

        self._input_queue: Queue[Tracklet] = Queue()

        self.store: TrackletStore = TrackletStore(num_players)

        self.config: TrackerSettings = config
        # Each camera owns 360/num_cameras degrees of the ring; whatever its field has beyond
        # that is the overlap it shares with its neighbours.
        self.geometry: Geometry = Geometry(num_cameras, config.fov, 360.0 / num_cameras)
        # Where the sensor's picture ends, per camera bearing: (top, bottom) in degrees from eye
        # level. Set by `_set_frame`; the reach read-outs are measured against it.
        self._picture_edges: tuple[Callable[[float], float], Callable[[float], float]] | None = None

        # Wire fov and rig changes to geometry. The three that move the overlap band must also
        # republish it, or the readout and the panorama's lines go stale on a live drag. The ring
        # and the lens height also move the reach, which is cheap to redo from the cached edges.
        TrackerSettings.fov.bind(config, lambda v: (self._set_frame(v), self._update_seam_angles()))
        RigSettings.camera_radius.bind(config.rig, lambda v: (self.geometry.set_camera_radius(v),
                                                              self._set_zone(), self._publish_reach()))
        RigSettings.camera_height.bind(config.rig, lambda v: (self.geometry.set_camera_height(v),
                                                              self._publish_reach()))
        RigSettings.zone_min_radius.bind(config.rig, lambda _: self._set_zone())
        RigSettings.zone_max_radius.bind(config.rig, lambda _: self._set_zone())
        TrackerSettings.foot_offset.bind(config, lambda v: self.geometry.set_foot_offset(v))

        # bind() does not fire with the current value, and the preset is loaded
        # before this tracker is constructed — push config into geometry once now.
        self._sync_geometry_from_config()

        self._update_seam_angles()

        # Last emitted primary per world id, as an observation id — view-selection state,
        # used for hysteresis
        self._primary_for_world: dict[int, int] = {}

        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletDictCallback] = set()
        self._observation_callbacks: set[TrackletListCallback] = set()

        # Detections the intake dropped, latest per device track, tagged with the filter — published
        # on the observation channel so the panorama can draw them, never on the primary channel.
        # Touched only on the tracker thread (`_add_tracklet` and `_update_and_notify`).
        self._rejected: dict[tuple[int, int], Tracklet] = {}

    def _sync_geometry_from_config(self) -> None:
        """Apply current config values to geometry. Needed at construction because ``bind`` does
        not fire with the initial value and the preset is loaded before the tracker exists.

        The rig goes first: the overlap band is derived from the ring and the zone, so pushing
        `fov` before them would derive it once against the defaults."""
        r: RigSettings = self.config.rig
        self.geometry.set_camera_radius(r.camera_radius)
        self.geometry.set_camera_height(r.camera_height)
        self.geometry.set_foot_offset(self.config.foot_offset)
        self._set_zone()
        self._set_frame(self.config.fov)

    def _set_zone(self) -> None:
        r: RigSettings = self.config.rig
        self.geometry.set_zone(r.zone_min_radius, r.zone_max_radius)
        # Published so the panorama draws its marks on the same cylinder the azimuth is corrected
        # at; derived from the zone, never set. Straight across, no conversion — the setting, the
        # geometry and the panorama all speak radii.
        r.parallax_radius = self.geometry.parallax_radius
        self._update_seam_angles()

    def _set_frame(self, fov: float) -> None:
        """The delivered frame's geometry, from the same functions the camera's warp is built
        with: `fov` for the columns, `frame_window` for the rows.

        Rows are tangents of elevation with the horizon at `horizon_px`, not linear about the
        centre row, so the row model is a window rather than a `vfov`. Derived here, once, from
        the shared camera fields, and published as read-only fields on `rig` so the panorama draws
        with exactly the numbers the tracker tracks with — as the frame's two edge **angles**, from
        which `panorama_map.row_model` rebuilds the row form exactly. Mono and landscape, which is
        what this tracker has always assumed."""
        self.geometry.set_fov(fov)
        c: TrackerSettings = self.config
        lens_centre: tuple[float, float] = (c.lens_centre_x, c.lens_centre_y)
        src: tuple[int, int] = mode_size(False, c.resolution)
        rows: int = delivered_height(False, c.resolution, fov, c.tilt, c.lens_fov, lens_centre, c.frame_height)
        window = frame_window(src, (src[0], rows), src[0], fov, c.tilt, c.lens_fov, lens_centre)
        self.geometry.set_window(window, rows)
        p: RigSettings = c.rig
        p.hfov = fov
        p.vfov = window.elevation_top - window.elevation_bottom
        p.tilt = c.tilt
        p.angle_bottom = window.elevation_bottom
        p.angle_top = window.elevation_top
        self._set_picture_edges(_coverage(src, rows, fov, c.tilt, c.lens_fov, lens_centre),
                                window, src[0], fov)
        self._publish_reach()

    def _set_picture_edges(self, coverage: np.ndarray, window: FrameWindow, width: int,
                           fov: float) -> None:
        """Per camera bearing, the angle where the sensor's picture ends: its top and its bottom.

        Not the frame's top and bottom rows — the sensor fills less than the frame toward the sides
        (the black arch), and at some presets less than the frame even on axis — so a reach judged
        against the rows would be too generous exactly where it matters. NaN outside the field or
        on a column with no picture, which `reach_radius` reads as "not in frame". The column of a
        bearing is taken the way `coverage_summary` takes it, so the two describe the same pixels.
        """
        dpp: float = degrees_per_pixel(fov, width)
        centre: float = (width - 1) / 2.0

        def edge(bearing: float, which: int) -> float:
            if dpp <= 0.0:
                return math.nan
            x: float = centre + bearing / dpp
            if x < -0.5 or x > width - 0.5:
                return math.nan
            row: int = int(coverage[int(round(min(max(x, 0.0), width - 1.0))), which])
            return window.elevation(row) if row >= 0 else math.nan

        self._picture_edges = (lambda bearing: edge(bearing, 0), lambda bearing: edge(bearing, 1))

    def _publish_reach(self) -> None:
        """How near the fixture a person can stand and still be in frame — the tilt's trade, live.

        Feet on the camera axis only: the frame is pinned at the sensor's lowest centre-column
        reach, so the bottom row is covered at every bearing. Raised hands on the axis and on the
        seam, the worse of the seam's two sides (a lens-centre offset makes them differ): the
        sensor's top edge falls toward the frame edges, and a seam is where people cross.
        """
        if self._picture_edges is None:
            return
        top, bottom = self._picture_edges
        r: RigSettings = self.config.rig
        ring: float = max(0.0, r.camera_radius)
        seam: float = self.geometry.target_fov / 2.0
        r.feet_from = reach_radius(0.0, 0.0, r.camera_height, ring, bottom)
        r.hands_from = reach_radius(HANDS_HEIGHT, 0.0, r.camera_height, ring, top)
        r.hands_seam = max(reach_radius(HANDS_HEIGHT, side, r.camera_height, ring, top)
                                for side in (-seam, seam))

    def _update_seam_angles(self) -> None:
        """The shared overlap, for the panorama. Moved by `fov`, by the ring and by the zone's far
        edge — the three inputs it is derived from — so every one of them rebinds to this. The
        fusion settings are not here: they are already in the units they are drawn in.

        Published in **world azimuth**, which is what can be measured against the panorama's degree
        grid and what the overlap lines are drawn from. The local-angle band `angle_in_overlap`
        tests stays inside `Geometry`."""
        self.config.rig.overlap = self.geometry.overlap_azimuth

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        super().start()

    def stop(self) -> None:
        self._running = False

        with self._callback_lock:
            self._tracklet_callbacks.clear()

        self.join(timeout=1.0)  # Wait for the thread to finish

    def notify_update(self) -> None:
        if self._running:
            self._update_event.set()

    def run(self) -> None:
        while self._running:
            self._update_event.wait(timeout=0.1)
            self._update_event.clear()

            try:
                while True:
                    try:
                        tracklet: Tracklet = self._input_queue.get(block=False)
                        self._add_tracklet(tracklet)
                    except Empty:
                        break

                self._update_and_notify()
            except Exception:
                logger.exception("PanoramicTracker error")

    def _add_tracklet(self, new_tracklet: Tracklet) -> None:
        cam_id: int = new_tracklet.cam_id
        ext_id: int = new_tracklet.external_id
        key: tuple[int, int] = (cam_id, ext_id)

        # The device has finished with this track: its id is now free to be handed to a
        # different person, so the observation leaves the live index. It stays LOST and keeps
        # anchoring until `lost_timeout`, which is what lets a far camera link across a seam
        # after the near one gave up.
        if new_tracklet.is_removed:
            self._rejected.pop(key, None)
            if self.store.get_world_id(cam_id, ext_id) is not None:
                self.store.end_device_track(cam_id, ext_id)
            return

        # No detection this frame, but the device still holds the track: the same id will come
        # back, so the observation stays live.
        if new_tracklet.is_lost:
            self._rejected.pop(key, None)
            if self.store.get_world_id(cam_id, ext_id) is not None:
                self.store.lose_tracklet(cam_id, ext_id)
            return

        # Annotate first, before any filter: a detection a filter drops is still drawn on the
        # panorama, tagged with the reason, and it needs its angles for that. Pure geometry, so the
        # order changes nothing that is decided below.
        local_angle, world_angle, _overlap, distance, height = self.geometry.get_angles_and_overlap(new_tracklet.roi, cam_id)
        new_tracklet = replace(new_tracklet, annotation=Annotation(local_angle, world_angle, _overlap, distance, height))
        existing: bool = self.store.get_world_id(cam_id, ext_id) is not None

        # Too young or too small: not counted, and — for a person already tracked — not refreshed,
        # so their observation goes stale and expires.
        if new_tracklet.external_age_in_frames <= self.config.age_filter:
            self._reject(new_tracklet, Rejection.YOUNG)
            return
        if new_tracklet.roi.height < self.config.height_filter:
            self._reject(new_tracklet, Rejection.SMALL)
            return

        # Past the zone's far edge the tracker does not see this person — handled exactly like a
        # missed detection, before every branch below, so it cannot be born, re-acquired or linked
        # there. Someone already tracked goes LOST: their pose carries on from their last box for
        # `pose.tracklets.detection_timeout`, so a jump or a moment of hidden feet changes nothing
        # visible; forgotten after `lost_timeout`; the same
        # person again if they step back inside before that. Their latest position is kept, tagged,
        # so the panorama's mark follows them out and says why it is fading.
        if self.config.zone_filter and self.geometry.beyond_zone(local_angle, distance):
            if existing:
                self._rejected.pop(key, None)
                self.store.lose_tracklet(cam_id, ext_id,
                                         latest=self._tagged(new_tracklet, Rejection.PAST_EDGE))
            else:
                self._reject(new_tracklet, Rejection.PAST_EDGE)
            return

        # Existing observation — refresh in place, even inside the edge dead
        # zone: starving it would freeze its angles and expire it via lost_timeout
        # while the camera still tracks the person.
        if existing:
            self._rejected.pop(key, None)
            self.store.replace_tracklet(new_tracklet)
            return

        if not new_tracklet.is_active:
            return

        # A re-acquisition in the same camera is a continuation, not an arrival, so it is
        # allowed anywhere in frame — including the edge dead zone, which only exists to stop
        # *new* people being born on a seam.
        anchor_world: int | None = self._find_same_camera_anchor(new_tracklet)
        if anchor_world is not None:
            self._rejected.pop(key, None)
            self.store.add_tracklet(new_tracklet, world_id=anchor_world)
            return

        # Brand-new observations are ignored too close to the FOV edge
        if self.geometry.angle_in_edge(local_angle, self.config.seam.dead_zone):
            self._reject(new_tracklet, Rejection.DEAD_ZONE)
            return
        self._rejected.pop(key, None)
        if _overlap:
            candidate_world: int | None = self._find_world_candidate(new_tracklet)
            if candidate_world is not None:
                self.store.add_tracklet(new_tracklet, world_id=candidate_world)
                return
        self.store.add_tracklet(new_tracklet)

    @staticmethod
    def _tagged(tracklet: Tracklet, reason: Rejection) -> Tracklet:
        assert isinstance(tracklet.annotation, Annotation)
        return replace(tracklet, annotation=replace(tracklet.annotation, rejected=reason))

    def _reject(self, tracklet: Tracklet, reason: Rejection) -> None:
        """Not counted this frame: keep the detection, tagged, for the panorama instead of losing it."""
        self._rejected[(tracklet.cam_id, tracklet.external_id)] = self._tagged(tracklet, reason)

    def _find_same_camera_anchor(self, new_tracklet: Tracklet) -> int | None:
        """The world of a lost observation in the SAME camera that this new one continues.

        The device tracker is zero-term: no appearance model, association from box overlap
        alone. A person it drops — an occlusion, a missed detection — returns under a different
        id, and without this they would become a new world mid-field. Matching on position makes
        that continuity an explicit rule of ours, rather than an accident of the device reusing
        its numbers. Bounded by ``reacquire_angle``, kept small because two people standing
        closer than that could be confused for one another — and by nothing else: the reason a
        person was dropped is usually that something changed about their box, so gating the
        re-acquisition on the box still agreeing asks the wrong question.
        """
        assert isinstance(new_tracklet.annotation, Annotation)
        new_angle: float = new_tracklet.annotation.local_angle
        best_world: int | None = None
        best_diff: float = float('inf')
        for t in self.store.all_tracklets():
            if t.cam_id != new_tracklet.cam_id or t.is_active or t.is_removed:
                continue
            if not isinstance(t.annotation, Annotation):
                continue
            diff: float = abs(t.annotation.local_angle - new_angle)
            if diff > self.config.reacquire_angle:
                continue
            if diff < best_diff:
                best_diff = diff
                best_world = t.id
        return best_world

    def _observations_match(self, a: Tracklet, b: Tracklet) -> bool:
        """True if two observations from DIFFERENT cameras are one person.

        Two gates, both in real units. **Azimuth** is the decisive one: within
        ``seam.link_angle`` degrees of world bearing, which is the quantity the panorama check
        verifies, and which is wide enough to cover the disagreement a body of real width
        produces at a seam. **Height** is a veto, as a percentage of the larger of the two
        measured heights (``seam.link_height``) — scale-free, so two cameras at genuinely
        different distances from the same person still agree, where their box heights in pixels
        do not. It is skipped unless both readings are measurements (`height_is_measured`), so a
        jump, a mangled box or feet off the floor cannot refuse a link that the azimuth supports.
        """
        if a.cam_id == b.cam_id:
            return False
        if not isinstance(a.annotation, Annotation) or not isinstance(b.annotation, Annotation):
            return False
        if self.geometry.angle_diff(a.annotation.world_angle, b.annotation.world_angle) > self.config.seam.link_angle:
            return False
        return self._heights_match(a.annotation.height, b.annotation.height)

    def _heights_match(self, height_a: float, height_b: float) -> bool:
        """The scale-free height veto: `|a - b| / max(a, b)` against ``seam.link_height``, a
        fraction. Passes whenever either side has no usable reading — see `_observations_match`."""
        if not (height_is_measured(height_a) and height_is_measured(height_b)):
            return True
        largest: float = max(height_a, height_b)
        return abs(height_a - height_b) / largest <= self.config.seam.link_height

    def _find_world_candidate(self, new_tracklet: Tracklet) -> int | None:
        """Return the world id whose other-camera observation best matches
        ``new_tracklet`` in angle and height; None if none match. LOST
        observations still anchor (removal is bounded by ``lost_timeout``) so the
        link survives the previous camera losing the person first. Among
        multiple matching worlds the closest in world angle wins."""
        assert isinstance(new_tracklet.annotation, Annotation)
        best_world: int | None = None
        best_diff: float = float('inf')
        for t in self.store.all_tracklets():
            if t.is_removed:
                continue
            if not self._observations_match(new_tracklet, t):
                continue
            assert isinstance(t.annotation, Annotation)
            diff: float = self.geometry.angle_diff(new_tracklet.annotation.world_angle, t.annotation.world_angle)
            if diff < best_diff:
                best_diff = diff
                best_world = t.id
        return best_world

    def _update_and_notify(self) -> None:
        # Expire timed-out observations
        for t in self.store.all_tracklets():
            if t.is_expired(self.config.lost_timeout):
                self.store.retire_tracklet(t.obs_id)

        # Late safety net: collapse worlds whose observations match each other
        # (handles ambiguous simultaneous arrivals that each got their own world).
        for keep_id, drop_id in self._find_world_collapse_pairs():
            if self.store.merge_worlds(keep_id, drop_id):
                self._primary_for_world.pop(drop_id, None)

        # Emit one primary per world the tracker still remembers, LOST or not, until `lost_timeout`
        # retires it. How stale is too stale is each consumer's call, not this one's: pose stops
        # posing a person after `pose.tracklets.detection_timeout`, and the show counts only active
        # tracklets. Filtering here instead would decide for all of them with one number. A world
        # retired just above is still in the store until the end of this tick; it is not emitted.
        emitted: TrackletDict = {}
        for world_id in self.store.all_world_ids():
            primary: Tracklet | None = self._pick_primary(world_id)
            if primary is not None and not primary.is_removed:
                emitted[world_id] = primary
        self._notify_callback(emitted)

        # Dropped detections a camera has stopped reporting: a device track normally ends with LOST
        # or REMOVED, which clears its entry, but a camera that simply goes quiet does not.
        now: float = time.time()
        lost_timeout: float = self.config.lost_timeout
        for key in [k for k, t in self._rejected.items() if now - t.last_active > lost_timeout]:
            del self._rejected[key]

        # Every live observation, for the calibration view: the two cameras' separate opinions
        # of a person on a seam, which the primaries above deliberately reduce to one — and every
        # detection a filter dropped, tagged, so nobody vanishes from the strip without a reason.
        self._notify_observation_callback(
            [t for t in self.store.all_tracklets() if not t.is_removed] + list(self._rejected.values())
        )

        # Drop REMOVED observations and prune stale primary entries
        for t in self.store.all_tracklets():
            if t.status == TrackingStatus.REMOVED:
                self.store.remove_tracklet(t.obs_id)
        live_worlds: set[int] = set(self.store.all_world_ids())
        for world_id in list(self._primary_for_world):
            if world_id not in live_worlds:
                del self._primary_for_world[world_id]

    def _pick_primary(self, world_id: int) -> Tracklet | None:
        """Sticky primary selection with hysteresis to avoid per-tick flicker.

        The incumbent is held through transient LOST states: takeover
        candidates must be active, but a LOST incumbent only yields once a
        competitor beats its last-known edge distance by the hysteresis
        ratio. Truly departed observations are bounded by the ``lost_timeout``
        retirement in ``_update_and_notify``.
        """
        members: list[Tracklet] = self.store.get_tracklets(world_id)
        if not members:
            return None
        active: list[Tracklet] = [t for t in members if t.is_active and isinstance(t.annotation, Annotation)]
        if not active:
            chosen: Tracklet = max(members, key=lambda t: t.last_active)
            self._primary_for_world[world_id] = chosen.obs_id
            return chosen

        def edge(t: Tracklet) -> float:
            assert isinstance(t.annotation, Annotation)
            return self.geometry.angle_from_edge(t.annotation.local_angle)

        current_key: int | None = self._primary_for_world.get(world_id)
        current: Tracklet | None = next(
            (t for t in members
             if t.obs_id == current_key
             and not t.is_removed
             and isinstance(t.annotation, Annotation)),
            None,
        )
        if current is None:
            chosen = max(active, key=edge)
        else:
            best_competitor: Tracklet = max(active, key=edge)
            if best_competitor is current:
                chosen = current
            else:
                hysteresis: float = self.config.seam.hysteresis
                chosen = best_competitor if edge(best_competitor) >= edge(current) / hysteresis else current

        self._primary_for_world[world_id] = chosen.obs_id
        return chosen

    def _find_world_collapse_pairs(self) -> list[tuple[int, int]]:
        """Find world id pairs that should be collapsed because their observations
        match across cameras. Returns (keep_id, drop_id) pairs; older world wins.
        Each world id appears in at most one pair to avoid collapsing into a
        world that is itself about to be dropped. Only mutual nearest matches
        collapse, so an observation already explained by a partner in its own
        world cannot drag a neighbouring world into a merge."""
        observations: list[Tracklet] = [
            t for t in self.store.all_tracklets()
            if not t.is_removed
            and isinstance(t.annotation, Annotation)
            # Eligible where a second opinion exists at all: the picture's overlap, not a tuned
            # zone. Whether two eligible observations are one person is `_observations_match`.
            and self.geometry.angle_in_overlap(t.annotation.local_angle)
        ]

        def nearest_match(t: Tracklet) -> Tracklet | None:
            best: Tracklet | None = None
            best_diff: float = float('inf')
            assert isinstance(t.annotation, Annotation)
            for o in observations:
                if o is t or not self._observations_match(t, o):
                    continue
                assert isinstance(o.annotation, Annotation)
                diff: float = self.geometry.angle_diff(t.annotation.world_angle, o.annotation.world_angle)
                if diff < best_diff:
                    best_diff = diff
                    best = o
            return best

        nearest: list[Tracklet | None] = [nearest_match(t) for t in observations]

        used: set[int] = set()  # world ids already committed to a pair
        pairs: list[tuple[int, int]] = []
        for (i, a), (j, b) in combinations(enumerate(observations), 2):
            if a.id == b.id:
                continue
            if a.id in used or b.id in used:
                continue
            # A LOST observation may anchor a merge, but never merge two
            # worlds on lost data alone.
            if not (a.is_active or b.is_active):
                continue
            if nearest[i] is not b or nearest[j] is not a:
                continue
            # Older world wins
            members_a: list[Tracklet] = self.store.get_tracklets(a.id)
            members_b: list[Tracklet] = self.store.get_tracklets(b.id)
            oldest_a: float = min(t.created_at for t in members_a) if members_a else float('inf')
            oldest_b: float = min(t.created_at for t in members_b) if members_b else float('inf')
            keep_id, drop_id = (a.id, b.id) if oldest_a <= oldest_b else (b.id, a.id)
            used.add(a.id)
            used.add(b.id)
            pairs.append((keep_id, drop_id))
        return pairs

    # CALLBACKS
    def _notify_callback(self, tracklets: TrackletDict) -> None:
        with self._callback_lock:
            for c in self._tracklet_callbacks:
                c(tracklets)

    def _notify_observation_callback(self, observations: list[Tracklet]) -> None:
        with self._callback_lock:
            for c in self._observation_callbacks:
                c(observations)

    def add_tracklet_callback(self, callback: TrackletDictCallback) -> None:
        with self._callback_lock:
            self._tracklet_callbacks.add(callback)

    def add_observation_callback(self, callback: TrackletListCallback) -> None:
        """Every live observation each tick, one per camera that can see a person — not one per
        person. For the calibration view; the show reads ``add_tracklet_callback``."""
        with self._callback_lock:
            self._observation_callbacks.add(callback)

    def submit_cam_tracklets(self, cam_id: int, cam_tracklets: list[DepthTracklet]) -> None:
        for t in cam_tracklets:
            tracklet: Tracklet | None = Tracklet.from_depthcam(cam_id, t)
            if tracklet is None:
                logger.warning(f"Invalid tracklet from camera {cam_id}, skipping.")
                continue
            self._input_queue.put(tracklet)
