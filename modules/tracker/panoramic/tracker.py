# Standard library imports
import logging
import time
from dataclasses import dataclass, replace
from itertools import combinations
from queue import Empty, Queue
from threading import Lock, Thread, Event
from typing import Callable

# Local application imports
from modules.oak import DepthTracklet, mode_size, frame_window, delivered_height
from .. import (
    BaseTracker, TrackerAnnotation,
    Tracklet, TrackingStatus, TrackletDict, TrackletDictCallback,
)
from .store import TrackletStore
from .geometry import Geometry
from .settings import SeamSettings, ParallaxSettings, TrackerSettings

TrackletListCallback = Callable[[list[Tracklet]], None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Annotation(TrackerAnnotation):
    """What one camera's box says about one person, all of it derived in `Geometry`.

    `distance` is from that camera (m); `height` is absolute (m) and needs no re-projection,
    since the same camera sees the person's feet and head.
    """
    local_angle: float
    world_angle: float
    overlap: bool
    distance: float = 0.0
    height: float = 0.0


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
    delivers one primary per world — the show's input, one pose per person — and
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

        # Wire fov and parallax changes to geometry
        TrackerSettings.fov.bind(config, lambda v: (self._set_frame(v), self._update_seam_angles()))
        ParallaxSettings.ring_radius.bind(config.parallax, lambda v: self.geometry.set_ring_radius(v))
        ParallaxSettings.camera_height.bind(config.parallax, lambda v: self.geometry.set_camera_height(v))

        # bind() does not fire with the current value, and the preset is loaded
        # before this tracker is constructed — push config into geometry once now.
        self._sync_geometry_from_config()

        # Wire seam ratio changes to the angles display
        SeamSettings.reject.bind(config.seam, lambda _: self._update_seam_angles())
        SeamSettings.reach.bind(config.seam, lambda _: self._update_seam_angles())
        self._update_seam_angles()

        # Last emitted primary per world id, as an observation id — view-selection state,
        # used for hysteresis
        self._primary_for_world: dict[int, int] = {}

        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletDictCallback] = set()
        self._observation_callbacks: set[TrackletListCallback] = set()

    def _sync_geometry_from_config(self) -> None:
        """Apply current config values to geometry. Needed at construction
        because ``bind`` does not fire with the initial value and the preset is
        loaded before the tracker exists."""
        self._set_frame(self.config.fov)
        self.geometry.set_ring_radius(self.config.parallax.ring_radius)
        self.geometry.set_camera_height(self.config.parallax.camera_height)

    def _set_frame(self, fov: float) -> None:
        """The delivered frame's geometry, from the same functions the camera's warp is built
        with: `fov` for the columns, `frame_window` for the rows.

        Rows are tangents of elevation with the horizon at `horizon_px`, not linear about the
        centre row, so the row model is a window rather than a `vfov`. Derived here, once, from
        the shared camera fields, and published as read-only fields on `parallax` so the
        panorama draws with exactly the numbers the tracker tracks with. Mono and landscape,
        which is what this tracker has always assumed."""
        self.geometry.set_fov(fov)
        c: TrackerSettings = self.config
        lens_centre: tuple[float, float] = (c.lens_centre_x, c.lens_centre_y)
        src: tuple[int, int] = mode_size(False, c.resolution)
        rows: int = delivered_height(False, c.resolution, fov, c.tilt, c.lens_fov, lens_centre, c.frame_height)
        window = frame_window(src, (src[0], rows), src[0], fov, c.tilt, c.lens_fov, lens_centre)
        self.geometry.set_window(window, rows)
        p: ParallaxSettings = c.parallax
        p.vfov = window.elevation_top - window.elevation_bottom
        p.elevation_bottom = window.elevation_bottom
        p.elevation_top = window.elevation_top
        p.horizon_row = window.horizon_px / (rows - 1)
        p.focal_rows = window.focal / (rows - 1)

    def _update_seam_angles(self) -> None:
        a = self.config.seam.angles
        a.fov = self.geometry.cam_fov
        a.overlap = self.geometry.fov_overlap
        a.reject = self.geometry.fov_overlap * self.config.seam.reject
        a.reach = self.geometry.fov_overlap * self.config.seam.reach

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

        # The device has finished with this track: its id is now free to be handed to a
        # different person, so the observation leaves the live index. It stays LOST and keeps
        # anchoring until `timeout`, which is what lets a far camera link across a seam after
        # the near one gave up.
        if new_tracklet.is_removed:
            if self.store.get_world_id(cam_id, ext_id) is not None:
                self.store.end_device_track(cam_id, ext_id)
            return

        # No detection this frame, but the device still holds the track: the same id will come
        # back, so the observation stays live.
        if new_tracklet.is_lost:
            if self.store.get_world_id(cam_id, ext_id) is not None:
                self.store.lose_tracklet(cam_id, ext_id)
            return

        # Filter out tracklets that are too young or too small
        if new_tracklet.external_age_in_frames <= self.config.min_age:
            return
        if new_tracklet.roi.height < self.config.min_height:
            return

        # Annotate with local/world angles, overlap flag, estimated distance and height
        local_angle, world_angle, _overlap, distance, height = self.geometry.get_angles_and_overlap(new_tracklet.roi, cam_id, self.config.seam.reject)
        new_tracklet = replace(new_tracklet, annotation=Annotation(local_angle, world_angle, _overlap, distance, height))

        # Existing observation — refresh in place, even inside the edge dead
        # zone: starving it would freeze its angles and expire it via timeout
        # while the camera still tracks the person.
        if self.store.get_world_id(cam_id, ext_id) is not None:
            self.store.replace_tracklet(new_tracklet)
            return

        if not new_tracklet.is_active:
            return

        # A re-acquisition in the same camera is a continuation, not an arrival, so it is
        # allowed anywhere in frame — including the edge dead zone, which only exists to stop
        # *new* people being born on a seam.
        anchor_world: int | None = self._find_same_camera_anchor(new_tracklet)
        if anchor_world is not None:
            self.store.add_tracklet(new_tracklet, world_id=anchor_world)
            return

        # Brand-new observations are ignored too close to the FOV edge
        if self.geometry.angle_in_edge(local_angle, self.config.seam.reject):
            return
        if _overlap:
            candidate_world: int | None = self._find_world_candidate(new_tracklet)
            if candidate_world is not None:
                self.store.add_tracklet(new_tracklet, world_id=candidate_world)
                return
        self.store.add_tracklet(new_tracklet)

    def _find_same_camera_anchor(self, new_tracklet: Tracklet) -> int | None:
        """The world of a lost observation in the SAME camera that this new one continues.

        The device tracker is zero-term: no appearance model, association from box overlap
        alone. A person it drops — an occlusion, a missed detection — returns under a different
        id, and without this they would become a new world mid-field. Matching on position and
        height makes that continuity an explicit rule of ours, rather than an accident of the
        device reusing its numbers. Bounded by ``seam.relink_angle``, kept small because two
        people standing closer than that could be confused for one another.
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
            if diff > self.config.seam.relink_angle:
                continue
            if abs(t.roi.height - new_tracklet.roi.height) > self.config.seam.max_height_diff:
                continue
            if diff < best_diff:
                best_diff = diff
                best_world = t.id
        return best_world

    def _observations_match(self, a: Tracklet, b: Tracklet) -> bool:
        """True if two observations from different cameras are close enough in
        world angle and height to be considered the same person."""
        if a.cam_id == b.cam_id:
            return False
        if not isinstance(a.annotation, Annotation) or not isinstance(b.annotation, Annotation):
            return False
        if self.geometry.angle_diff(a.annotation.world_angle, b.annotation.world_angle) > self.geometry.fov_overlap * self.config.seam.reach:
            return False
        if abs(a.roi.height - b.roi.height) > self.config.seam.max_height_diff:
            return False
        return True

    def _find_world_candidate(self, new_tracklet: Tracklet) -> int | None:
        """Return the world id whose other-camera observation best matches
        ``new_tracklet`` in angle and height; None if none match. LOST
        observations still anchor (removal is bounded by ``timeout``) so the
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
            if t.is_expired(self.config.timeout):
                self.store.retire_tracklet(t.obs_id)

        # Late safety net: collapse worlds whose observations match each other
        # (handles ambiguous simultaneous arrivals that each got their own world).
        for keep_id, drop_id in self._find_world_collapse_pairs():
            if self.store.merge_worlds(keep_id, drop_id):
                self._primary_for_world.pop(drop_id, None)

        # Emit one primary per world, while it is still being seen. `emit_hold` is shorter than
        # `timeout` on purpose: an observation keeps anchoring a seam crossing long after the
        # person it describes should stop driving the light, the sound and the hit detector.
        now: float = time.time()
        hold: float = self.config.emit_hold
        emitted: TrackletDict = {}
        for world_id in self.store.all_world_ids():
            primary: Tracklet | None = self._pick_primary(world_id)
            if primary is None:
                continue
            # `_pick_primary` returns the most recently active member, so one test covers the
            # whole world: if even that one is stale, nobody has seen this person lately.
            if now - primary.last_active > hold:
                continue
            emitted[world_id] = primary
        self._notify_callback(emitted)

        # Every live observation, for the calibration view: the two cameras' separate opinions
        # of a person on a seam, which the primaries above deliberately reduce to one.
        self._notify_observation_callback(
            [t for t in self.store.all_tracklets() if not t.is_removed]
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
        ratio. Truly departed observations are bounded by the timeout-based
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
            and self.geometry.angle_in_overlap(t.annotation.local_angle, self.config.seam.reach - 1.0)
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
