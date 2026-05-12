# Standard library imports
import logging
from dataclasses import dataclass, replace
from itertools import combinations
from queue import Empty, Queue
from threading import Lock, Thread, Event

# Local application imports
from modules.oak import DepthTracklet
from modules.settings import BaseSettings, Field, Group
from .. import (
    BaseTracker, TrackerAnnotation,
    Tracklet, TrackingStatus, TrackletDict, TrackletDictCallback,
)
from .panoramic_tracklet_manager import PanoramicTrackletManager
from .panoramic_geometry import PanoramicGeometry, DistortAlgorithm

logger = logging.getLogger(__name__)


class SeamAngles(BaseSettings):
    fov: Field[float] = Field(0.0, access=Field.READ, description="Camera FOV (°)")
    overlap: Field[float] = Field(0.0, access=Field.READ, description="Camera FOV overlap zone (°)")
    reject: Field[float] = Field(0.0, access=Field.READ, description="Dead zone at camera edges (°)")
    reach: Field[float] = Field(0.0, access=Field.READ, description="Cross-camera matching zone from camera edges (°)")


class SeamSettings(BaseSettings):
    reject: Field[float] = Field(0.5, min=0.0, max=0.75, step=0.05,
                                 description="Dead zone size as a fraction of the overlap zone.")
    reach: Field[float] = Field(1.3, min=1.0, max=1.5, step=0.05,
                                description="Matching zone size as a fraction of the overlap zone.")
    hysteresis: Field[float] = Field(0.9, min=0.1, max=1.0, step=0.05,
                                     description="Lower values make active camera stickier.")
    angles: Group[SeamAngles] = Group(SeamAngles)


class DistortionSettings(BaseSettings):
    algorithm: Field[DistortAlgorithm] = Field(DistortAlgorithm.NONE)
    k1: Field[float] = Field(0.0, min=-2.0, max=2.0, step=0.01)
    k2: Field[float] = Field(0.0, min=-2.0, max=2.0, step=0.01)


class PanoramicTrackerSettings(BaseSettings):
    fov: Field[float] = Field(110.0, min=90.0, max=130.0, step=0.5, visible=False)
    min_age: Field[int] = Field(5, min=0, max=9, step=1,
                              description="Minimum age in frames before a tracklet is considered.")
    min_height: Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05,
                                     description="Minimum ROI height to accept a tracklet.")
    timeout: Field[float] = Field(2.0, min=1.0, max=5.0, step=0.1,
                                  description="Seconds before an inactive tracklet is retired.")
    seam: Group[SeamSettings] = Group(SeamSettings)
    distortion: Group[DistortionSettings] = Group(DistortionSettings)


@dataclass(frozen=True)
class PanoramicAnnotation(TrackerAnnotation):
    local_angle: float
    world_angle: float
    overlap: bool


class PanoramicTracker(Thread, BaseTracker):
    """
    Tracks N people across a ring of cameras sharing a 360° field of view.

    Each camera runs its own on-device YOLO tracker and emits per-camera tracklets
    with stable local IDs. This class fuses those streams into world-space
    identities. Per-camera tracklet observations are stored as immutable records
    keyed by ``(cam_id, external_id)``: their ``cam_id``, ``external_id``,
    ``roi``, and ``annotation`` are never rewritten by fusion. A separate world
    id (drawn from the pool) groups one or more observations into a single
    identity emitted to callbacks.

    - **Cross-camera continuity**: when a brand-new observation arrives inside
      an overlap zone and exactly one existing world has a same-position match
      in another camera, the new observation is linked into that world. No
      merge, no rewrite — the person keeps the same id as they cross.
    - **View selection (primary)**: per world, the tracker emits one primary
      observation each tick. Selection is sticky with hysteresis (governed by
      ``seam.hysteresis``): the current primary stays primary unless a
      competitor's distance from the FOV edge exceeds it by the hysteresis
      ratio. Smooth handoff at seams without flicker.
    - **Late safety net**: if two genuine new arrivals at a seam each got their
      own world, a per-tick scan can collapse them into one via ``merge_worlds``.

    Processing runs in a background thread. Camera data is submitted via
    ``submit_cam_tracklets`` and results are delivered via registered callbacks.
    """

    def __init__(self, config: PanoramicTrackerSettings, num_players: int, num_cameras: int) -> None:
        super().__init__()

        self._running: bool = False
        self._update_event: Event = Event()

        self._max_players: int = num_players

        self._input_queue: Queue[Tracklet] = Queue()

        self.tracklet_manager: PanoramicTrackletManager = PanoramicTrackletManager(self._max_players)

        self.config: PanoramicTrackerSettings = config
        self.geometry: PanoramicGeometry = PanoramicGeometry(num_cameras, config.fov, 90.0)

        # Wire fov and distortion changes to geometry
        PanoramicTrackerSettings.fov.bind(config, lambda v: (self.geometry.set_fov(v), self._update_seam_angles()))
        DistortionSettings.algorithm.bind(config.distortion, lambda v: self.geometry.set_algorithm(v))
        DistortionSettings.k1.bind(config.distortion, lambda v: self.geometry.set_k1(v))
        DistortionSettings.k2.bind(config.distortion, lambda v: self.geometry.set_k2(v))

        # Wire seam ratio changes to the angles display
        SeamSettings.reject.bind(config.seam, lambda _: self._update_seam_angles())
        SeamSettings.reach.bind(config.seam, lambda _: self._update_seam_angles())
        self._update_seam_angles()

        # Last emitted primary per world id — view-selection state, used for hysteresis
        self._primary_for_world: dict[int, tuple[int, int]] = {}

        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletDictCallback] = set()

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

        # LOST/REMOVED: mark the observation lost and let timeout + cleanup handle removal
        if new_tracklet.is_lost or new_tracklet.is_removed:
            if self.tracklet_manager.get_world_id(cam_id, ext_id) is not None:
                self.tracklet_manager.lose_tracklet(cam_id, ext_id)
            return

        # Filter out tracklets that are too young or too small
        if new_tracklet.external_age_in_frames <= self.config.min_age:
            return
        if new_tracklet.roi.height < self.config.min_height:
            return

        # Annotate with local/world angles and overlap flag
        local_angle, world_angle, _overlap = self.geometry.get_angles_and_overlap(new_tracklet.roi, cam_id, self.config.seam.reject)
        new_tracklet = replace(new_tracklet, annotation=PanoramicAnnotation(local_angle, world_angle, _overlap))

        # Filter out tracklets too close to the FOV edge
        if self.geometry.angle_in_edge(local_angle, self.config.seam.reject):
            return

        # Existing observation — refresh in place
        if self.tracklet_manager.get_world_id(cam_id, ext_id) is not None:
            self.tracklet_manager.replace_tracklet(new_tracklet)
            return

        # Brand-new observation
        if not new_tracklet.is_active:
            return
        if _overlap:
            candidate_world: int | None = self._find_world_candidate(new_tracklet)
            if candidate_world is not None:
                self.tracklet_manager.add_tracklet(new_tracklet, world_id=candidate_world)
                return
        self.tracklet_manager.add_tracklet(new_tracklet)

    def _observations_match(self, a: Tracklet, b: Tracklet) -> bool:
        """True if two observations from different cameras are close enough in
        world angle and height to be considered the same person."""
        if a.cam_id == b.cam_id:
            return False
        if not isinstance(a.annotation, PanoramicAnnotation) or not isinstance(b.annotation, PanoramicAnnotation):
            return False
        if self.geometry.angle_diff(a.annotation.world_angle, b.annotation.world_angle) > self.geometry.fov_overlap * self.config.seam.reach:
            return False
        if abs(a.roi.height - b.roi.height) > 0.1:
            return False
        return True

    def _find_world_candidate(self, new_tracklet: Tracklet) -> int | None:
        """Return the unique world id whose other-camera observation matches
        ``new_tracklet`` in angle and height; None if zero or multiple match."""
        candidate_worlds: set[int] = set()
        for t in self.tracklet_manager.all_tracklets():
            if not t.is_active:
                continue
            if self._observations_match(new_tracklet, t):
                candidate_worlds.add(t.id)
        return next(iter(candidate_worlds)) if len(candidate_worlds) == 1 else None

    def _update_and_notify(self) -> None:
        # Expire timed-out observations
        for t in self.tracklet_manager.all_tracklets():
            if t.is_expired(self.config.timeout):
                self.tracklet_manager.retire_tracklet(t.cam_id, t.external_id)

        # Late safety net: collapse worlds whose observations match each other
        # (handles ambiguous simultaneous arrivals that each got their own world).
        for keep_id, drop_id in self._find_world_collapse_pairs():
            if self.tracklet_manager.merge_worlds(keep_id, drop_id):
                self._primary_for_world.pop(drop_id, None)

        # Emit one primary per world
        emitted: TrackletDict = {}
        for world_id in self.tracklet_manager.all_world_ids():
            primary: Tracklet | None = self._pick_primary(world_id)
            if primary is not None:
                emitted[world_id] = primary
        self._notify_callback(emitted)

        # Drop REMOVED observations and prune stale primary entries
        for t in self.tracklet_manager.all_tracklets():
            if t.status == TrackingStatus.REMOVED:
                self.tracklet_manager.remove_tracklet(t.cam_id, t.external_id)
        live_worlds: set[int] = set(self.tracklet_manager.all_world_ids())
        for world_id in list(self._primary_for_world):
            if world_id not in live_worlds:
                del self._primary_for_world[world_id]

    def _pick_primary(self, world_id: int) -> Tracklet | None:
        """Sticky primary selection with hysteresis to avoid per-tick flicker."""
        members: list[Tracklet] = self.tracklet_manager.get_tracklets(world_id)
        if not members:
            return None
        active: list[Tracklet] = [t for t in members if t.is_active and isinstance(t.annotation, PanoramicAnnotation)]
        if not active:
            chosen: Tracklet = max(members, key=lambda t: t.last_active)
            self._primary_for_world[world_id] = (chosen.cam_id, chosen.external_id)
            return chosen

        def edge(t: Tracklet) -> float:
            assert isinstance(t.annotation, PanoramicAnnotation)
            return self.geometry.angle_from_edge(t.annotation.local_angle)

        current_key: tuple[int, int] | None = self._primary_for_world.get(world_id)
        current: Tracklet | None = next((t for t in active if (t.cam_id, t.external_id) == current_key), None)
        if current is None:
            chosen = max(active, key=edge)
        else:
            best_competitor: Tracklet = max(active, key=edge)
            if best_competitor is current:
                chosen = current
            else:
                hysteresis: float = self.config.seam.hysteresis
                chosen = best_competitor if edge(best_competitor) >= edge(current) / hysteresis else current

        self._primary_for_world[world_id] = (chosen.cam_id, chosen.external_id)
        return chosen

    def _find_world_collapse_pairs(self) -> list[tuple[int, int]]:
        """Find world id pairs that should be collapsed because their observations
        match across cameras. Returns (keep_id, drop_id) pairs; older world wins.
        Each world id appears in at most one pair to avoid collapsing into a
        world that is itself about to be dropped."""
        observations: list[Tracklet] = [
            t for t in self.tracklet_manager.all_tracklets()
            if t.is_active
            and isinstance(t.annotation, PanoramicAnnotation)
            and self.geometry.angle_in_overlap(t.annotation.local_angle, self.config.seam.reach - 1.0)
        ]

        used: set[int] = set()  # world ids already committed to a pair
        pairs: list[tuple[int, int]] = []
        for a, b in combinations(observations, 2):
            if a.id == b.id:
                continue
            if a.id in used or b.id in used:
                continue
            if not self._observations_match(a, b):
                continue
            # Older world wins
            members_a: list[Tracklet] = self.tracklet_manager.get_tracklets(a.id)
            members_b: list[Tracklet] = self.tracklet_manager.get_tracklets(b.id)
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

    def add_tracklet_callback(self, callback: TrackletDictCallback) -> None:
        with self._callback_lock:
            self._tracklet_callbacks.add(callback)

    def submit_cam_tracklets(self, cam_id: int, cam_tracklets: list[DepthTracklet]) -> None:
        for t in cam_tracklets:
            tracklet: Tracklet | None = Tracklet.from_depthcam(cam_id, t)
            if tracklet is None:
                logger.warning(f"PanoramicTracker: Invalid tracklet from camera {cam_id}, skipping.")
                continue
            self._input_queue.put(tracklet)
