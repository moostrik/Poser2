# Standard library imports
import logging
import time
from dataclasses import replace
from queue import Empty, Queue
from threading import Lock, Thread, Event

# Local application imports
from modules.oak import DepthTracklet
from ..tracker_base import BaseTracker
from ..tracklet import Tracklet, TrackingStatus, TrackletDict, TrackletDictCallback, TrackletListCallback
from .annotation import Annotation, Rejection
from .observations import ObservationStore
from .rig import Rig
from .rig_sync import RigSync
from .seams import Seams
from .settings import RigSettings, TrackerSettings

logger = logging.getLogger(__name__)


class Tracker(Thread, BaseTracker):
    """Tracks people across a ring of cameras sharing a 360° field of view.

    Each camera's on-device tracker sends per-camera tracklets; this joins them into world
    identities. The parts:

    - `Rig` models the installation and turns a box into an `Annotation` (local angle, world
      azimuth, distance, height); `RigSync` keeps it and the published read-outs in step with
      `RigSettings`.
    - `ObservationStore` holds the per-camera observations — immutable, host-owned ids — grouped
      into world ids.
    - `Seams` applies `SeamSettings`: linking a new view across a seam, collapsing worlds that are
      one person, and handing the primary view over between cameras.

    This class runs the thread: the intake (filters, same-camera re-acquisition, the dead zone, then
    the seam link), the per-tick expiry and emission, and the callbacks. Two output channels:
    ``add_tracklet_callback`` delivers one primary per remembered world — LOST only when no camera
    sees the person; freshness is the consumer's call — and ``add_observation_callback`` every
    observation plus every rejected detection, for the panorama.
    """

    def __init__(self, config: TrackerSettings, num_players: int, num_cameras: int) -> None:
        super().__init__()

        self._running: bool = False
        self._update_event: Event = Event()

        self._input_queue: Queue[Tracklet] = Queue()

        self.observations: ObservationStore = ObservationStore(num_players)

        self.config: TrackerSettings = config
        # Each camera owns 360/num_cameras degrees of the ring; whatever its field has beyond
        # that is the overlap it shares with its neighbours.
        self.rig: Rig = Rig(config.fov, 360.0 / num_cameras)
        self.rig_sync: RigSync = RigSync(config, self.rig)
        self.seams: Seams = Seams(self.observations, self.rig, config)

        # A live rig change only raises this; the tracker thread applies it before its next tick, so
        # the `Rig` is never written while the intake reads it.
        self._rig_changed: Event = Event()
        self._rig_fields = (
            (config.rig, RigSettings.camera_radius),
            (config.rig, RigSettings.camera_height),
            (config.rig, RigSettings.zone_min_radius),
            (config.rig, RigSettings.zone_max_radius),
            (config, TrackerSettings.foot_offset),
        )
        for settings, field in self._rig_fields:
            settings.bind(field, self._on_rig_changed)

        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletDictCallback] = set()
        self._observation_callbacks: set[TrackletListCallback] = set()

        # Detections the intake rejected, latest per device track, with their `Rejection` — published
        # on the observation channel so the panorama can draw them, never on the primary channel.
        # Touched only on the tracker thread (`_add_tracklet` and `_update_and_notify`).
        self._rejected: dict[tuple[int, int], Tracklet] = {}
        # Set while births are refused for want of a world id, so that is logged once, not per frame.
        self._pool_full: bool = False

    def column_to_azimuth(self, cam_id: int, x: float) -> float:
        """World azimuth (degrees, [0, 360)) of a normalised column of camera `cam_id`, through the
        same chain the tracker's own azimuths come from (`Rig.column_to_azimuth`)."""
        return self.rig.column_to_azimuth(cam_id, x)

    # LIFECYCLE
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        super().start()

    def stop(self) -> None:
        self._running = False

        for settings, field in self._rig_fields:
            settings.unbind(field, self._on_rig_changed)

        with self._callback_lock:
            self._tracklet_callbacks.clear()
            self._observation_callbacks.clear()

        if self.is_alive():
            self.join(timeout=1.0)  # Wait for the thread to finish

    def notify_update(self) -> None:
        if self._running:
            self._update_event.set()

    def run(self) -> None:
        while self._running:
            self._update_event.wait(timeout=0.1)
            self._update_event.clear()

            try:
                if self._rig_changed.is_set():
                    self._rig_changed.clear()
                    self._apply_settings()

                while True:
                    try:
                        tracklet: Tracklet = self._input_queue.get(block=False)
                        self._add_tracklet(tracklet)
                    except Empty:
                        break

                self._update_and_notify()
            except Exception:
                logger.exception("PanoramicTracker error")

    def _on_rig_changed(self, _value: object) -> None:
        self._rig_changed.set()

    def _apply_settings(self) -> None:
        """Bring the `Rig` and the published read-outs up to the current rig settings."""
        self.rig_sync.apply()

    # INTAKE
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
            if self.observations.world_of(cam_id, ext_id) is not None:
                self.observations.end_device_track(cam_id, ext_id)
            return

        # No detection this frame, but the device still holds the track: the same id will come
        # back, so the observation stays live.
        if new_tracklet.is_lost:
            self._rejected.pop(key, None)
            if self.observations.world_of(cam_id, ext_id) is not None:
                self.observations.lose(cam_id, ext_id)
            return

        # A status the device mapping does not know (`TrackingStatus.NONE`) says nothing usable.
        if not new_tracklet.is_active:
            return

        # Annotate first, before any filter: a detection a filter rejects is still drawn on the
        # panorama, labelled with the reason, and it needs its angles for that. Pure geometry, so the
        # order changes nothing that is decided below.
        annotation: Annotation = self.rig.annotate(new_tracklet.roi, cam_id)
        new_tracklet = replace(new_tracklet, annotation=annotation)
        existing: bool = self.observations.world_of(cam_id, ext_id) is not None

        # Too young, too small, or past the zone's far edge: not counted — handled exactly like a
        # missed detection, before every branch below, so nobody is born, re-acquired or linked on
        # it. Someone already tracked goes LOST: their pose carries on from their last box for
        # `pose.tracklets.detection_timeout`, so a jump or a moment of hidden feet changes nothing
        # visible; forgotten after `lost_timeout`; the same person again if they are counted before
        # that. Their latest box is kept with its rejection, so the panorama's mark follows them and says why it
        # is fading.
        reason: Rejection | None = None
        if new_tracklet.external_age_in_frames <= self.config.age_filter:
            reason = Rejection.YOUNG
        elif new_tracklet.roi.height < self.config.height_filter:
            reason = Rejection.SMALL
        elif self.config.zone_filter and self.rig.beyond_zone(annotation.local_angle, annotation.distance):
            reason = Rejection.PAST_EDGE
        if reason is not None:
            if existing:
                self._rejected.pop(key, None)
                self.observations.lose(cam_id, ext_id, latest=self._with_rejection(new_tracklet, reason))
            else:
                self._reject(new_tracklet, reason)
            return

        # Existing observation — refresh in place, even inside the edge dead
        # zone: starving it would freeze its angles and expire it via lost_timeout
        # while the camera still tracks the person.
        if existing:
            self._rejected.pop(key, None)
            self.observations.refresh(new_tracklet)
            return

        # A re-acquisition in the same camera is a continuation, not an arrival, so it is
        # allowed anywhere in frame — including the edge dead zone, which only exists to stop
        # *new* people being born on a seam.
        anchor_world: int | None = self._reacquired_world(new_tracklet)
        if anchor_world is not None:
            self._rejected.pop(key, None)
            self.observations.add(new_tracklet, world_id=anchor_world)
            return

        # Brand-new observations are ignored too close to the FOV edge
        if self.rig.angle_in_edge(annotation.local_angle, self.config.seam.dead_zone):
            self._reject(new_tracklet, Rejection.DEAD_ZONE)
            return
        if annotation.overlap:
            linked_world: int | None = self.seams.linked_world(new_tracklet)
            if linked_world is not None:
                self._rejected.pop(key, None)
                self.observations.add(new_tracklet, world_id=linked_world)
                return

        # A new person. Checked here rather than read off `add_tracklet`'s None, which cannot say why.
        if not self.observations.has_free_id():
            if not self._pool_full:
                logger.warning("Every world id is in use: new people are not tracked until one frees up")
                self._pool_full = True
            self._reject(new_tracklet, Rejection.NO_ID)
            return
        self._pool_full = False
        self._rejected.pop(key, None)
        self.observations.add(new_tracklet)

    def _reacquired_world(self, new_tracklet: Tracklet) -> int | None:
        """The world of a lost observation in the SAME camera that this new one continues.

        The device tracker has no appearance model, so a person it drops returns under a new id.
        Matching on position makes that continuity our rule. Bounded by ``reacquire_angle`` alone —
        a dropped person's box usually changed, so requiring it to agree asks the wrong question.
        """
        assert isinstance(new_tracklet.annotation, Annotation)
        new_angle: float = new_tracklet.annotation.local_angle
        best_world: int | None = None
        best_diff: float = float('inf')
        for t in self.observations.all():
            if t.cam_id != new_tracklet.cam_id or t.is_active or t.is_removed:
                continue
            if not isinstance(t.annotation, Annotation):
                continue
            if self.observations.camera_sees_world(new_tracklet.cam_id, t.id):
                continue                    # already re-found: this is someone else
            diff: float = abs(t.annotation.local_angle - new_angle)
            if diff > self.config.reacquire_angle:
                continue
            if diff < best_diff:
                best_diff = diff
                best_world = t.id
        return best_world

    @staticmethod
    def _with_rejection(tracklet: Tracklet, reason: Rejection) -> Tracklet:
        assert isinstance(tracklet.annotation, Annotation)
        return replace(tracklet, annotation=replace(tracklet.annotation, rejected=reason))

    def _reject(self, tracklet: Tracklet, reason: Rejection) -> None:
        """Not counted this frame: keep the detection, with its rejection, for the panorama."""
        self._rejected[(tracklet.cam_id, tracklet.external_id)] = self._with_rejection(tracklet, reason)

    # TICK
    def _update_and_notify(self) -> None:
        now: float = time.time()

        # Expire timed-out observations
        for t in self.observations.all():
            if t.is_expired(self.config.lost_timeout):
                self.observations.retire(t.obs_id)

        self.seams.collapse_worlds()

        # Emit one primary per world the tracker still remembers until `lost_timeout` retires it —
        # LOST only when no camera sees the person. How stale is too stale is each consumer's call,
        # not this one's: pose stops posing a person after `pose.tracklets.detection_timeout`, and
        # the show counts only active tracklets. Filtering here instead would decide for all of them
        # with one number. A world retired just above is still in the store until the end of this
        # tick; it is not emitted.
        emitted: TrackletDict = {}
        for world_id in self.observations.world_ids():
            primary: Tracklet | None = self.seams.pick_primary(world_id, now)
            if primary is not None:
                emitted[world_id] = primary
        self._notify_callback(emitted)

        # Dropped detections a camera has stopped reporting: a device track normally ends with LOST
        # or REMOVED, which clears its entry, but a camera that simply goes quiet does not.
        lost_timeout: float = self.config.lost_timeout
        for key in [k for k, t in self._rejected.items() if now - t.last_active > lost_timeout]:
            del self._rejected[key]

        # Every live observation, for the calibration view: the two cameras' separate opinions
        # of a person on a seam, which the primaries above deliberately reduce to one — and every
        # detection a filter rejected, so nobody vanishes from the strip without a reason.
        self._notify_observation_callback(
            [t for t in self.observations.all() if not t.is_removed] + list(self._rejected.values())
        )

        # Drop REMOVED observations and the handover state of worlds that went with them
        for t in self.observations.all():
            if t.status == TrackingStatus.REMOVED:
                self.observations.remove(t.obs_id)
        self.seams.prune(set(self.observations.world_ids()))

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
