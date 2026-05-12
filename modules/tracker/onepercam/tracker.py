# Standard library imports
import logging
from queue import Empty, Queue
from threading import Lock, Thread, Event

# Local application imports
from modules.oak import DepthTracklet
from modules.settings import BaseSettings, Field
from .. import (
    BaseTracker,
    Tracklet, TrackingStatus, TrackletDict, TrackletDictCallback,
)
from .store import TrackletStore

logger = logging.getLogger(__name__)


class TrackerSettings(BaseSettings):
    tracklet_min_age:         Field[int]   = Field(3,    min=0,   max=9,   step=1)
    add_centre_threshold:     Field[float] = Field(0.15, min=0.0, max=1.0, step=0.05)
    update_centre_threshold:  Field[float] = Field(0.3,  min=0.0, max=1.0, step=0.05)
    timeout:                  Field[float] = Field(2.0,  min=1.0, max=5.0, step=0.1)
    add_height_threshold:     Field[float] = Field(0.5,  min=0.0, max=1.0, step=0.05)
    update_height_threshold:  Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05)
    add_bottom_threshold:     Field[float] = Field(0.2,  min=0.0, max=1.0, step=0.05)


class Tracker(Thread, BaseTracker):
    def __init__(self, config: TrackerSettings, num_trackers: int) -> None:
        super().__init__()

        self._running: bool = False
        self._update_event: Event = Event()
        self._input_queue: Queue[list[Tracklet]] = Queue()
        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletDictCallback] = set()

        self._num_cams: int = num_trackers
        self.config: TrackerSettings = config

        self.store: TrackletStore = TrackletStore(self._num_cams)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        super().start()

    def stop(self) -> None:
        self._running = False
        with self._callback_lock:
            self._tracklet_callbacks.clear()
        self.join(timeout=1.0)

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
                        tracklets: list[Tracklet] = self._input_queue.get(block=False)
                        self._add_tracklet(tracklets)
                    except Empty:
                        break

                self._update_and_notify()
            except Exception:
                logger.exception("OnePerCamTracker error")

    def _add_tracklet(self, tracklets: list[Tracklet]) -> None:
        # UPDATE EXISTING TRACKLET
        update_tracklet: Tracklet | None = None
        for t in tracklets:
            if t in self.store:
                update_tracklet = t
                break

        if update_tracklet is not None:
            if update_tracklet.is_removed:
                self.store.retire_tracklet(update_tracklet.cam_id)
                return

            # if tracklet is LOST, too small or too far from centre, lose it
            if (
                update_tracklet.is_lost or
                update_tracklet.roi.height < self.config.update_height_threshold or
                abs(update_tracklet.roi.center.x - 0.5) > self.config.update_centre_threshold
            ):
                self.store.lose_tracklet(update_tracklet.cam_id)
                return

            # else update the tracklet
            self.store.replace_tracklet(update_tracklet.cam_id, update_tracklet)
            return

        # ADD TRACKLET
        add_tracklets: list[Tracklet] = []
        for t in tracklets:

            if t.external_age_in_frames <= self.config.tracklet_min_age:
                continue

            if not t.is_active:
                continue

            if abs(t.roi.center.x - 0.5) > self.config.add_centre_threshold:
                continue

            if t.roi.height < self.config.add_height_threshold:
                continue

            if t.roi.bottom < 1.0 - self.config.add_bottom_threshold:
                continue

            add_tracklets.append(t)

        if not add_tracklets:
            return

        add_tracklets = sorted(add_tracklets, key=lambda x: x.external_age_in_frames, reverse=True)
        add_tracklet: Tracklet = add_tracklets[0]
        self.store.add_tracklet(add_tracklet)

    def _update_and_notify(self) -> None:
        # retire expired tracklets
        for tracklet in self.store.all_tracklets():
            if tracklet.is_expired(self.config.timeout):
                self.store.retire_tracklet(tracklet.id)

        # Notify callbacks
        callback_tracklets: TrackletDict = {}
        for tracklet in self.store.all_tracklets():
            callback_tracklets[tracklet.id] = tracklet
        self._notify_callback(callback_tracklets)

        # Remove expired tracklets
        for tracklet in self.store.all_tracklets():
            if tracklet.status == TrackingStatus.REMOVED:
                self.store.remove_tracklet(tracklet.id)

    def _notify_callback(self, tracklets: TrackletDict) -> None:
        with self._callback_lock:
            for callback in self._tracklet_callbacks:
                callback(tracklets)

    def add_tracklet_callback(self, callback: TrackletDictCallback) -> None:
        with self._callback_lock:
            self._tracklet_callbacks.add(callback)

    def submit_cam_tracklets(self, cam_id: int, cam_tracklets: list[DepthTracklet]) -> None:
        tracklet_list: list[Tracklet] = []
        for t in cam_tracklets:
            tracklet: Tracklet | None = Tracklet.from_depthcam(cam_id, t)
            if tracklet is not None:
                tracklet_list.append(tracklet)

        if tracklet_list:
            self._input_queue.put(tracklet_list)
