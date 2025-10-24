# Standard library imports
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread, Event
from time import sleep, time
from typing import List, Optional

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet
from modules.Settings import Settings
from modules.tracker.Tracklet import Tracklet, TrackletCallback, TrackingStatus
from modules.tracker.TrackerBase import BaseTracker, TrackerType, TrackerMetadata
from modules.tracker.onepercam.OnePerCamSmoothRect import OnePerCamSmoothRect
from modules.tracker.onepercam.OnePerCamTrackletManager import OnePerCamTrackletManager as TrackletManager
from modules.tracker.onepercam.OnePerCamTrackerGui import OnePerCamTrackerGui

from modules.utils.PointsAndRects import Rect, Point2f
from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass (frozen=True)
class OnePerCamMetadata(TrackerMetadata):
    smooth_rect: Rect

    @property
    def tracker_type(self) -> TrackerType:
        return TrackerType.ONEPERCAM

class OnePerCamTracker(Thread, BaseTracker):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self._running: bool = False
        self._update_event: Event = Event()
        self._input_queue: Queue[List[Tracklet]] = Queue()
        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletCallback] = set()

        self._num_cams: int =               settings.camera_num
        self.tracklet_min_age: int =        settings.tracker_min_age
        self.timeout: float =               settings.tracker_timeout

        self.update_centre_threshold: float =  0.3
        self.update_height_treshold: float =   0.25

        self.add_centre_threshold: float =     0.15
        self.add_height_threshold: float =     0.5
        self.add_bottom_threshold: float =     0.2

        self.tracklet_manager: TrackletManager = TrackletManager(self._num_cams)
        self.smooth_rects: OnePerCamSmoothRect = OnePerCamSmoothRect(self._num_cams)

        self.gui = OnePerCamTrackerGui(gui, self, settings)

        hot_reload = HotReloadMethods(self.__class__)

    @property
    def tracker_type(self) -> TrackerType:
        return TrackerType.ONEPERCAM

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

            while True:
                try:
                    tracklets: list[Tracklet] = self._input_queue.get(timeout=0.001)
                    self._add_tracklet(tracklets)
                except Empty:
                    break

            self._update_and_notify()

    def _add_tracklet(self, tracklets: list[Tracklet]) -> None:

        # UPDATE EXISTING TRACKLET
        update_tracklet: Optional[Tracklet] = None
        for t in tracklets:
            if t in self.tracklet_manager:
                update_tracklet = t
                break

        if update_tracklet is not None:
            if update_tracklet.is_removed:
                self.tracklet_manager.retire_tracklet(update_tracklet.cam_id)
                return

            # if tracklet is LOST, too small or too far from centre, lose it
            if (
                update_tracklet.is_lost or
                update_tracklet.roi.height < self.update_height_treshold or
                abs(update_tracklet.roi.center.x - 0.5) > self.update_centre_threshold
            ):
                self.tracklet_manager.lose_tracklet(update_tracklet.cam_id)
                return

            # else update the tracklet
            self.tracklet_manager.replace_tracklet(update_tracklet.cam_id, update_tracklet)
            return

        # ADD TRACKLET
        add_tracklets: List[Tracklet] = []
        for t in tracklets:

            if t.external_age_in_frames <= self.tracklet_min_age:
                continue

            if not t.is_active:
                continue

            if abs(t.roi.center.x - 0.5) > self.add_centre_threshold:
                continue

            if t.roi.height < self.add_height_threshold:
                continue

            if t.roi.bottom < 1.0 - self.add_bottom_threshold:
                continue

            add_tracklets.append(t)

        if not add_tracklets:
            return

        add_tracklets = sorted(add_tracklets, key=lambda x: x.external_age_in_frames, reverse=True)
        add_tracklet: Tracklet = add_tracklets[0]
        self.tracklet_manager.add_tracklet(add_tracklet)

    def _update_and_notify(self) -> None:
        # retire expired tracklets
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.is_expired(self.timeout):
                self.tracklet_manager.retire_tracklet(tracklet.id)
                # print(f"Retiring expired tracklet {tracklet.id} at {time()}")

        # Create metadata for each tracklet
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.needs_notification:
                smooth_rect = self.smooth_rects.update(tracklet)
                if smooth_rect is not None:
                    metadata = OnePerCamMetadata(
                        smooth_rect=smooth_rect
                    )
                    self.tracklet_manager.set_metadata(tracklet.id, metadata)

        # Notify callbacks
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.needs_notification:
                self._notify_callback(tracklet)

        self.tracklet_manager.mark_all_as_notified()

        # Remove expired tracklets
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.status == TrackingStatus.REMOVED:
                self.tracklet_manager.remove_tracklet(tracklet.id)

    def _notify_callback(self, tracklet: Tracklet) -> None:
        with self._callback_lock:
            for callback in self._tracklet_callbacks:
                callback(tracklet)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        with self._callback_lock:
            self._tracklet_callbacks.add(callback)

    def add_cam_tracklets(self, cam_id: int, cam_tracklets: List[DepthTracklet]) -> None:
        tracklet_list: List[Tracklet] = []
        for t in cam_tracklets:
            tracklet: Optional[Tracklet] = Tracklet.from_depthcam(cam_id, t)
            if tracklet is not None:
                tracklet_list.append(tracklet)

        if tracklet_list:
            self._input_queue.put(tracklet_list)
