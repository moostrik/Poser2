# Standard library imports
from dataclasses import dataclass, replace
from itertools import combinations
from queue import Empty, Queue
from threading import Lock, Thread, Event
from time import sleep, time
from typing import Optional

# Local application imports
from modules.oak.camera.definitions import Tracklet as DepthTracklet
from modules.tracker.TrackerBase import BaseTracker, TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackingStatus, TrackletDict, TrackletDictCallback
from modules.tracker.panoramic.PanoramicTrackletManager import PanoramicTrackletManager
from modules.tracker.panoramic.PanoramicGeometry import PanoramicGeometry
from modules.tracker.panoramic.PanoramicDefinitions import *
from modules.settings import BaseSettings, Field

import logging
logger = logging.getLogger(__name__)


class PanoramicTrackerSettings(BaseSettings):
    """Configuration for PanoramicTracker."""
    fov:                      Field[float] = Field(CAM_360_FOV,              min=90.0, max=130.0, step=0.5)
    tracklet_min_age:         Field[int]   = Field(5,                        min=0,    max=9,     step=1)
    tracklet_min_height:      Field[float] = Field(0.25,                     min=0.0,  max=1.0,   step=0.05)
    timeout:                  Field[float] = Field(2.0,                      min=1.0,  max=5.0,   step=0.1)
    cam_360_edge_threshold:   Field[float] = Field(CAM_360_EDGE_THRESHOLD,   min=0.0,  max=0.6,   step=0.1)
    cam_360_overlap_expansion:Field[float] = Field(CAM_360_OVERLAP_EXPANSION,min=0.0,  max=1.0,   step=0.1)
    cam_360_hysteresis_factor:Field[float] = Field(CAM_360_HYSTERESIS_FACTOR)


@dataclass (frozen=True)
class PanoramicMetadata(TrackerMetadata):
    local_angle: float
    world_angle: float
    overlap: bool

    @property
    def tracker_type(self) -> TrackerType:
        return TrackerType.PANORAMIC

@dataclass(frozen=True)
class PanoramicOverlapInfo:
    keep_id: int
    remove_id: int
    distance: float
    reason: str

class PanoramicTracker(Thread, BaseTracker):
    def __init__(self, config: PanoramicTrackerSettings, num_players: int, num_cameras: int) -> None:
        super().__init__()

        self.running: bool = False
        self.update_event: Event = Event()

        self.max_players: int = num_players

        self.input_queue: Queue[Tracklet] = Queue()

        self.tracklet_manager: PanoramicTrackletManager = PanoramicTrackletManager(self.max_players)

        self.config: PanoramicTrackerSettings = config
        self.geometry: PanoramicGeometry = PanoramicGeometry(num_cameras, config.fov, CAM_360_TARGET_FOV)

        # Wire fov changes to geometry
        PanoramicTrackerSettings.fov.bind(config, lambda v: self.geometry.set_fov(v))

        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletDictCallback] = set()

    @property
    def tracker_type(self) -> TrackerType:
        return TrackerType.PANORAMIC

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        super().start()

    def stop(self) -> None:
        self.running = False

        with self._callback_lock:
            self._tracklet_callbacks.clear()

        self.join()  # Wait for the thread to finish

    def notify_update(self) -> None:
        if self.running:
            self.update_event.set()

    def run(self) -> None:
        while self.running:
            self.update_event.wait(timeout=0.1)
            self.update_event.clear()

            try:
                while True:
                    try:
                        tracklet: Tracklet = self.input_queue.get(block=False)
                        self._add_tracklet(tracklet)
                    except Empty:
                        break

                self._update_and_notify()
            except Exception:
                logger.exception("PanoramicTracker error")

    def _add_tracklet(self, new_tracklet: Tracklet) -> None:
        # If tracklet is removed, retire it
        if new_tracklet.is_removed:
            existing_tracklet_id: Optional[int] = self.tracklet_manager.get_id_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
            if existing_tracklet_id is not None:
                self.tracklet_manager.retire_tracklet(existing_tracklet_id)
            return
        if new_tracklet.is_lost:
            existing_tracklet_id: Optional[int] = self.tracklet_manager.get_id_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
            if existing_tracklet_id is not None:
                self.tracklet_manager.replace_tracklet(existing_tracklet_id, new_tracklet)
            return

        # filter out tracklets that are too young or too small
        if new_tracklet.external_age_in_frames <= self.config.tracklet_min_age:
            return
        if new_tracklet.roi.height < self.config.tracklet_min_height:
            return

        # Construct PanoramicTrackerInfo with local and world angles
        local_angle, world_angle, _overlap = self.geometry.get_angles_and_overlap(new_tracklet.roi, new_tracklet.cam_id, self.config.cam_360_edge_threshold)
        new_tracklet = replace(new_tracklet, metadata=PanoramicMetadata(local_angle, world_angle, _overlap))

        # Filter out tracklets that are too close to the edge of a camera's field of view
        if self.geometry.angle_in_edge(local_angle, self.config.cam_360_edge_threshold):
            return

        # Check if the new tracklet already exists in the tracker and replace
        existing_tracklet_id: Optional[int] = self.tracklet_manager.get_id_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
        if existing_tracklet_id is not None:
            self.tracklet_manager.replace_tracklet(existing_tracklet_id, new_tracklet)

        # If it doesn't exist, add it as a new tracklet
        elif new_tracklet.is_active:
            self.tracklet_manager.add_tracklet(new_tracklet)

    def _update_and_notify(self) -> None:
        # expire timed-out tracklets
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.is_expired(self.config.timeout):
                self.tracklet_manager.lose_tracklet(tracklet.id)

        # merge overlapping tracklets
        overlaps: set[PanoramicOverlapInfo] = self._find_overlapping_tracklets(self.tracklet_manager.all_tracklets())
        for overlap in overlaps:
            self.tracklet_manager.merge_tracklets(overlap.keep_id, overlap.remove_id)

        # notify all active tracklets + any REMOVED ones (so downstream clears their slots)
        callback_tracklets: TrackletDict = {}
        for tracklet in self.tracklet_manager.all_tracklets():
            callback_tracklets[tracklet.id] = tracklet
        self._notify_callback(callback_tracklets)

        self.tracklet_manager.mark_all_as_notified()

        # remove REMOVED tracklets after notification
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.status == TrackingStatus.REMOVED:
                self.tracklet_manager.remove_tracklet(tracklet.id)

    def _find_overlapping_tracklets(self, tracklets: list[Tracklet]) -> set[PanoramicOverlapInfo]:

        for i, tracklet in enumerate(tracklets):
            # Reconstruct PanoramicTrackerInfo with updated overlap field
            if isinstance(tracklet.metadata, PanoramicMetadata):
                updated_overlap = self.geometry.angle_in_overlap(tracklet.metadata.local_angle, self.config.cam_360_overlap_expansion)

                tracklets[i] = replace(
                    tracklet,
                    metadata=PanoramicMetadata(
                        local_angle=tracklet.metadata.local_angle,
                        world_angle=tracklet.metadata.world_angle,
                        overlap=updated_overlap
                    )
                )

        overlaps: list[PanoramicOverlapInfo] = []

        for P_A, P_B in combinations(tracklets, 2):
            if not getattr(P_A.metadata, "overlap", False) or not getattr(P_B.metadata, "overlap", False):
                continue
            if P_A.cam_id == P_B.cam_id:
                continue

            angle_diff: float = self.geometry.angle_diff(getattr(P_A.metadata, "world_angle", 45.0), getattr(P_B.metadata, "world_angle", 45.0))
            if angle_diff > self.geometry.fov_overlap * (1.0 + self.config.cam_360_overlap_expansion):
                continue

            # look at the hight of the trackets for extra filtering
            height_diff: float = abs(P_A.roi.height - P_B.roi.height)
            if height_diff > 0.1:
                continue

            if not P_A.is_active and P_B.is_active:
                overlaps.append(PanoramicOverlapInfo(P_B.id, P_A.id, angle_diff, f'{P_A.id} inactive'))
                continue

            elif not P_B.is_active and P_A.is_active:
                overlaps.append(PanoramicOverlapInfo(P_A.id, P_B.id, angle_diff, f'{P_B.id} inactive'))
                continue

            if not P_A.is_active and not P_B.is_active:
                continue

            if P_A.age_in_seconds < P_B.age_in_seconds:
                newest, oldest = P_A, P_B
            else:
                newest, oldest = P_B, P_A
            edge_newest: float = self.geometry.angle_from_edge(getattr(newest.metadata, "local_angle", 45.0))
            edge_oldest: float = self.geometry.angle_from_edge(getattr(oldest.metadata, "local_angle", 45.0))
            # print(f"Comparing newest {newest.id} (edge {edge_newest}) to oldest {oldest.id} (edge {edge_oldest})")
            if edge_newest >= edge_oldest / self.config.cam_360_hysteresis_factor:
                overlaps.append(PanoramicOverlapInfo(newest.id, oldest.id, angle_diff, f'{edge_newest} >= {edge_oldest}'))
            else:
                overlaps.append(PanoramicOverlapInfo(oldest.id, newest.id, angle_diff,  f'{edge_newest} < {edge_oldest}'))

        overlap_sets: set[PanoramicOverlapInfo] = set(overlaps)
        return overlap_sets

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
            tracklet: Optional[Tracklet] = Tracklet.from_depthcam(cam_id, t)
            if tracklet is None:
                logger.warning(f"PanoramicTracker: Invalid tracklet from camera {cam_id}, skipping.")
                continue
            self.input_queue.put(tracklet)
