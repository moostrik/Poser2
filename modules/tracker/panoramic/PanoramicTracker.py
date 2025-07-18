from __future__ import annotations

# Standard library imports
import time
from dataclasses import dataclass, replace
from itertools import combinations
from queue import Empty, Queue
from threading import Thread, Lock
from typing import Optional

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet as CamTracklet
from modules.tracker.BaseTracker import BaseTracker, TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletCallback, TrackingStatus
from modules.tracker.panoramic.PanoramicTrackletManager import PanoramicTrackletManager
from modules.tracker.panoramic.PanoramicTrackerGui import PanoramicTrackerGui
from modules.tracker.panoramic.PanoramicGeometry import PanoramicGeometry
from modules.tracker.panoramic.PanoramicDefinitions import *
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


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
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.running: bool = False
        self.max_players: int = settings.max_players
        self.update_interval: float = 0.5 / settings.camera_fps # Run update loop faster than camera FPS to avoid missing frames due to timing jitter

        self.input_queue: Queue[Tracklet] = Queue()

        self.tracklet_manager: PanoramicTrackletManager = PanoramicTrackletManager(self.max_players)

        self.geometry: PanoramicGeometry = PanoramicGeometry(settings.camera_num, CAM_360_FOV, CAM_360_TARGET_FOV)

        self.tracklet_min_age: int =            settings.tracker_min_age
        self.tracklet_min_height: float =       settings.tracker_min_height
        self.timeout: float =                   settings.tracker_timeout
        self.cam_360_edge_threshold: float =    CAM_360_EDGE_THRESHOLD
        self.cam_360_overlap_expansion: float = CAM_360_OVERLAP_EXPANSION
        self.cam_360_hysteresis_factor: float = CAM_360_HYSTERESIS_FACTOR

        self.callback_lock = Lock()
        self.tracklet_callbacks: set[TrackletCallback] = set()
        self.gui = PanoramicTrackerGui(gui, self, settings)

        hot_reload = HotReloadMethods(self.__class__)

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

        with self.callback_lock:
            self.tracklet_callbacks.clear()

        self.join()  # Wait for the thread to finish

    def run(self) -> None:
        next_update_time: float = time.time() + self.update_interval

        while self.running:
            current_time: float = time.time()

            # Process all available tracklets as fast as possible
            processed_any = False
            while True:
                try:
                    tracklet: Tracklet = self.input_queue.get(timeout=0.001)  # Very short timeout
                    self._add_tracklet(tracklet)
                    processed_any = True
                except Empty:
                    break

            # Update and notify at precise intervals
            if current_time >= next_update_time:
                self._update_and_notify()
                next_update_time += self.update_interval
                # In case of drift, catch up
                while current_time > next_update_time:
                    next_update_time = current_time + self.update_interval

            # Small sleep to prevent excessive CPU usage when no tracklets are available
            if not processed_any:
                time.sleep(0.001)  # 1ms sleep

    def _update_and_notify(self) -> None:
        self._update_tracklets()
        self._notify_and_reset_changes()
        self._remove_expired_tracklets()

    def _add_tracklet(self, new_tracklet: Tracklet) -> None:
        # If tracklet is removed, retire it
        if new_tracklet.is_removed:
            existing_tracklet_id: Optional[int] = self.tracklet_manager.get_id_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
            if existing_tracklet_id is not None:
                self.tracklet_manager.retire_tracklet(existing_tracklet_id)
            return

        # filter out tracklets that are too young or too small
        if new_tracklet.external_age_in_frames <= self.tracklet_min_age:
            return
        if new_tracklet.roi.height < self.tracklet_min_height:
            return

        # Construct PanoramicTrackerInfo with local and world angles
        local_angle, world_angle, _overlap = self.geometry.get_angles_and_overlap(new_tracklet.roi, new_tracklet.cam_id, self.cam_360_edge_threshold)
        new_tracklet = replace(new_tracklet, metadata=PanoramicMetadata(local_angle, world_angle, _overlap))

        # Filter out tracklets that are too close to the edge of a camera's field of view
        if self.geometry.angle_in_edge(local_angle, self.cam_360_edge_threshold):
            return

        # Check if the new tracklet already exists in the tracker and replace
        existing_tracklet_id: Optional[int] = self.tracklet_manager.get_id_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
        if existing_tracklet_id is not None:
            self.tracklet_manager.replace_tracklet(existing_tracklet_id, new_tracklet)

        # If it doesn't exist, add it as a new tracklet
        elif new_tracklet.is_active:
            self.tracklet_manager.add_tracklet(new_tracklet)

    def _update_tracklets(self) -> None:
        # retire expired tracklets
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.is_expired(self.timeout):
                self.tracklet_manager.retire_tracklet(tracklet.id)

        # merge overlapping tracklets
        overlaps: set[PanoramicOverlapInfo] = self.find_overlapping_tracklets(self.tracklet_manager.all_tracklets())
        for overlap in overlaps:
            self.tracklet_manager.merge_tracklets(overlap.keep_id, overlap.remove_id)

    def _notify_and_reset_changes(self) -> None:
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.needs_notification:
                self._notify_callback(tracklet)

        self.tracklet_manager.mark_all_as_notified()

    def _remove_expired_tracklets(self) -> None:
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.status == TrackingStatus.REMOVED:
                self.tracklet_manager.remove_tracklet(tracklet.id)

    def find_overlapping_tracklets(self, tracklets: list[Tracklet]) -> set[PanoramicOverlapInfo]:

        for i, tracklet in enumerate(tracklets):
            # Reconstruct PanoramicTrackerInfo with updated overlap field
            if isinstance(tracklet.metadata, PanoramicMetadata):
                updated_overlap = self.geometry.angle_in_overlap(tracklet.metadata.local_angle, self.cam_360_overlap_expansion)

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
            if angle_diff > self.geometry.fov_overlap * (1.0 + self.cam_360_overlap_expansion):
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
            if edge_newest >= edge_oldest / self.cam_360_hysteresis_factor:
                overlaps.append(PanoramicOverlapInfo(newest.id, oldest.id, angle_diff, f'{edge_newest} >= {edge_oldest}'))
            else:
                overlaps.append(PanoramicOverlapInfo(oldest.id, newest.id, angle_diff,  f'{edge_newest} < {edge_oldest}'))

        overlap_sets: set[PanoramicOverlapInfo] = set(overlaps)
        return overlap_sets

    # CALLBACKS
    def _notify_callback(self, tracklet: Tracklet) -> None:
        with self.callback_lock:
            for c in self.tracklet_callbacks:
                c(tracklet)

            # if tracklet.id == 1:
            #     addtracklets = []
            #     for i in range(5, 8):
            #         # Create a new tracklet with a different ID
            #         another_tracklet: Tracklet = replace(tracklet, id=i)
            #         addtracklets.append(another_tracklet)
            #     for at in addtracklets:
            #         for c in self.tracklet_callbacks:
            #             c(at)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        with self.callback_lock:
            self.tracklet_callbacks.add(callback)

    def add_cam_tracklets(self, cam_id: int, cam_tracklets: list[CamTracklet]) -> None :
        for t in cam_tracklets:
            tracklet: Optional[Tracklet] = Tracklet.from_depthcam(cam_id, t)
            if tracklet is None:
                print(f"PanoramicTracker: Invalid tracklet from camera {cam_id}, skipping.")
                return
            self.input_queue.put(tracklet)
