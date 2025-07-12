from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from itertools import combinations
from queue import Empty, Queue
from threading import Thread, Lock
from typing import Optional

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet as CamTracklet
from modules.tracker.Tracklet import Tracklet, TrackletCallback, TrackingStatus
from modules.tracker.TrackletManager import TrackletManager
from modules.tracker.BaseTracker import BaseTrackerInfo
from modules.tracker.panoramic.PanoramicTrackerGui import PanoramicTrackerGui
from modules.tracker.panoramic.PanoramicGeometry import PanoramicGeometry
from modules.tracker.panoramic.PanoramicDefinitions import *
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass (frozen=True)
class PanoramicTrackerInfo(BaseTrackerInfo):
    local_angle: float
    world_angle: float
    overlap: bool

@dataclass(frozen=True)
class PanoramicOverlapInfo:
    keep_id: int
    remove_id: int
    distance: float

class PanoramicTracker(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.running: bool = False
        self.max_players: int = settings.max_players
        self.cleanup_interval: float = 1.0 / settings.camera_fps

        self.input_queue: Queue[Tracklet] = Queue()

        self.tracklet_manager: TrackletManager = TrackletManager(self.max_players)

        self.geometry: PanoramicGeometry = PanoramicGeometry(settings.camera_num, CAM_360_FOV, CAM_360_TARGET_FOV)

        self.tracklet_min_age: int =            settings.tracker_min_age
        self.tracklet_min_height: float =       settings.tracker_min_height
        self.timeout: float =                   settings.tracker_timeout
        self.roi_expansion: float =             settings.pose_crop_expansion
        self.cam_360_edge_threshold: float =    CAM_360_EDGE_THRESHOLD
        self.cam_360_overlap_expansion: float = CAM_360_OVERLAP_EXPANSION
        self.cam_360_hysteresis_factor: float = CAM_360_HYSTERESIS_FACTOR

        self.callback_lock = Lock()
        self.tracklet_callbacks: set[TrackletCallback] = set()
        self.gui = PanoramicTrackerGui(gui, self, settings)

        hot_reload = HotReloadMethods(self.__class__)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        super().start()

    def stop(self) -> None:
        self.running = False

        with self.callback_lock:
            self.tracklet_callbacks.clear()

    def run(self) -> None:
        while self.running:
            # Try to get all available tracklets as soon as they arrive
            try:
                tracklet: Tracklet = self.input_queue.get(timeout=0.05)
                # try:
                self.update_tracklet(tracklet)
                # except Exception as e:
                #     print(f"Error updating tracklet with tracklet from camera {tracklet.cam_id}: {e}")
            except Empty:
                pass

    def cleanup_tracklets(self) -> None:
        """Remove tracklets that are not active anymore"""
        removed: list[Tracklet] = self.tracklet_manager.cleanup(self.timeout)
        for tracklet in removed:
            self._notify_callback(tracklet)

    def update_tracklet(self, new_tracklet: Tracklet) -> None:

        if new_tracklet.status in (TrackingStatus.REMOVED, TrackingStatus.LOST):
            return
        if new_tracklet.external_age_in_frames <= self.tracklet_min_age:
            return
        if new_tracklet.roi.height < self.tracklet_min_height:
            return

        local_angle, world_angle, _overlap = self.geometry.get_angles_and_overlap(new_tracklet.roi, new_tracklet.cam_id, self.cam_360_edge_threshold)
        new_tracklet.tracker_info = PanoramicTrackerInfo(local_angle, world_angle, _overlap)

        # Filter out tracklets that are too close to the edge of a camera's field of view
        if self.geometry.angle_in_edge(local_angle, self.cam_360_edge_threshold):
            return

        # Check if the new tracklet already exists in the tracker and update if necessary
        existing_tracklet: Optional[Tracklet] = self.tracklet_manager.get_tracklet_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
        if existing_tracklet is not None:
            self.tracklet_manager.replace_tracklet(existing_tracklet, new_tracklet)
        else:
            self.tracklet_manager.add_tracklet(new_tracklet)

        # Remove tracklets that are not active anymore
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.is_expired(self.timeout):
                self.tracklet_manager.remove_tracklet(tracklet.id)
                tracklet.status = TrackingStatus.REMOVED
                self._notify_callback(tracklet)

        overlaps: set[PanoramicOverlapInfo] = self.find_overlapping_tracklets()
        for overlap in overlaps:
            keep: Optional[Tracklet] = self.tracklet_manager.get_tracklet(overlap.keep_id)
            remove: Optional[Tracklet] = self.tracklet_manager.get_tracklet(overlap.remove_id)
            if keep is None or remove is None:
                print(f"Warning: One of the tracklets in the overlap {overlap} is None sets{overlaps}. Skipping removal.")
                continue

            remove_id: int = self.tracklet_manager.merge_tracklets(keep, remove)

            # If the merge was not successful, we create a dummy tracklet to trigger the callback
            if remove.status != TrackingStatus.NEW:
                dummy: Tracklet = Tracklet(
                    cam_id=keep.cam_id,
                    id=remove_id,
                    status=TrackingStatus.REMOVED,
                )
                self._notify_callback(dummy)

        for new_tracklet in self.tracklet_manager.all_tracklets():
            if new_tracklet.is_active:
                self._notify_callback(new_tracklet)

    def find_overlapping_tracklets(self) -> set[PanoramicOverlapInfo]:

        tracklets: list[Tracklet] = self.tracklet_manager.all_tracklets()
        for tracklet in tracklets:
            # Reconstruct PanoramicTrackerInfo with updated overlap field
            if isinstance(tracklet.tracker_info, PanoramicTrackerInfo):
                updated_overlap = self.geometry.angle_in_overlap(tracklet.tracker_info.local_angle, self.cam_360_overlap_expansion)
                tracklet.tracker_info = PanoramicTrackerInfo(
                    local_angle=tracklet.tracker_info.local_angle,
                    world_angle=tracklet.tracker_info.world_angle,
                    overlap=updated_overlap
                )

        overlaps: list[PanoramicOverlapInfo] = []

        for P_A, P_B in combinations(tracklets, 2):
            if not getattr(P_A.tracker_info, "overlap", False) or not getattr(P_B.tracker_info, "overlap", False):
                continue
            if P_A.cam_id == P_B.cam_id:
                continue

            angle_diff: float = self.geometry.angle_diff(getattr(P_A.tracker_info, "world_angle", 45.0), getattr(P_B.tracker_info, "world_angle", 45.0))
            if angle_diff > self.geometry.fov_overlap * (1.0 + self.cam_360_overlap_expansion):
                continue

            # look at the hight of the trackets for extra filtering
            height_diff: float = abs(P_A.roi.height - P_B.roi.height)
            if height_diff > 0.1:
                continue

            if not P_A.is_active and P_B.is_active:
                overlaps.append(PanoramicOverlapInfo(P_B.id, P_A.id, angle_diff))
                continue

            elif not P_B.is_active and P_A.is_active:
                overlaps.append(PanoramicOverlapInfo(P_A.id, P_B.id, angle_diff))
                continue

            if not P_A.is_active and not P_B.is_active:
                continue

            if P_A.age_in_seconds < P_B.age_in_seconds:
                newest, oldest = P_A, P_B
            else:
                newest, oldest = P_B, P_A
            edge_newest: float = self.geometry.angle_from_edge(getattr(newest.tracker_info, "local_angle", 45.0))
            edge_oldest: float = self.geometry.angle_from_edge(getattr(oldest.tracker_info, "local_angle", 45.0))
            # print(f"Comparing newest {newest.id} (edge {edge_newest}) to oldest {oldest.id} (edge {edge_oldest})")
            if edge_newest >= edge_oldest / self.cam_360_hysteresis_factor:
                overlaps.append(PanoramicOverlapInfo(newest.id, oldest.id, angle_diff))
            else:
                overlaps.append(PanoramicOverlapInfo(oldest.id, newest.id, angle_diff))

        overlap_sets: set[PanoramicOverlapInfo] = set(overlaps)
        return overlap_sets

    # CALLBACKS
    def _notify_callback(self, tracklet: Tracklet) -> None:
        with self.callback_lock:
            for c in self.tracklet_callbacks:
                c(tracklet)
    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        with self.callback_lock:
            self.tracklet_callbacks.add(callback)

    def add_cam_tracklet(self, cam_id: int, cam_tracklet: CamTracklet) -> None :
        tracklet: Optional[Tracklet] = Tracklet.from_depthcam(cam_id, cam_tracklet)
        if tracklet is None:
            print(f"PanoramicTracker: Invalid tracklet from camera {cam_id}, skipping.")
            return
        self.input_queue.put(tracklet)
