from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from itertools import combinations
from queue import Empty, Queue
from threading import Thread, Lock
from time import time, sleep
from typing import Optional, Protocol

# Third-party imports
import numpy as np
from pandas import Timestamp

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet as CamTracklet
from modules.tracker.Tracklet import Person, PersonCallback, TrackingStatus
from modules.tracker.TrackletManager import PersonManager
from modules.tracker.BaseTracker import BaseTrackerInfo
from modules.tracker.panoramic.PanoramicTrackerGui import PanoramicTrackerGui
from modules.tracker.panoramic.PanoramicGeometry import PanoramicGeometry
from modules.tracker.panoramic.PanoramicDefinitions import *
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass(frozen=True)
class TempTracklet:
    cam_id: int
    tracklet: CamTracklet
    time_stamp: Timestamp = field(default_factory=Timestamp.now)

TempTrackletDict = dict[int, list[TempTracklet]]



@dataclass (frozen=True)
class PanoramicTrackerInfo(BaseTrackerInfo):
    local_angle: float
    world_angle: float
    overlap: bool


class PanoramicTracker(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.running: bool = False
        self.max_persons: int = settings.max_players
        self.cleanup_interval: float = 1.0 / settings.camera_fps

        self.tracklet_queue: Queue[TempTracklet] = Queue()

        self.person_manager: PersonManager = PersonManager(self.max_persons)

        self.geometry: PanoramicGeometry = PanoramicGeometry(settings.camera_num, CAM_360_FOV, CAM_360_TARGET_FOV)

        self.tracklet_min_age: int =            settings.tracker_min_age
        self.tracklet_min_height: float =       settings.tracker_min_height
        self.person_timeout: float =            settings.tracker_timeout
        self.person_roi_expansion: float =      settings.pose_crop_expansion
        self.cam_360_edge_threshold: float =    CAM_360_EDGE_THRESHOLD
        self.cam_360_overlap_expansion: float = CAM_360_OVERLAP_EXPANSION
        self.cam_360_hysteresis_factor: float = CAM_360_HYSTERESIS_FACTOR

        self.callback_lock = Lock()
        self.person_callbacks: set[PersonCallback] = set()
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
            self.person_callbacks.clear()

    def run(self) -> None:
        last_cleanup: float = time()
        while self.running:
            # Try to get all available tracklets as soon as they arrive
            try:
                tracklet: TempTracklet = self.tracklet_queue.get(timeout=0.1)
                # try:
                self.update_persons(tracklet)
                # except Exception as e:
                #     print(f"Error updating persons with tracklet from camera {tracklet.cam_id}: {e}")
            except Empty:
                pass

    def update_persons(self, new_tracklet: TempTracklet) -> None:

        tracklet: CamTracklet = new_tracklet.tracklet
        # Filter out invalid persons
        if tracklet.status == TrackingStatus.REMOVED:
        # if new_person.status == TrackingStatus.REMOVED:
            return
        if tracklet.age <= self.tracklet_min_age:
            return
        # print(f"Updating person from camera {new_person.cam_id} with tracklet {new_person.tracklet.id} and status {new_person.status} ")
        if tracklet.roi.height < self.tracklet_min_height:
            return

        # Calculate the local and world angles for the new person
        local_angle, world_angle = self.geometry.calc_angle(tracklet.roi, new_tracklet.cam_id)

        # Filter out persons that are too close to the edge of a camera's field of view
        if self.geometry.angle_in_edge(local_angle, self.cam_360_edge_threshold):
            return

        in_overlap: bool = self.geometry.angle_in_overlap(local_angle, self.cam_360_overlap_expansion)
        tracker_info = PanoramicTrackerInfo(local_angle, world_angle, in_overlap)

        new_person: Optional[Person] = Person(-1, new_tracklet.cam_id, new_tracklet.tracklet, new_tracklet.time_stamp, tracker_info)

        # Check if the new person already exists in the tracker and update if necessary
        existing_person: Optional[Person] = self.person_manager.get_person_by_cam_and_tracklet(new_person.cam_id, new_person.tracklet.id)
        if existing_person is not None:
            self.person_manager.replace_person(existing_person, new_person)

        # Add the new person to the tracker (if it is not lost)
        if existing_person is None and new_person.status != TrackingStatus.LOST:
            self.person_manager.add_person(new_person)

        # Remove persons that are not active anymore
        for person in self.person_manager.all_persons():
            if person.last_time < time() - self.person_timeout:
                self.remove_person(person)

        self.remove_overlapping_persons()

        for person in self.person_manager.all_persons():
            if person.is_active:
                self._person_callback(person)

    def remove_overlapping_persons(self) -> None:

        persons: list[Person] = self.person_manager.all_persons()
        for person in persons:
            # Reconstruct PanoramicTrackerInfo with updated overlap field
            if isinstance(person.tracker_info, PanoramicTrackerInfo):
                updated_overlap = self.geometry.angle_in_overlap(person.tracker_info.local_angle, self.cam_360_overlap_expansion)
                person.tracker_info = PanoramicTrackerInfo(
                    local_angle=person.tracker_info.local_angle,
                    world_angle=person.tracker_info.world_angle,
                    overlap=updated_overlap
                )

        overlaps: list[tuple[int, int]] = []

        for P_A, P_B in combinations(persons, 2):
            if not getattr(P_A.tracker_info, "overlap", False) or not getattr(P_B.tracker_info, "overlap", False):
                continue
            if P_A.cam_id == P_B.cam_id:
                continue

            angle_diff: float = self.geometry.angle_diff(getattr(P_A.tracker_info, "world_angle", 45.0), getattr(P_B.tracker_info, "world_angle", 45.0))
            if angle_diff > self.geometry.fov_overlap * (1.0 + self.cam_360_overlap_expansion):
                continue

            # look at the hight of the trackets for extra filtering
            height_diff: float = abs(P_A.tracklet.roi.height - P_B.tracklet.roi.height)
            if height_diff > 0.1:
                continue

            if not P_A.is_active and P_B.is_active:
                overlaps.append((P_B.id, P_A.id))
                continue

            elif not P_B.is_active and P_A.is_active:
                overlaps.append((P_A.id, P_B.id))
                continue

            if not P_A.is_active and not P_B.is_active:
                continue

            # print(f"Comparing persons {P_A.id} and {P_B.id}: angle_diff={angle_diff:.2f}, height_diff={height_diff:.2f}")

            if P_A.age < P_B.age:
                newest, oldest = P_A, P_B
            else:
                newest, oldest = P_B, P_A
            edge_newest: float = self.geometry.angle_from_edge(getattr(newest.tracker_info, "local_angle", 45.0))
            edge_oldest: float = self.geometry.angle_from_edge(getattr(oldest.tracker_info, "local_angle", 45.0))
            # print(f"Comparing newest {newest.id} (edge {edge_newest}) to oldest {oldest.id} (edge {edge_oldest})")
            if edge_newest >= edge_oldest / self.cam_360_hysteresis_factor:
                overlaps.append((newest.id, oldest.id))
            else:
                overlaps.append((oldest.id, newest.id))

        overlap_sets: set[tuple[int, int]] = set(overlaps)

        for overlap in overlap_sets:
            # print(f"Removing overlapping persons: {overlap}")
            keep_person: Optional[Person] = self.person_manager.get_person(overlap[0])
            remove_person: Optional[Person] = self.person_manager.get_person(overlap[1])
            if keep_person is None or remove_person is None:
                print(f"Warning: One of the persons in the overlap {overlap} is None sets{overlap_sets}. Skipping removal.")
                continue

            remove_id: int = self.person_manager.merge_persons(keep_person, remove_person)

            # If the merge was not successful, we create a dummy person to trigger the callback
            if remove_person.status != TrackingStatus.NEW:
                dummy_person: Person = Person(
                    remove_id,
                    remove_person.cam_id,
                    remove_person.tracklet,
                    remove_person.time_stamp,
                    remove_person.tracker_info  # Pass the tracker_info from the removed person
                )
                dummy_person.status = TrackingStatus.REMOVED
                self._person_callback(dummy_person)

    def remove_person(self, person: Person) -> None:
        self.person_manager.remove_person(person.id)
        person.status = TrackingStatus.REMOVED
        self._person_callback(person)

    def add_cam_tracklet(self, cam_id: int, tracklet: CamTracklet) -> None :
        cam_tracklet = TempTracklet(cam_id, tracklet)
        self.tracklet_queue.put(cam_tracklet)

    # CALLBACKS
    def _person_callback(self, person: Person) -> None:
        with self.callback_lock:
            for c in self.person_callbacks:
                c(person)
    def add_person_callback(self, callback: PersonCallback) -> None:
        if self.running:
            print('Manager is running, cannot add callback')
            return
        self.person_callbacks.add(callback)
