from __future__ import annotations

# Standard library imports
from itertools import combinations
from queue import Empty, Queue
from threading import Thread, Lock
from time import time, sleep
from typing import Optional

# Third-party imports
import numpy as np
from pandas import Timestamp

# Local application imports
from modules.Settings import Settings
from modules.person.panoramic.PanoramicTrackerGui import PanoramicTrackerGui
from modules.person.Person import Person, PersonDict, PersonCallback, PersonDictCallback, TrackingStatus
from modules.person.PersonManager import PersonManager
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType
from modules.person.panoramic.PanoramicGeometry import PanoramicGeometry
from modules.person.panoramic.PanoramicDefinitions import *

from modules.utils.HotReloadMethods import HotReloadMethods

class CamTracklet:
    def __init__(self, cam_id: int, tracklet: Tracklet) -> None:
        self.cam_id: int = cam_id
        self.tracklet: Tracklet = tracklet
        self.time_stamp: Timestamp = Timestamp.now()

CamTrackletDict = dict[int, list[Tracklet]]


class PanoramicTracker(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.input_mutex: Lock = Lock()
        self.running: bool = False
        self.max_persons: int = settings.pose_num
        self.cleanup_interval: float = 1.0 / settings.camera_fps

        self.input_frames: dict[int, np.ndarray] = {}
        self.tracklet_queue: Queue[CamTracklet] = Queue()

        self.person_manager: PersonManager = PersonManager(self.max_persons)

        self.geometry: PanoramicGeometry = PanoramicGeometry(settings.camera_num, CAM_360_FOV, CAM_360_TARGET_FOV)

        self.min_tracklet_age: int =            MIN_TRACKLET_AGE
        self.min_tracklet_height: float =       MIN_TRACKLET_HEIGHT
        self.cam_360_edge_threshold: float =    CAM_360_EDGE_THRESHOLD
        self.cam_360_overlap_expansion: float = CAM_360_OVERLAP_EXPANSION
        self.cam_360_hysteresis_factor: float = CAM_360_HYSTERESIS_FACTOR
        self.person_roi_expansion: float =      PERSON_ROI_EXPANSION
        self.person_timeout: float =            PERSON_TIMEOUT

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
                tracklet: CamTracklet = self.tracklet_queue.get(timeout=0.1)
                # try:
                self.update_persons(tracklet)
                # except Exception as e:
                #     print(f"Error updating persons with tracklet from camera {tracklet.cam_id}: {e}")
            except Empty:
                pass  # No more tracklets right now

            # self.cleanup_persons()

            # # Periodic tasks (e.g., cleanup)
            # now: float = time()
            # if now - last_cleanup >= self.cleanup_interval:
            #     self.cleanup_persons()
            #     last_cleanup = now

            # sleep(0.01)  # Prevent busy-waiting

    def update_persons(self, cam_tracklet: CamTracklet) -> None:

        new_person: Optional[Person] = Person(-1, cam_tracklet.cam_id, cam_tracklet.tracklet, cam_tracklet.time_stamp)

        # Filter out invalid persons
        if new_person.status == TrackingStatus.REMOVED:
            return
        if new_person.tracklet.age <= self.min_tracklet_age:
            return
        # print(f"Updating person from camera {new_person.cam_id} with tracklet {new_person.tracklet.id} and status {new_person.status} ")
        if new_person.tracklet.roi.height < self.min_tracklet_height:
            return

        # Calculate the local and world angles for the new person
        new_person.local_angle, new_person.world_angle = self.geometry.calc_angle(new_person.tracklet.roi, new_person.cam_id)

        # Filter out persons that are too close to the edge of a camera's field of view
        if self.geometry.angle_in_edge(new_person.local_angle, self.cam_360_edge_threshold):
            return

        # Check if the new person already exists in the tracker and update if necessary
        existing_person: Optional[Person] = self.person_manager.get_person_by_cam_and_tracklet(new_person.cam_id, new_person.tracklet.id)
        if existing_person is not None:
            self.person_manager.replace_person(existing_person, new_person)

        # Add the new person to the tracker (if it is not lost)
        if existing_person is None and new_person.status != TrackingStatus.LOST:
            new_person.status = TrackingStatus.NEW
            self.person_manager.add_person(new_person)

        # Remove persons that are not active anymore
        for person in self.person_manager.all_persons():
            if person.last_time < time() - self.person_timeout:
                self.remove_person(person)

        self.remove_overlapping_persons()

        for person in self.person_manager.all_persons():
            if person.is_active and person.img is None:
                image: Optional[np.ndarray] = self._get_image(person.cam_id)
                if image is None:
                    print(f"Warning: No image available for person {person.id} in camera {person.cam_id}.")
                    return

                person.set_pose_roi(image, self.person_roi_expansion)
                person.set_pose_image(image)

                self._person_callback(person)

    def remove_overlapping_persons(self) -> None:

        persons: list[Person] = self.person_manager.all_persons()
        for person in persons:
            person.overlap = self.geometry.angle_in_overlap(person.local_angle, self.cam_360_overlap_expansion)

        overlaps: list[tuple[int, int]] = []

        for P_A, P_B in combinations(persons, 2):
            if not P_A.overlap or not P_B.overlap:
                continue
            if P_A.cam_id == P_B.cam_id:
                continue

            angle_diff: float = self.geometry.angle_diff(P_A.world_angle, P_B.world_angle)
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
            edge_newest: float = self.geometry.angle_from_edge(newest.local_angle)
            edge_oldest: float = self.geometry.angle_from_edge(oldest.local_angle)
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
                dummy_person: Person = Person(remove_id, remove_person.cam_id, remove_person.tracklet, remove_person.time_stamp)
                dummy_person.status = TrackingStatus.REMOVED
                self._person_callback(dummy_person)

    def remove_person(self, person: Person) -> None:
        self.person_manager.remove_person(person.id)
        person.status = TrackingStatus.REMOVED
        self._person_callback(person)

    # INPUTS
    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None :
        if frame_type != FrameType.VIDEO:
            return
        with self.input_mutex:
            self.input_frames[id] = image
    def _get_image(self, id: int) -> Optional[np.ndarray]:
        with self.input_mutex:
            if not self.input_frames.get(id) is None:
                return self.input_frames[id]
            return None
    def add_tracklet(self, cam_id: int, tracklet: Tracklet) -> None :
        cam_tracklet = CamTracklet(cam_id, tracklet)
        self.tracklet_queue.put(cam_tracklet)

    # CALLBACKS
    def _person_callback(self, person: Person) -> None:
        if person.status == TrackingStatus.REMOVED:
            print(f"Person {person.id} in camera {person.cam_id} has been removed.")
            print(self.person_manager._id_pool.available)

        with self.callback_lock:
            for c in self.person_callbacks:
                c(person)
    def add_person_callback(self, callback: PersonCallback) -> None:
        if self.running:
            print('Manager is running, cannot add callback')
            return
        self.person_callbacks.add(callback)
