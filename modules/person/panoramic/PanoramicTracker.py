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
from modules.person.Person import Person, PersonDict, PersonCallback, PersonDictCallback, PersonIdPool, TrackingStatus
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType
from modules.person.panoramic.PanooramicGeometry import PanoramicGeometry
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

        self.person_id_pool: PersonIdPool = PersonIdPool(self.max_persons)
        self.persons: PersonDict = {}

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

        existing_person: Optional[Person]
        # Check if the new person already exists in the tracker and update if necessary
        existing_person = PanoramicTracker.find_same_person(new_person, self.persons)
        if existing_person is not None:
            new_person.set_from_existing(existing_person)
            self.persons[new_person.id] = new_person
            # print(f"Person {new_person.id} already exists in camera {new_person.cam_id}, updating with new data.")

        # Add the new person to the tracker (if it is not lost)
        if existing_person is None and new_person.status != TrackingStatus.LOST:
            self._add_person(new_person, self.persons, self.person_id_pool)

        self.cleanup_persons()

        self.remove_overlapping_persons()


        for id, person in self.persons.items():
            if person.is_active and person.img is None:
                image: Optional[np.ndarray] = self._get_image(person.cam_id)
                if image is None:
                    print(f"Warning: No image available for person {person.id} in camera {person.cam_id}.")
                    return

                person.set_pose_roi(image, self.person_roi_expansion)
                person.set_pose_image(image)


                self._person_callback(person)

#

    def cleanup_persons(self) -> None:
        rejected_persons: list[Person] = []

        for person in self.persons.values():
            if person.last_time < time() - self.person_timeout:
                rejected_persons.append(person)
            elif person.status == TrackingStatus.REMOVED:
                rejected_persons.append(person)

        for person in rejected_persons:
            PanoramicTracker._remove_person(person, self.persons, self.person_id_pool)
            self._person_callback(person)

        keys: list[int] = list(self.persons.keys())
        ids: list[int] = [p.id for p in self.persons.values()]
        if keys != ids:
            print(f"Warning: Person dict keys {keys} do not match person object ids {ids}")

    @staticmethod
    def find_same_person(person: Person, persons: PersonDict) -> Optional[Person]:
        for p in persons.values():
            if person.cam_id == p.cam_id and person.tracklet.id == p.tracklet.id:
                return p
        return None

    def remove_overlapping_persons(self) -> None:
        for person in self.persons.values():
            person.overlap = self.geometry.angle_in_overlap(person.local_angle, self.cam_360_overlap_expansion)

        overlaps: list[tuple[int, int]] = []

        for P_A in self.persons.values():
            if not P_A.overlap:
                continue
            for P_B in self.persons.values():
                if not P_B.overlap:
                    continue
                if P_A.cam_id == P_B.cam_id:
                    continue

                angle_diff: float = self.geometry.angle_diff(P_A.world_angle, P_B.world_angle)
                if angle_diff < self.geometry.fov_overlap * (1.0 + self.cam_360_overlap_expansion):
                    if P_A.start_time > P_B.start_time:
                        newest, oldest = P_A, P_B
                    else:
                        newest, oldest = P_B, P_A
                    edge_newest: float = self.geometry.angle_from_edge(newest.local_angle)
                    edge_oldest: float = self.geometry.angle_from_edge(oldest.local_angle)
                    # print(f"Comparing newest {newest.id} (edge {edge_newest}) to oldest {oldest.id} (edge {edge_oldest})")
                    if edge_newest > edge_oldest / self.cam_360_hysteresis_factor:
                        overlaps.append((newest.id, oldest.id))
                    else:
                        overlaps.append((oldest.id, newest.id))

        overlap_sets: set[tuple[int, int]] = set(overlaps)

        # print(self.geometry.angle_from_edge(104))

        for overlap in overlap_sets:
            print(f"Removing overlapping persons: {overlap} from {self.persons.keys()}")
            keep_person: Person = self.persons[overlap[0]]
            remove_person: Person = self.persons[overlap[1]]
            edge_keep: float = self.geometry.angle_from_edge(keep_person.local_angle)
            edge_remove: float = self.geometry.angle_from_edge(remove_person.local_angle)
            # print("keep", keep_person.id, edge_keep, keep_person.status,  "remove", remove_person.id, edge_remove , remove_person.status)

            if keep_person.start_time < remove_person.start_time:
                # self._remove_person(remove_person, self.persons, self.person_id_pool)

                self.person_id_pool.release(remove_person.id)
                self.persons.pop(remove_person.id, None)
                # self.persons[keep_person.id] = keep_person

            else:

                self.person_id_pool.release(keep_person.id)
                self.persons.pop(keep_person.id, None)
                keep_person.set_from_existing(remove_person)
                self.persons[remove_person.id] = keep_person


            # print all ids in self.persons
            # print(id for id in self.persons.keys())
            # print(self.persons.keys(), overlap_sets)





        # if overlaps:
        #     print(f"Found {overlap_sets}")

        # overlapping persons list, the first element is the one to keep, the other to delete
        overlapping_persons: list[tuple[int, int]] = []

        # overlapping_persons: list[Person] = []

        # for P_A, P_B in combinations(persons.values(), 2):
        #     if not P_A.overlap or not P_B.overlap:
        #         continue
        #     angle_diff: float = geometery.angle_diff(P_A.world_angle, P_B.world_angle)
        #     if angle_diff < geometery.fov_overlap * (1.0 + overlap_expansion):
        #         if not P_A.is_active and P_B.is_active:
        #             overlapping_persons.append((P_B.id, P_A.id))
        #             print(f"Found overlapping person: {P_B.id} with angle diff {angle_diff}, status {P_B.status}, P_A status {P_A.status}")
        #             continue
        #         if not P_B.is_active and P_A.is_active:
        #             print(f"Found overlapping person: {P_A.id} with angle diff {angle_diff}, status {P_B.status}, P_A status {P_A.status}")
        #             overlapping_persons.append((P_A.id, P_B.id))
        #             continue
        #         if P_A.status == TrackingStatus.TRACKED and P_B.status == TrackingStatus.TRACKED:
        #             # print(f"Found overlapping persons: {P_A.id} and {P_B.id} with angle diff {angle_diff}")
        #             if P_A.start_time > P_B.start_time:
        #                 newest, other = P_A, P_B
        #             else:
        #                 newest, other = P_B, P_A

        #             # print start times
        #             print(f"Comparing newest {newest.id} (start time {time() - newest.start_time}) to other {other.id} (start time {time() - other.start_time})")

        #             edge_newest: float = geometery.angle_from_edge(newest.local_angle)
        #             edge_other: float = geometery.angle_from_edge(other.local_angle)
        #             # print(f"Comparing newest {newest.id} (edge {edge_newest}) to other {other.id} (edge {edge_other}), hysteresis {hysteresis_factor}")
        #             if edge_newest >= edge_other * hysteresis_factor:
        #                 print(f"Deleting {other.id} (other) because NEWEST {newest.id} is further from edge (passes hysteresis)")
        #                 overlapping_persons
        #             else:
        #                 print(f"Deleting {newest.id} (newest) because {other.id} is further from edge (fails hysteresis)")
        #                 overlapping_persons[newest.id] = newest  # Mark newest for deletion


        # # if overlapping_persons:
        # #     print(f"Found overlapping persons: {[p.id for p in overlapping_persons.values()]}", flush=True)

        # return overlapping_persons

    @staticmethod
    def _add_person(person: Person, persons_dict: PersonDict, person_id_pool: PersonIdPool) -> None:
        if person.status == TrackingStatus.LOST or person.status == TrackingStatus.REMOVED:
            print(f"Person {person.id} is not in a valid state to be added: {person.status}")
            return

        try:
            person_id: int = person_id_pool.acquire()
        except:
            print('No more person ids available')
            return

        person.id = person_id
        person.status = TrackingStatus.NEW
        persons_dict[person_id] = person

    @staticmethod
    def _remove_person(person: Person, persons_dict: PersonDict, person_id_pool: PersonIdPool) -> None:
        person_id: int = person.id
        person.status = TrackingStatus.REMOVED
        try:
            person_id_pool.release(person_id)
        except Exception as e:
            print(f"Error releasing person ID {person_id}: {e}")
            return

        persons_dict.pop(person.id, None)

        # print(f"Person {person_id} removed from tracker. Remaining persons: {list(persons_dict.keys())}")

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
        with self.callback_lock:
            for c in self.person_callbacks:
                c(person)
    def add_person_callback(self, callback: PersonCallback) -> None:
        if self.running:
            print('Manager is running, cannot add callback')
            return
        self.person_callbacks.add(callback)
