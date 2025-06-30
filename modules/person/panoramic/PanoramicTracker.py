from __future__ import annotations

# Standard library imports
from threading import Thread, Lock
from time import time, sleep
from typing import Optional

# Third-party imports
import numpy as np

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
        self.timestamp: float = time()

CamTrackletDict = dict[int, list[Tracklet]]


class PanoramicTracker(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.input_mutex: Lock = Lock()
        self.running: bool = False
        self.max_persons: int = settings.pose_num

        self.input_frames: dict[int, np.ndarray] = {}
        self.input_tracklets: CamTrackletDict = {}

        self.person_id_pool: PersonIdPool = PersonIdPool(self.max_persons)
        self.persons: PersonDict = {}

        self.cam_360: PanoramicGeometry = PanoramicGeometry(settings.camera_num, CAM_360_FOV, CAM_360_TARGET_FOV)

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
        while self.running:
            try:
                self.update_persons()
            except Exception as e:
                print(f"Error in PanoramicTracker: {e}")
            sleep(0.01)
            pass

    def update_persons(self) -> None:
        # print(self.persons)

        self._cleanup_inactive_persons(self.persons, self.person_id_pool, self.person_timeout)

        tracklets: CamTrackletDict = self._get_tracklets()
        tracklet_persons: list[Person] = self._get_persons_from_tracklets(tracklets) # active tracklets pass filter included
        # if len(tracklet_persons):
            # print(f"Found {len(tracklet_persons)} tracklets in {len(tracklets)} cameras.")
        self._filter_tracking_status(tracklet_persons)
        self._filter_age(tracklet_persons, self.min_tracklet_age)
        self._filter_size(tracklet_persons, self.min_tracklet_height)
        self.cam_360.calc_angles(tracklet_persons)
        self._filter_edge(tracklet_persons, self.cam_360, self.cam_360_edge_threshold)
        self._filter_and_update_active(self.persons, tracklet_persons)
        self._filter_and_update_overlap(self.persons, tracklet_persons, self.cam_360, self.cam_360_overlap_expansion, self.cam_360_hysteresis_factor)
        self._add_persons(self.persons, tracklet_persons, self.person_id_pool)


        # find new persons that have no image, these are the only ones that need to be processed and send to the output
        for person in self.persons.values():
            if person.img is not None:
                continue

            # person.status = TrackingStatus.NEW
            image: Optional[np.ndarray] = self._get_image(person.cam_id)
            if image is None:
                print(f"Warning: No image available for person {person.id} in camera {person.cam_id}.")
                continue

            person.set_pose_roi(image, self.person_roi_expansion)
            person.set_pose_image(image)
            self._person_callback(person)

    @staticmethod
    def _get_persons_from_tracklets(cam_tracklets: CamTrackletDict) -> list[Person]:
        persons: list[Person] = []
        for cam_id in cam_tracklets.keys():
            tracklets: list[Tracklet] = cam_tracklets[cam_id]
            for tracklet in tracklets:
                person: Person = Person(-1, cam_id, tracklet)

                # person.status = TrackingStatus[tracklet.status.name]
                persons.append(person)
        return persons

    @staticmethod
    def _filter_tracking_status(persons: list[Person]) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if person.status == TrackingStatus.REMOVED:
                rejected_persons.append(person)
        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def _filter_age(persons: list[Person], age: int) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if person.tracklet.age <= age:
                rejected_persons.append(person)
        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def _filter_size(persons: list[Person], size_range: float) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if person.tracklet.roi.height < size_range:
                rejected_persons.append(person)
        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def _filter_edge(persons: list[Person], circular: PanoramicGeometry, edge_range: float) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if circular.angle_in_edge(person.local_angle, edge_range):
                rejected_persons.append(person)

        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def _filter_and_update_active(active_persons: PersonDict, new_persons: list[Person]) -> None:
        def find_same_person(person: Person, persons: list[Person]) -> Optional[Person]:
            for p in persons:
                if person.cam_id == p.cam_id and person.tracklet.id == p.tracklet.id:
                    return p
            return None


        for person in active_persons.values():
            same_person: Optional[Person] = find_same_person(person, new_persons)
            # print(same_person)
            if same_person is not None:
                new_persons.remove(same_person)
                # print(same_person.status, person.status)
                person.update_from(same_person)
                if same_person.status == TrackingStatus.TRACKED:
                    person.last_time = time()
                # person.status = TrackingStatus.TRACKED

    @staticmethod
    def _filter_and_update_overlap(active_persons: PersonDict, new_persons: list[Person], radial: PanoramicGeometry,
                                  overlap_expansion: float, hysteresis_factor: float) -> None:

        for person in new_persons:
            if radial.angle_in_overlap(person.world_angle, overlap_expansion):
                person.overlap = True

        # update overlap persons
        for person in active_persons.values():
            if not radial.angle_in_overlap(person.world_angle, overlap_expansion):
                person.overlap = False
            else:
                person.overlap = True

                overlap_persons: list[Person] = []
                for new_person in new_persons:
                    angle_diff: float = radial.angle_diff(person.world_angle, new_person.world_angle)
                    if angle_diff < radial.fov_overlap * (1.0 + overlap_expansion):
                        # also check roi?
                        overlap_persons.append(new_person)

                if overlap_persons:
                    for p in overlap_persons:
                        new_persons.remove(p)
                        # chose the person furthest away from the edge
                        if (radial.angle_from_edge(person.local_angle) < radial.angle_from_edge(p.local_angle) * hysteresis_factor):
                            person.update_from(overlap_persons[0])

    @staticmethod
    def _add_persons(active_persons: PersonDict, new_persons: list[Person], person_id_pool: PersonIdPool) -> None:
        for person in new_persons:

            if person.status != TrackingStatus.NEW and person.status != TrackingStatus.TRACKED:
                continue

            try:
                person_id: int = person_id_pool.acquire()
            except:
                print('No more person ids available')
                continue

            person.id = person_id
            person.status = TrackingStatus.NEW
            active_persons[person_id] = person

    @ staticmethod
    def _cleanup_inactive_persons(persons: PersonDict, person_id_pool: PersonIdPool, activity_duration: float = 1.0) -> None:

        # remove inactive persons
        rejected_persons: list[Person] = []
        for key, person in persons.items():
            if person.status == TrackingStatus.REMOVED:
                rejected_persons.append(person)
                continue
        for person in rejected_persons:
            persons.pop(person.id, None)
            print('Remove inactive person id:', person.id, 'cam', person.cam_id, 'tracklet', person.tracklet.id, 'overlap:', person.overlap, 'angle:', person.local_angle, 'world:', person.world_angle)

        for key in persons.keys():
            person: Person = persons[key]
            person = persons[key]
            if person.last_time < time() - activity_duration:
                person_id_pool.release(person.id)
                rejected_persons.append(person)
                person.status = TrackingStatus.REMOVED


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
        with self.input_mutex:
            tracklets: list[Tracklet] = self.input_tracklets.get(cam_id, [])
            tracklets.append(tracklet)
            self.input_tracklets[cam_id] = tracklets
    def _get_tracklets(self) -> CamTrackletDict:
        with self.input_mutex:
            tracklets: CamTrackletDict =  self.input_tracklets.copy()
            self.input_tracklets.clear()
            return tracklets

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
