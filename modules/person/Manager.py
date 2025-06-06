from __future__ import annotations

import cv2
import numpy as np
from threading import Thread, Lock
from time import time, sleep

from modules.Settings import Settings
from modules.person.Gui import Gui
from modules.person.Person import Person, PersonDict, PersonCallback
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType
from modules.person.pose.PoseDetection import PoseDetection, ModelType, ModelTypeNames
from modules.person.CircularCoordinates import CircularCoordinates
from modules.person.Definitions import *

from modules.person.Utils import IdPool

CamTrackletDict = dict[int, list[Tracklet]]

class Manager(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.input_mutex: Lock = Lock()
        self.running: bool = False
        self.max_persons: int = settings.num_players

        self.input_frames: dict[int, np.ndarray] = {}
        self.input_tracklets: CamTrackletDict = {}

        self.person_id_pool: IdPool = IdPool(self.max_persons)
        self.persons: PersonDict = {}

        self.circular_coordinates: CircularCoordinates = CircularCoordinates(settings.num_cams)

        self.pose_detectors: dict[int, PoseDetection] = {}
        self.pose_detector_frame_size: int = 256
        if not settings.pose:
            print('Pose Detection: Disabled')
        else:
            for i in range(self.max_persons):
                self.pose_detectors[i] = PoseDetection(settings.model_path, settings.model_type)
            print('Pose Detection:', self.max_persons, 'six instances of model', ModelTypeNames[settings.model_type.value])


        self.roi_expansion: float = 0.1
        self.activity_duration: float = 1.0  # seconds

        self.callbacks: set[PersonCallback] = set()
        self.gui = Gui(gui, self, settings)

    def start(self) -> None:
        if self.running:
            return

        for detector in self.pose_detectors.values():
            detector.addMessageCallback(self.callback)
            detector.start()
            self.pose_detector_frame_size = detector.get_frame_size()

        self.running = True

        super().start()

    def stop(self) -> None:
        self.running = False
        self.callbacks.clear()

    def run(self) -> None:
        while self.running:
            self.update_persons()
            sleep(0.01)
            pass

    def update_persons(self) -> None:
        tracklets: CamTrackletDict = self.get_tracklets()
        tracklet_persons: list[Person] = self.get_persons_from_tracklets(tracklets) # active tracklets pass filter included
        self.filter_active(tracklet_persons)
        self.filter_age(tracklet_persons, 4)
        self.filter_size(tracklet_persons, 0.25)
        self.circular_coordinates.calc_angles(tracklet_persons)
        self.filter_edge(tracklet_persons, self.circular_coordinates, 0.5)
        self.filter_and_update_active(self.persons, tracklet_persons)
        self.filter_and_update_overlap(self.persons, tracklet_persons, self.circular_coordinates)
        # self.filter_edge(tracklet_persons, self.circular_coordinates, 0.6)
        self.add_persons(self.persons, tracklet_persons, self.person_id_pool)
        self.cleanup_inactive_persons(self.persons, self.person_id_pool, self.activity_duration)

        for person in self.persons.values():
            image: np.ndarray = self.get_image(person.cam_id)
            person.set_pose_roi(image, self.roi_expansion)
            person.set_pose_image(image)
            detector: PoseDetection | None = self.pose_detectors.get(person.id, None)
            self.detect_pose(person, detector, self.callback)

    @staticmethod
    def get_persons_from_tracklets(cam_tracklets: CamTrackletDict) -> list[Person]:
        persons: list[Person] = []
        for cam_id in cam_tracklets.keys():
            tracklets: list[Tracklet] = cam_tracklets[cam_id]
            for tracklet in tracklets:
                person: Person = Person(-1, cam_id, tracklet)
                persons.append(person)
        return persons

    @staticmethod
    def filter_active(persons: list[Person]) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if person.tracklet.status != Tracklet.TrackingStatus.TRACKED:
                rejected_persons.append(person)
        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def filter_age(persons: list[Person], age: int) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if person.tracklet.age <= age:
                rejected_persons.append(person)
        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def filter_size(persons: list[Person], size_range: float) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if person.tracklet.roi.height < size_range:
                rejected_persons.append(person)
        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def filter_edge(persons: list[Person], circular: CircularCoordinates, edge_range: float) -> None:
        rejected_persons: list[Person] = []
        for person in persons:
            if circular.angle_in_edge(person.local_angle, edge_range):
                rejected_persons.append(person)

        for person in rejected_persons:
            persons.remove(person)

    @staticmethod
    def filter_and_update_active(active_persons: PersonDict, new_persons: list[Person]) -> None:
        def find_same_person(person: Person, persons: list[Person]) -> Person | None:
            for p in persons:
                if person.cam_id == p.cam_id and person.tracklet.id == p.tracklet.id:
                    return p
            return None

        for person in active_persons.values():
            same_person: Person | None = find_same_person(person, new_persons)
            if same_person is not None:
                new_persons.remove(same_person)
                person.from_person(same_person)

    @staticmethod
    def filter_and_update_overlap(active_persons: PersonDict, new_persons: list[Person], circular: CircularCoordinates) -> None:
        for person in new_persons:
            if person.world_angle and circular.angle_in_overlap(person.world_angle, 1.3):
                person.overlap = False

        # update overlap persons
        for person in active_persons.values():
            if not circular.angle_in_overlap(person.world_angle, 1.3):
                person.overlap = False
            else:
                person.overlap = True

                overlap_persons: list[Person] = []
                for new_person in new_persons:
                    angle_diff: float = abs(person.world_angle - new_person.world_angle)
                    angle_diff = min(angle_diff, 360 - angle_diff)  # Ensure the angle difference is within 0 to 180 degrees
                    if angle_diff < 13:
                        # also check roi
                        overlap_persons.append(new_person)

                if overlap_persons:
                    for p in overlap_persons:
                        new_persons.remove(p)
                        # chose the person furthest away from the edge
                        if (circular.angle_from_edge(person.local_angle) < circular.angle_from_edge(p.local_angle) * 0.9):
                            person.from_person(overlap_persons[0])

    @staticmethod
    def add_persons(active_persons: PersonDict, new_persons: list[Person], person_id_pool: IdPool) -> None:
        for person in new_persons:
            try:
                person_id: int = person_id_pool.acquire()
                print('New person id:', person_id, 'for cam', person.cam_id, 'tracklet', person.tracklet.id, 'overlap:', person.overlap == FilterType.OVERLAP, 'angle:', person.local_angle, 'world:', person.world_angle)
            except:
                print('No more person ids available')
                continue

            person.id = person_id
            active_persons[person_id] = person

    @ staticmethod
    def cleanup_inactive_persons(persons: PersonDict, person_id_pool: IdPool, activity_duration: float = 1.0) -> None:
        rejected_persons: list[Person] = []
        for key in persons.keys():
            person: Person = persons[key]
            person = persons[key]
            if person.last_time < time() - activity_duration:
                person_id_pool.release(person.id)
                rejected_persons.append(person)
                person.active = False

        # remove inactive persons
        for person in rejected_persons:
            persons.pop(person.id, None)
            print('Remove inactive person id:', person.id, 'cam', person.cam_id, 'tracklet', person.tracklet.id, 'overlap:', person.overlap == FilterType.OVERLAP, 'angle:', person.local_angle, 'world:', person.world_angle)

    @ staticmethod
    def detect_pose(person: Person, detector:PoseDetection | None, callback: PersonCallback) -> None:
        if person.pose is not None or detector is None:
            callback(person)
            return
        detector.set_detection(person)

    # INPUTS
    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None :
        if frame_type != FrameType.VIDEO:
            return
        with self.input_mutex:
            self.input_frames[id] = image
    def get_image(self, id: int) -> np.ndarray:
        with self.input_mutex:
            if self.input_frames.get(id) is None:
                # print('No image with id', id)
                return np.zeros((self.pose_detector_frame_size, self.pose_detector_frame_size, 3), np.uint8)
            return self.input_frames[id]

    def add_tracklet(self, cam_id: int, tracklet: Tracklet) -> None :
        with self.input_mutex:
            tracklets: list[Tracklet] = self.input_tracklets.get(cam_id, [])
            tracklets.append(tracklet)
            self.input_tracklets[cam_id] = tracklets
    def get_tracklets(self) -> CamTrackletDict:
        with self.input_mutex:
            tracklets: CamTrackletDict =  self.input_tracklets.copy()
            self.input_tracklets.clear()
            return tracklets

    # CALLBACKS
    def callback(self, detection: Person) -> None:
        for c in self.callbacks:
            c(detection)
    def addCallback(self, callback: PersonCallback) -> None:
        if self.running:
            print('Manager is running, cannot add callback')
            return
        self.callbacks.add(callback)


