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
        tracklet_persons: list[Person] = self.get_persons_from_tracklets(tracklets)
        self.circular_coordinates.calc_angles(tracklet_persons)

        self.find_unique_persons(self.persons, tracklet_persons, self.person_id_pool, self.circular_coordinates)
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
                if tracklet.status == Tracklet.TrackingStatus.NEW or tracklet.status == Tracklet.TrackingStatus.TRACKED:
                    person: Person = Person(-1, cam_id, tracklet)
                    persons.append(person)
        return persons

    @staticmethod
    def find_unique_persons(active_persons: PersonDict, new_persons: list[Person], person_id_pool: IdPool, circular: CircularCoordinates) -> None:
        # update existing persons
        for person in active_persons.values():
            same_person: Person | None = Manager.find_same_person(person, new_persons)
            if same_person is not None:
                new_persons.remove(same_person)

                same_person.id = person.id
                same_person.start_time = person.start_time
                same_person.last_time = time()
                active_persons[same_person.id] = same_person

        # check if persons are on the edge of the camera view
        for person in active_persons.values():
            if person.angle and circular.angle_in_overlap(person.angle, 0.4):
                same_person: Person | None = None
                for new_person in new_persons:
                    angle_diff: float = abs(person.angle - new_person.angle)
                    if angle_diff < 5:
                        # also check roi
                        same_person = new_person
                if same_person is not None:
                    new_persons.remove(same_person)

                    same_person.id = person.id
                    same_person.start_time = person.start_time
                    same_person.last_time = time()
                    active_persons[same_person.id] = same_person


        for person in new_persons:
            try:
                person_id: int = person_id_pool.acquire()
            except:
                print('No more person ids available')
                continue

            person.id = person_id
            active_persons[person_id] = person

    @staticmethod
    def find_same_person(person: Person, persons: list[Person]) -> Person | None:
        for p in persons:
            if person.cam_id == p.cam_id and person.tracklet.id == p.tracklet.id:
                return p
        return None

    @ staticmethod
    def cleanup_inactive_persons(persons: PersonDict, person_id_pool: IdPool, activity_duration: float = 1.0) -> None:
        for key in persons.keys():
            person: Person = persons[key]
            person = persons[key]
            if person.last_time < time() - activity_duration:
                person_id_pool.release(person.id)
                person.active = False

        # remove inactive persons
        persons = {k: v for k, v in persons.items() if v.active}

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


