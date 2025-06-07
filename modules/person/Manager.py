from __future__ import annotations

import cv2
import numpy as np
from threading import Thread, Lock
from time import time, sleep

from modules.Settings import Settings
from modules.person.Gui import Gui
from modules.person.Person import Person, PersonDict, PersonCallback
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType
from modules.person.pose.Detection import Detection, ModelType, ModelTypeNames
from modules.person.Camera360Array import Camera360Array
from modules.person.Definitions import *

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

        self.cam_360: Camera360Array = Camera360Array(settings.num_cams, CAM_360_FOV, CAM_360_TARGET_FOV)

        self.pose_detectors: dict[int, Detection] = {}
        self.pose_detector_frame_size: int = 256
        if not settings.pose:
            print('Pose Detection: Disabled')
        else:
            for i in range(self.max_persons):
                self.pose_detectors[i] = Detection(settings.model_path, settings.model_type)
            print('Pose Detection:', self.max_persons, 'instances of model', ModelTypeNames[settings.model_type.value])

        self.pose_roi_expansion: float = PERSON_ROI_EXPANSION
        self.person_timeout: float = PERSON_TIMEOUT

        self.min_tracklet_age: int = MIN_TRACKLET_AGE
        self.min_tracklet_height: float = MIN_TRACKLET_HEIGHT
        self.cam_360_edge_threshold: float = CAM_360_EDGE_THRESHOLD
        self.cam_360_overlap_expansion: float = CAM_360_OVERLAP_EXPANSION
        self.cam_360_hysteresis_factor: float = CAM_360_HYSTERESIS_FACTOR

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
        self.filter_age(tracklet_persons, self.min_tracklet_age)
        self.filter_size(tracklet_persons, self.min_tracklet_height)
        self.cam_360.calc_angles(tracklet_persons)
        self.filter_edge(tracklet_persons, self.cam_360, self.cam_360_edge_threshold)
        self.filter_and_update_active(self.persons, tracklet_persons)
        self.filter_and_update_overlap(self.persons, tracklet_persons, self.cam_360, self.cam_360_overlap_expansion, self.cam_360_hysteresis_factor)
        self.add_persons(self.persons, tracklet_persons, self.person_id_pool)
        self.cleanup_inactive_persons(self.persons, self.person_id_pool, self.person_timeout)

        for person in self.persons.values():
            image: np.ndarray = self.get_image(person.cam_id)
            person.set_pose_roi(image, self.pose_roi_expansion)
            person.set_pose_image(image)
            detector: Detection | None = self.pose_detectors.get(person.id, None)
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
    def filter_edge(persons: list[Person], circular: Camera360Array, edge_range: float) -> None:
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
                person.update_from(same_person)

    @staticmethod
    def filter_and_update_overlap(active_persons: PersonDict, new_persons: list[Person], radial: Camera360Array,
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
    def add_persons(active_persons: PersonDict, new_persons: list[Person], person_id_pool: IdPool) -> None:
        for person in new_persons:
            try:
                person_id: int = person_id_pool.acquire()
                # print('New person id:', person_id, 'for cam', person.cam_id, 'tracklet', person.tracklet.id, 'overlap:', person.overlap, 'angle:', person.local_angle, 'world:', person.world_angle)
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
            # print('Remove inactive person id:', person.id, 'cam', person.cam_id, 'tracklet', person.tracklet.id, 'overlap:', person.overlap, 'angle:', person.local_angle, 'world:', person.world_angle)

    @ staticmethod
    def detect_pose(person: Person, detector:Detection | None, callback: PersonCallback) -> None:
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


