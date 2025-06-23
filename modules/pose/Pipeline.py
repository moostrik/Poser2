from __future__ import annotations

import cv2
import numpy as np
from threading import Thread, Lock
from time import time, sleep

from modules.Settings import Settings
from modules.person.Person import Person, PersonCallback
from modules.pose.Definitions import PoseList
from modules.pose.Detection import Detection
from modules.pose.Window import Window

class Pipeline(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        
        self.running: bool = False
        self.person_mutex: Lock = Lock()
        self.max_persons: int = settings.pose_num
        
        # Input queue for persons to process
        self.input_persons: dict[int, Person] = {}
        
        # Pose detection components
        self.pose_detectors: dict[int, Detection] = {}
        self.pose_windows: dict[int, Window] = {}
        self.pose_model_path: str = settings.path_model
        self.pose_model_type = settings.pose_model_type
        
        # Time window parameters
        self.window_size: float = settings.pose_window_size
        self.window_step: float = settings.pose_window_step
        
        # Setup pose detectors if enabled
        if not settings.pose_active:
            print('Pose Processing Pipeline: Disabled')
        else:
            # Initialize detectors for each person
            for i in range(self.max_persons):
                self.pose_detectors[i] = Detection(self.pose_model_path, self.pose_model_type)
                self.pose_windows[i] = Window(window_size=self.window_size, step=self.window_step)
            print(f'Pose Processing Pipeline: Initialized with {self.max_persons} processing units')
        
        # Callbacks
        self.person_callbacks: set[PersonCallback] = set()
    
    def start(self) -> None:
        if self.running:
            return
        
        # Start detectors
        for detector in self.pose_detectors.values():
            detector.addMessageCallback(self._person_callback)
            detector.start()
        
        self.running = True
        super().start()
    
    def stop(self) -> None:
        self.running = False
        self.person_callbacks.clear()
        
        # Stop detectors
        for detector in self.pose_detectors.values():
            detector.stop()
    
    def run(self) -> None:
        while self.running:
            self.process_persons()
            sleep(0.01)
    
    def process_persons(self) -> None:
        persons_to_process = self.get_persons()
        
        for person in persons_to_process.values():
            # Skip if person already has a pose or no detector available
            if person.pose is not None or person.id not in self.pose_detectors:
                self._person_callback(person)
                continue
            
            # Send to detector
            detector = self.pose_detectors.get(person.id)
            if detector:
                detector.set_detection(person)
    
    def add_person(self, person: Person) -> None:
        """Add a person to the processing queue"""
        with self.person_mutex:
            self.input_persons[person.id] = person
    
    def get_persons(self) -> dict[int, Person]:
        """Get and clear the person queue"""
        with self.person_mutex:
            persons = self.input_persons.copy()
            self.input_persons.clear()
            return persons
    
    def process_pose_window(self, person: Person) -> None:
        """Process pose through time window if available"""
        if person.pose is None or person.id not in self.pose_windows:
            return
        
        window = self.pose_windows.get(person.id)
        if window:
            window.add_pose(person.pose)
            # Here you can add additional processing on the window data
            # For example, filtering, correlation analysis, etc.
    
    def add_person_callback(self, callback: PersonCallback) -> None:
        """Add callback for processed persons"""
        if self.running:
            print('Pipeline is running, cannot add callback')
            return
        self.person_callbacks.add(callback)
    
    def _person_callback(self, person: Person) -> None:
        """Handle processed person, applying additional processing if needed"""
        # Apply window processing if pose is available
        self.process_pose_window(person)
        
        # Forward to callbacks
        for callback in self.person_callbacks:
            callback(person)