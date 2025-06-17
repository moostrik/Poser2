from __future__ import annotations

import cv2
import numpy as np
from threading import Thread, Lock
from time import time, sleep

from modules.av.Definitions import *
from modules.av.Gui import Gui

from modules.Settings import Settings
from modules.person.Person import Person, PersonDict


class Composition(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.gui: Gui = gui
        self.settings: Settings = settings

        self.running: bool = False
        self.interval: float = 1.0 / settings.light_rate
        self.last_update: float = 0.0

        self.person_dict: PersonDict = {}
        self.dict_lock: Lock = Lock()

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        self.running = True
        self.last_update = time()
        while self.running:
            persons: PersonDict = self.get_person_dict()

            if persons:
                pass

            # sleep until next update
            now: float = time()
            if now - self.last_update < self.interval:
                sleep(self.interval - (now - self.last_update))
            self.last_update = now

    def set_person_dict(self, person_dict: PersonDict) -> None:
        with self.dict_lock:
                self.person_dict = person_dict

    def get_person_dict(self) -> PersonDict:
        with self.dict_lock:
            return self.person_dict.copy()