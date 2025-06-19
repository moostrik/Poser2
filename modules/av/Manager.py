from __future__ import annotations

import cv2
import numpy as np
from threading import Thread, Lock
from time import time, sleep
from random import uniform

from modules.av.Definitions import *
from modules.av.Gui import Gui

from modules.Settings import Settings
from modules.person.Person import Person, PersonDict


class Manager(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()

        self.gui: Gui = gui
        self.settings: Settings = settings

        self.running: bool = False
        self.interval: float = 1.0 / settings.light_rate
        self.last_update: float = 0.0

        self.person_dict: PersonDict = {}
        self.dict_lock: Lock = Lock()

        self.resolution: int = settings.light_resolution
        self.output: AvOutput = AvOutput(self.resolution)

        self.output_callbacks: list[AvOutputCallback] = []

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        self.running = True
        self.last_update = time()
        while self.running:

            output: AvOutput = AvOutput(self.resolution)
            # output.img[0, :, 0] = np.random.uniform(0.0, 255.0, self.resolution)
            # output.img[0, :, 1] = np.random.uniform(0.0, 255.0, self.resolution)

            # ramp from 0 to 255
            # output.img[0, :, 0] = np.linspace(0, 1.0, self.resolution)
            # # ramp from 255 to 0
            # output.img[0, :, 1] = np.linspace(1.0, 0, self.resolution)



            # output.img[0, :, 2] = np.random.uniform(0.0, 1.0, self.resolution)


            # set each tenth pixel to 1
            # output.img[0, ::10, 0] = 1.0

            # set each blue pixel to 1 if random is above 0.9
            # output.img[0, :, 0] = np.where(np.random.uniform(0.0, 1.0, self.resolution) > 0.9, 1.0, 0.0)

            # make the blue pixels a gradient from 0 to 1
            output.img[0, :, 0] = np.linspace(0, 1.0, self.resolution)

            self._output_callback(output)

            sleep(self.interval)

    # STATIC METHODS


    # SETTERS AND GETTERS
    def set_person_dict(self, person_dict: PersonDict) -> None:
        with self.dict_lock:
                self.person_dict = person_dict

    def get_person_dict(self) -> PersonDict:
        with self.dict_lock:
            return self.person_dict.copy()

    def _output_callback(self, output: AvOutput) -> None:
        for callback in self.output_callbacks:
            callback(output)

    def add_output_callback(self, output: AvOutputCallback) -> None:
        self.output_callbacks.append(output)