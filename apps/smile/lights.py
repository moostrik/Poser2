"""The room's lights over Art-Net: the cores (the PCs' modules) breathe, and the ring around the camera
follows the sequence (`set_progress`). `data/artnet.json` lists the modules in DMX order: a Core has 3
RGB LEDs, a Camera 24."""

from threading import Lock
import json
import logging
import math
import time

import numpy as np

from modules.inout import ArtNetSender, ArtNetSenderSettings
from apps.smile.render.Sequencer import AnimationType

logger = logging.getLogger(__name__)

CORE_LEDS: int = 3
CAMERA_LEDS: int = 24


def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)


def lfo(time: float, Htz: float, phase: float) -> float:
    return 0.5 * math.sin(time * math.pi * Htz + (0.75 + phase) * math.pi * 2) + 0.5


def lfor(time: float, Htz: float, phase: float, rangeMin: float, rangeMax: float) -> float:
    return lfo(time, Htz, phase) * (rangeMax - rangeMin) + rangeMin


def get_counter_active_light(value: float) -> list[int]:
    if value < 1:
        return [6, 18]
    if value < 2:
        return [5, 7, 17, 19]
    if value < 3:
        return [4, 8, 16, 20]
    if value < 4:
        return [3, 9, 15, 21]
    if value < 5:
        return [2, 10, 14, 22]
    if value < 6:
        return [1, 11, 13, 23]
    if value < 7:
        return [0, 12]
    else:
        return []


def get_counter_lights(value: float) -> list[int]:
    if value < 1:
        return []
    if value < 2:
        return [6, 18]
    if value < 3:
        return [5, 6, 7, 17, 18, 19]
    if value < 4:
        return [4, 5, 6, 7, 8, 16, 17, 18, 19, 20]
    if value < 5:
        return [3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21]
    if value < 6:
        return [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    if value < 7:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    else:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


class Lights:
    def __init__(self, settings: ArtNetSenderSettings, modules_file: str) -> None:
        self.cores: dict[str, int] = {}
        self.cameras: dict[str, int] = {}
        self.modules: dict = read_json_file(modules_file)

        address: int = 0
        for name, module in self.modules.items():
            if module['type'] == "Core":
                self.cores[name] = address
                address += CORE_LEDS * 3
            if module['type'] == "Camera":
                self.cameras[name] = address
                address += CAMERA_LEDS * 3

        self.sender: ArtNetSender = ArtNetSender(settings, address + 1, self.fill)
        self.start_time: float = time.time() + 3
        self.cam_lights: np.ndarray = np.zeros((CAMERA_LEDS, 3), dtype=float)

    def start(self) -> None:
        self.sender.start()

    def stop(self) -> None:
        self.sender.stop()

    def fill(self, packet: bytearray) -> None:
        self.update_module(packet)
        self.update_cameras(packet)

    def update_module(self, packet: bytearray) -> None:
        current_time: float = max(time.time() - self.start_time, 0)
        breath: float = lfor(current_time, 0.1666, 0, 0.05, 1)
        b: int = min(max(int(breath * 255), 0), 255)

        for a in self.cores.values():
            for i in range(CORE_LEDS):
                packet[a + i * 3 + 1] = b

    def update_cameras(self, packet: bytearray) -> None:
        lights: np.ndarray = self.cam_lights    # one reference: the setters replace the array
        power = 2.0
        byte_array: bytes = (lights ** power * 255).astype(np.uint8).tobytes()
        for a in self.cameras.values():
            packet[a:a + CAMERA_LEDS * 3] = byte_array

    # the setters build the ring and put it in place in one step: the sender thread reads cam_lights
    # between any two lines, and a ring that is half filled in is a dark frame on the lights

    def set_camera_counter(self, value: float, red: bool = False) -> None:
        lights: np.ndarray = np.zeros_like(self.cam_lights)
        value = max(min(value, 1.0), 0.0) * 7.0
        v: float = value - math.floor(value)

        leds: list[int] = get_counter_lights(value)
        active_leds: list[int] = get_counter_active_light(value)

        if red:
            lights[leds, 0] = 1
            lights[active_leds, 0] = v
        else:
            lights[leds, :] = 1
            lights[active_leds, :] = v
        self.cam_lights = lights

    def set_camera_white(self, value: float) -> None:
        value = max(min(value, 1.0), 0.0)
        self.cam_lights = np.full_like(self.cam_lights, value)

    def set_camera_red(self, value: float) -> None:
        value = max(min(value, 1.0), 0.0)
        lights: np.ndarray = np.zeros_like(self.cam_lights)
        lights[:, 0] = value
        self.cam_lights = lights

    def set_camera_black(self) -> None:
        self.cam_lights = np.zeros_like(self.cam_lights)

    def set_progress(self, progress: float, type: AnimationType) -> None:
        if type == AnimationType.NONE:
            self.set_camera_red(1.0)
        if type == AnimationType.DESTRUCT_IN:
            self.set_camera_red(1.0)
        if type == AnimationType.DESTRUCT_OUT:
            self.set_camera_counter(1.0 - progress, True)
        if type == AnimationType.CAMERA_IN:
            self.set_camera_white(progress * 0.2)
        if type == AnimationType.CAMERA_OUT:
            self.set_camera_white(0.2 + progress * 0.8)
        if type == AnimationType.FLASH_IN:
            self.set_camera_white(1.0)
        if type == AnimationType.FLASH_OUT:
            self.set_camera_white(1.0 - progress)
        if type == AnimationType.DIFFUSE_IN:
            self.set_camera_counter(progress * 0.1, True)
        if type == AnimationType.DIFFUSE_OUT:
            self.set_camera_counter(0.1 + progress * 0.9, True)
