from ipaddress import ip_address
from threading import Thread, Lock
import time
from stupidArtnet import StupidArtnet
from typing import Optional, Any
import json
import math
from apps.smile.render.Sequencer import AnimationType
from apps.smile.artnet.settings import ArtnetSettings
import logging
import numpy as np

logger = logging.getLogger(__name__)


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def lfo(time: float, Htz: float, phase: float) -> float:
    return 0.5 * math.sin(time * math.pi * Htz + (0.75 + phase) * math.pi * 2) + 0.5

def lfor(time: float, Htz: float, phase: float, rangeMin: float, rangeMax: float) -> float:
    return lfo(time, Htz, phase) * (rangeMax - rangeMin) + rangeMin

def is_valid_ip_address(ip: str) -> bool:
    try:
        ip_address(ip)
        return True
    except ValueError:
        return False

def is_valid_universe(value: int) -> bool:
    return value in range(0, 255)

def get_counter_active_light(value: float)  -> list[int]:
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

def get_counter_lights(value: float)  -> list[int]:
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


class Sender(Thread):
    def __init__(self, settings: ArtnetSettings, settings_file: str):
        super().__init__()
        self.settings: ArtnetSettings = settings

        self.mutex: Lock = Lock()

        self.running: bool = False

        self.core_led_num = 3
        self.camera_led_num = 24

        self.cores: dict = {}
        self.cameras: dict = {}
        self.modules = read_json_file(settings_file)

        address: int = 0
        for name in self.modules:
            module = self.modules[name]
            if module['type'] == "Core":
                 self.cores[name] = address
                 address += self.core_led_num * 3
            if module['type'] == "Camera":
                 self.cameras[name] = address
                 address += self.camera_led_num * 3

        # even: StupidArtnet rounds its own size up to even and refuses a packet of another length
        self.packet_size: int = address + 1
        self.packet_size += self.packet_size % 2

        self.packet = bytearray(self.packet_size)

        self.sender: Optional[StupidArtnet] = None

        self.start_time: float = time.time() + 3

        self.cam_r: float = 0
        self.cam_g: float = 0
        self.cam_b: float = 0

        self.cam_lights: np.ndarray =  np.zeros((self.camera_led_num, 3), dtype= float)

    def start(self) -> None :
        self.updateSender()
        self.settings.bind(ArtnetSettings.ip, self._on_address)
        self.settings.bind(ArtnetSettings.universe, self._on_address)
        self.settings.bind(ArtnetSettings.active, self._on_active)
        self.running = True
        super().start()

    def stop(self) -> None :
        if not self.is_alive():
            return

        self.settings.unbind(ArtnetSettings.ip, self._on_address)
        self.settings.unbind(ArtnetSettings.universe, self._on_address)
        self.settings.unbind(ArtnetSettings.active, self._on_active)
        self.running = False

        if self.sender:
            self.sender.flash_all()
            time.sleep(self._interval())
            self.sender.blackout()

        self.join()

    def run(self) -> None:
        while self.running:
            self.update()
            time.sleep(self._interval())

    def _interval(self) -> float:
        return 1.0 / max(self.settings.fps, 1.0)

    def update(self) -> None:
        self.update_module()
        self.update_cameras()

        if self.sender and self.settings.active:
            with self.mutex:
                self.sender.send(self.packet)

    def update_module(self) -> None:
        current_time: float = max(time.time() - self.start_time, 0)
        breath: float  = self.fade_power_lfo(current_time, 0.1666, 0, 0.05, 1, 2.0)
        breath = lfor(current_time, 0.1666, 0, 0.05, 1)
        # breath = breath ** 2.0
        b: int = min(max(int(breath * 255), 0), 255)

        for name in self.cores:
            a: int = self.cores[name]
            for i in range(self.core_led_num):
                adress: int = a + i * 3 + 1
                self.packet[adress] = b

    def update_cameras(self) -> None:

        lights: np.ndarray = self.cam_lights    # one reference: the setters replace the array
        for name in self.cameras:
            a: int = self.cameras[name]
            power = 2.0
            scaled_matrix = (lights ** power * 255).astype(np.uint8)
            byte_array = scaled_matrix.tobytes()
            # print(byte_array)
            for i in range(self.camera_led_num * 3):
                self.packet[a + i] = byte_array[i]
                # adress: int = a + i
                # power = 2.0
                # self.packet[adress] = int((self.cam_r ** power) * 255)
                # self.packet[adress + 1] = int((self.cam_b ** power) * 255)
                # self.packet[adress + 2] = int((self.cam_g ** power) * 255)

    def fade_power_lfo(self, time, Htz, phase, range_min, range_max, power) -> float:
        base: float = lfor(time, Htz, phase, range_min, range_max)
        fade: float = min(time / (1.0 / Htz * 1.5), 1.0)
        base *= fade
        return base ** power

    def _on_address(self, _=None) -> None:
        self.updateSender()

    def _on_active(self, _=None) -> None:
        self.flash()

    def updateSender(self) -> None:
        ip: str = self.settings.ip
        universe: int = self.settings.universe
        if not is_valid_ip_address(ip) or not is_valid_universe(universe):
            logger.warning('Invalid Art-Net address %s universe %s, keeping the current sender', ip, universe)
            return

        with self.mutex:

            if self.sender:
                self.sender.stop()

            self.sender = StupidArtnet(target_ip = ip,
                                       universe = universe,
                                       packet_size = self.packet_size,
                                       fps = int(self.settings.fps))
            logger.info('Artnet sending to %s on universe %s', ip, universe)

        self.flash()

    def flash(self) -> None:
        if not self.settings.active:
            return
        if not self.sender:
            return

        with self.mutex:
            self.sender.flash_all()
            time.sleep(self._interval())


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
        # self.set_camera_counter(progress)
        # return

        if type == AnimationType.NONE:
            self.set_camera_red(1.0)
            pass
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

        pass