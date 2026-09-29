from ipaddress import ip_address
from threading import Thread, Lock
import time
from stupidArtnet import StupidArtnet
from typing import Optional, Any
import json
import math
from modules.render.Sequencer import AnimationType
import numpy as np


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

def is_valid_universe(value: int | str) -> bool:
    if type(value) == str:
        try:
            value = int(value)
        except ValueError:
            return False
    value = int(value)
    if value in range(0, 255):
        return True
    return False


def is_valid_fps(value: float | str) -> bool:
    if type(value) == str:
        try:
            value = float(value)
        except ValueError:
            return False
    value = float(value)
    if 1.0 <= value <= 60.0:
        return True
    return False

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
    def __init__(self, settings_file: str, ip: str = "127.0.0.1", universe: int = 1, fps: float = 30.0):
        super().__init__()

        self.mutex: Lock = Lock()

        self.running: bool = False
        self.sending: bool = False

        self.ip: str = '10.0.0.11'
        # self.ip: str = '192.168.0.111'
        self.universe: int = 1
        self.fps: float = 40.0

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

        self.packet_size: int = address + 1

        self.packet = bytearray(self.packet_size)

        self.sender: Optional[StupidArtnet] = None

        self.start_time: float = time.time() + 3

        self.cam_r: float = 0
        self.cam_g: float = 0
        self.cam_b: float = 0

        self.cam_lights: np.ndarray =  np.zeros((self.camera_led_num, 3), dtype= float)

    def start(self) -> None :
        self.updateSender()
        self.running = True
        self.sending = True
        super().start()

    def stop(self) -> None :
        if not self.is_alive():
            return

        self.running = False

        if self.sender:
            self.sender.flash_all()
            time.sleep(1.0/self.fps)
            self.sender.blackout()

        self.join()

    def run(self) -> None:
        while self.running:
            self.update()
            time.sleep(1.0/self.fps)

    def update(self) -> None:
        self.update_module()
        self.update_cameras()

        if self.sender and self.sending:
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

        for name in self.cameras:
            a: int = self.cameras[name]
            power = 2.0
            scaled_matrix = (self.cam_lights ** power * 255).astype(np.uint8)
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

    def set_universe(self, value: int | str) -> None:
        if not is_valid_universe(value):
            return
        value = int(value)
        if self.universe == value:
            return
        self.universe = value
        if self.sender:
            self.sender.set_universe(value)
        self.flash()

    def set_sending(self, value: bool) -> None:
        self.sending = value
        self.flash()

    def set_fps(self, value: float | str) -> None:
        if not is_valid_fps(value):
            return
        value = float(value)
        if self.fps == value:
            return
        self.fps = value

        self.flash()

    def set_ip(self, value: str) -> None:
        if not is_valid_ip_address(value):
            return
        if self.ip == value:
            return
        self.ip = value
        if self.running:
            self.updateSender()

    def updateSender(self) -> None:
        with self.mutex:

            if self.sender:
                self.sender.stop()

            self.sender = StupidArtnet(target_ip = self.ip,
                                       universe = self.universe,
                                       packet_size = self.packet_size,
                                       fps = int(self.fps))
            print('Artnet sending to', self.ip, 'on universe', self.universe)

        self.flash()

    def flash(self) -> None:
        if not self.sending:
            return
        if not self.sender:
            return

        with self.mutex:
            self.sender.flash_all()
            time.sleep(1.0/self.fps)


    def set_camera_counter(self, value: float, red: bool = False) -> None:
        self.cam_lights[:] = 0.0
        value = max(min(value, 1.0), 0.0) * 7.0
        v: float = value - math.floor(value)

        leds: list[int] = get_counter_lights(value)

        active_leds: list[int] = get_counter_active_light(value)

        if red:
            self.cam_lights[leds, 0] = 1
            self.cam_lights[active_leds, 0] = v
        else:
            self.cam_lights[leds, :] = 1
            self.cam_lights[active_leds, :] = v



    def set_camera_white(self, value: float) -> None:
        value = max(min(value, 1.0), 0.0)
        self.cam_lights[:, 0] = value
        self.cam_lights[:, 1] = value
        self.cam_lights[:, 2] = value

    def set_camera_red(self, value: float) -> None:
        value = max(min(value, 1.0), 0.0)
        self.cam_lights[:, 0] = value
        self.cam_lights[:, 1] = 0 #value ** 3.0
        self.cam_lights[:, 2] = 0 #value ** 3.0

    def set_camera_black(self) -> None:
        self.cam_lights[:] = 0.0

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