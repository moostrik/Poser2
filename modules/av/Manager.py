# Standard library imports
from threading import Thread, Lock
from time import time, sleep

# Third-party imports

# Local application imports
from modules.av.Definitions import *
from modules.av.Gui import Gui
from modules.av.CompTest import CompTest
from modules.av.UdpSender import UdpSender
from modules.Settings import Settings
from modules.tracker.Tracklet import Tracklet, TrackletDict

from modules.gl.Utils import FpsCounter


class Manager(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()
        self.settings: Settings = settings

        self.running: bool = False
        self.interval: float = 1.0 / settings.light_rate
        self.last_update: float = 0.0

        self.person_dict: TrackletDict = {}
        self.dict_lock: Lock = Lock()

        self.resolution: int = settings.light_resolution
        self.output: AvOutput = AvOutput(self.resolution)

        self.comp_test: CompTest = CompTest(self.resolution)

        self.output_callbacks: list[AvOutputCallback] = []

        self.udp_sender: UdpSender = UdpSender(self.settings.light_resolution, self.settings.udp_port, self.settings.udp_ip_addresses)
        self.udp_sender.start()

        self.FPS: FpsCounter = FpsCounter()

        self.gui: Gui = Gui(gui, self)

    def stop(self) -> None:
        self.udp_sender.stop()
        self.running = False

    def run(self) -> None:
        self.running = True
        next_time: float = time()
        while self.running:

            self.output.img = self.comp_test.make_pattern()

            self._output_callback(self.output)
            self.udp_sender.send_message(self.output)

            next_time += self.interval
            sleep_time: float = next_time - time()
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                next_time = time()

            self.FPS.tick()
                # print(f"Manager fell behind by {-sleep_time:.4f} seconds, or: {-sleep_time / self.interval:.4f} frames")

    # STATIC METHODS


    # SETTERS AND GETTERS
    def set_person_dict(self, person_dict: TrackletDict) -> None:
        with self.dict_lock:
                self.person_dict = person_dict

    def get_person_dict(self) -> TrackletDict:
        with self.dict_lock:
            return self.person_dict.copy()

    def _output_callback(self, output: AvOutput) -> None:
        for callback in self.output_callbacks:
            callback(output)

    def add_output_callback(self, output: AvOutputCallback) -> None:
        self.output_callbacks.append(output)