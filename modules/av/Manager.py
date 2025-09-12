# Standard library imports
from queue import Empty, Queue
from threading import Event, Thread, Lock
from time import time, sleep

# Third-party imports

# Local application imports
from modules.av.Definitions import *
from modules.av.Gui import Gui
from modules.av.Comp import Comp
from modules.av.CompTest import CompTest, TestPattern
from modules.av.UdpSender import UdpSender
from modules.Settings import Settings
from modules.pose.PoseDefinitions import Pose

from modules.gl.Utils import FpsCounter


from modules.utils.HotReloadMethods import HotReloadMethods


class Manager(Thread):
    def __init__(self, gui, settings: Settings) -> None:
        super().__init__()
        self.settings: Settings = settings

        self._stop_event = Event()
        self.interval: float = 1.0 / settings.light_rate
        self.last_update: float = 0.0

        self.resolution: int = settings.light_resolution
        self.output: AvOutput = AvOutput(self.resolution)

        self.pose_input_queue: Queue[Pose] = Queue()

        self.comp: Comp = Comp(settings)
        self.comp_test: CompTest = CompTest(self.resolution)

        self.udp_sender: UdpSender = UdpSender(self.settings.light_resolution, self.settings.udp_port, self.settings.udp_ip_addresses)
        self.udp_sender.start()

        self.FPS: FpsCounter = FpsCounter()
        self.gui: Gui = Gui(gui, self, self.comp.settings)

        self.output_callbacks: list[AvOutputCallback] = []
        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def stop(self) -> None:
        self.udp_sender.stop()
        self._stop_event.set()
        self.join()

    def run(self) -> None:
        next_time: float = time()
        while not self._stop_event.is_set():

            poses: list[Pose] = []
            try:
                while True:
                    pose: Pose = self.pose_input_queue.get_nowait()
                    poses.append(pose)
            except Empty:
                pass

            self._update(poses)

            next_time += self.interval
            sleep_time: float = next_time - time()
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                next_time = time()

            self.FPS.tick()
            self.gui.update()

    def _update(self, poses: list[Pose]) -> None:
        # comp_img: np.ndarray = self.comp.update(poses)
        try:
            comp_img: np.ndarray = self.comp.update(poses)
            self.output.img = comp_img
            self.output.test = self.comp.output_comp
        except Exception as e:
            print(f"Error in Comp update: {e}")
            return

        if self.comp_test.pattern != TestPattern.NONE:
            self.output.img = self.comp_test.update()

        self._output_callback(self.output)
        self.udp_sender.send_message(self.output)

    # SETTERS
    def add_pose(self, pose: Pose) -> None:
        self.pose_input_queue.put(pose)

    # CALLBACKS
    def _output_callback(self, output: AvOutput) -> None:
        for callback in self.output_callbacks:
            callback(output)

    def add_output_callback(self, output: AvOutputCallback) -> None:
        self.output_callbacks.append(output)