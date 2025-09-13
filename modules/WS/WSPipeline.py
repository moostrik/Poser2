# Standard library imports
from queue import Empty, Queue
from threading import Event, Thread, Lock
from time import time, sleep

# Third-party imports

# Local application imports
from modules.WS.WSDataManager import WSDataManager, WSDataSettings
from modules.WS.WSDraw import WSDraw, WSDrawSettings
from modules.WS.WSDrawTest import WSDrawTest, TestPattern
from modules.WS.WSGui import WSGui
from modules.WS.WSSettings import WSSettings
from modules.WS.WSOutput import WSOutput, WSOutputCallback
from modules.WS.UdpSender import UdpSender

from modules.Settings import Settings
from modules.pose.PoseDefinitions import Pose
from modules.pose.PoseStream import PoseStreamData

from modules.gl.Utils import FpsCounter


from modules.utils.HotReloadMethods import HotReloadMethods


class WSPipeline(Thread):
    def __init__(self, gui, general_settings: Settings) -> None:
        super().__init__()

        self._stop_event = Event()
        self.pose_input_queue: Queue[Pose] = Queue()
        self.pose_stream_input_queue: Queue[PoseStreamData] = Queue()
        self.last_update: float = 0.0

        rate: int = general_settings.light_rate
        self.interval: float = 1.0 / rate
        resolution: int = general_settings.light_resolution
        num_players: int = general_settings.max_players

        self.settings = WSSettings(
            data_settings = WSDataSettings(),
            draw_settings = WSDrawSettings(),
        )

        self.output: WSOutput = WSOutput(resolution)

        self.data_manager: WSDataManager = WSDataManager(rate, num_players, self.settings.data_settings)

        self.comp: WSDraw = WSDraw(resolution, self.interval, self.data_manager, self.settings.draw_settings)

        self.comp_test: WSDrawTest = WSDrawTest(resolution)

        self.udp_sender: UdpSender = UdpSender(resolution, general_settings.udp_port, general_settings.udp_ip_addresses)

        self.FPS: FpsCounter = FpsCounter()
        self.gui: WSGui = WSGui(gui, self, self.settings)

        self.output_callbacks: list[WSOutputCallback] = []
        # self.hot_reloader = HotReloadMethods(self.__class__, True)

    def start(self) -> None:
        super().start()
        self.udp_sender.start()

    def stop(self) -> None:
        self.udp_sender.stop()
        self._stop_event.set()
        self.join()

    def run(self) -> None:
        next_time: float = time()
        while not self._stop_event.is_set():
            try:
                self._update()
            except Exception as e:
                print(f"Error in Comp update: {e}")

            next_time += self.interval
            sleep_time: float = next_time - time()
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                next_time = time()

    def _update(self) -> None:
        poses: list[Pose] = self.get_poses()
        streams: list[PoseStreamData] = self.get_pose_streams()

        self.data_manager.add_poses(poses)
        self.data_manager.add_streams(streams)
        self.data_manager.update()

        self.comp.update()

        self.output = self.comp.get_output()

        if self.comp_test.pattern != TestPattern.NONE:
            self.comp_test.update()
            self.output.light_img = self.comp_test.output_img

        self.udp_sender.send_message(self.output)

        self._output_callback(self.output)

        self.FPS.tick()
        self.gui.update_fps(self.FPS.get_fps())

    # SETTERS
    def add_pose(self, pose: Pose) -> None:
        self.pose_input_queue.put(pose)

    def get_poses(self) -> list[Pose]:
        poses: list[Pose] = []
        try:
            while True:
                pose: Pose = self.pose_input_queue.get_nowait()
                poses.append(pose)
        except Empty:
            pass
        return poses

    def add_pose_stream(self, stream: PoseStreamData) -> None:
        self.pose_stream_input_queue.put(stream)

    def get_pose_streams(self) -> list[PoseStreamData]:
        streams: list[PoseStreamData] = []
        try:
            while True:
                stream: PoseStreamData = self.pose_stream_input_queue.get_nowait()
                streams.append(stream)
        except Empty:
            pass
        return streams

    # CALLBACKS
    def _output_callback(self, output: WSOutput) -> None:
        for callback in self.output_callbacks:
            callback(output)

    def add_output_callback(self, output: WSOutputCallback) -> None:
        self.output_callbacks.append(output)