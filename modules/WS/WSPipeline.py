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
from modules.WS.WSSettings import WSSettings, CompMode
from modules.WS.WSOutput import WSOutput, WSOutputCallback
from modules.WS.WSUdpSender import WSUdpSender, WSUdpSenderSettings

from modules.Settings import Settings
from modules.pose.Pose import Pose, PoseDict
from modules.pose.similarity.Stream import StreamData

from modules.gl.Utils import FpsCounter


from modules.utils.HotReloadMethods import HotReloadMethods



class WSPipeline(Thread):
    def __init__(self, gui, general_settings: Settings) -> None:
        super().__init__()

        self._stop_event = Event()
        self.pose_input_queue: Queue[Pose] = Queue()
        self.pose_stream_input_queue: Queue[StreamData] = Queue()

        rate: int = general_settings.light_rate
        self.interval: float = 1.0 / rate
        resolution: int = general_settings.light_resolution
        num_players: int = general_settings.num_players

        self.settings = WSSettings(
            data_settings = WSDataSettings(),
            draw_settings = WSDrawSettings(),
        )

        self.output: WSOutput = WSOutput(resolution)

        self.data_manager: WSDataManager = WSDataManager(rate, num_players, self.settings.data_settings)

        self.comp: WSDraw = WSDraw(resolution, num_players, self.interval, self.data_manager, self.settings.draw_settings)

        self.comp_test: WSDrawTest = WSDrawTest(resolution)

        light_sender_settings: WSUdpSenderSettings = WSUdpSenderSettings(
            resolution=resolution,
            port=general_settings.udp_port,
            ip_addresses=general_settings.udp_ips_light,
            send_info=False,
            use_signed=False,
        )
        self.light_sender: WSUdpSender = WSUdpSender(light_sender_settings)

        sound_sender_settings: WSUdpSenderSettings = WSUdpSenderSettings(
            resolution=resolution,
            port=general_settings.udp_port,
            ip_addresses=general_settings.udp_ips_sound,
            send_info=True,
            use_signed=False,
        )
        self.sound_sender: WSUdpSender = WSUdpSender(sound_sender_settings)

        self.FPS: FpsCounter = FpsCounter()
        self.gui: WSGui = WSGui(gui, self, self.settings)

        self.output_callbacks: list[WSOutputCallback] = []
        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def start(self) -> None:
        self.light_sender.start()
        self.sound_sender.start()
        super().start()

    def stop(self) -> None:
        self.light_sender.stop()
        self.sound_sender.stop()
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
        streams: list[StreamData] = self.get_pose_streams()

        self.data_manager.add_poses(poses)
        self.data_manager.add_streams(streams)
        self.data_manager.update()

        self.comp.update()

        self.output = self.comp.get_output()

        if self.comp_test.pattern != TestPattern.NONE:
            self.comp_test.update()
            self.output.light_img = self.comp_test.output_img

        self.light_sender.send_message(self.output)
        self.sound_sender.send_message(self.output)

        self._output_callback(self.output)

        self.FPS.tick()
        self.gui.update_fps(self.FPS.get_fps())

    # SETTERS
    def add_poses(self, poses: PoseDict) -> None:
        for pose in poses.values():
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

    def add_pose_stream(self, stream: StreamData) -> None:
        self.pose_stream_input_queue.put(stream)

    def get_pose_streams(self) -> list[StreamData]:
        streams: list[StreamData] = []
        try:
            while True:
                stream: StreamData = self.pose_stream_input_queue.get_nowait()
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