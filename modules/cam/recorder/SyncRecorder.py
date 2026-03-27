
from threading import Thread, Lock, Event
from pathlib import Path
import time
from enum import Enum, auto
from numpy import ndarray
from queue import Queue, Empty

from modules.cam.CamSettings import CameraSettings
from modules.cam.depthcam.Definitions import CoderType, CoderFormat, FrameType, FRAME_TYPE_LABEL_DICT
from modules.cam.recorder.RecorderSettings import RecorderSettings
from modules.cam.recorder.FFmpegRecorder import FFmpegRecorder

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

def make_file_name(c: int, t: FrameType, chunk: int, format: str) -> str:
    return f"{c}_{FRAME_TYPE_LABEL_DICT[t]}_{chunk:03d}{format}"

def make_folder_name(num_cams: int, square: bool, color: bool, stereo: bool, group_id: str ="") -> str:
    return time.strftime("%Y%m%d-%H%M%S") + '_' + str(num_cams) + ('_square' if square else '_wide') + ('_color' if color else '_mono') + ('_stereo' if stereo else '') + '_' + group_id

def is_folder_for_settings(name: str, settings: CameraSettings) -> bool:
    parts: list[str] = name.split('_')
    if len(parts) < 2:
        return False
    if not parts[1].isdigit() or not  ('wide' in parts or 'square' in parts) or not ('color' in parts or 'mono' in parts):
        return False
    num_cams = int(parts[1])
    square: bool = 'square' in parts
    color: bool = 'color' in parts
    stereo: bool = 'stereo' in parts
    if num_cams >= settings.num_cameras and square == settings.square and color == settings.color and stereo == settings.stereo:
        return True
    return False

EncoderString: dict[CoderFormat, dict[CoderType, str]] = {
    CoderFormat.H264: {
        CoderType.CPU:  'libx264',
        CoderType.GPU:  'h264_nvenc',
        CoderType.iGPU: 'h264_qsv'
    },
    CoderFormat.H265: {
        CoderType.CPU:  'libx265',
        CoderType.GPU:  'hevc_nvenc',
        CoderType.iGPU: 'hevc_qsv'
    }
}

class RecState(Enum):
    IDLE =  auto()
    START = auto()
    REC =   auto()
    STOP =  auto()

class SyncRecorder(Thread):
    def __init__(self, settings: CameraSettings) -> None:
        super().__init__()
        self.output_path: Path = Path(settings.video_path)
        self.temp_path: Path = Path(settings.temp_path)

        self.settings: CameraSettings = settings

        self.recorders: dict[int, dict[FrameType, FFmpegRecorder]] = {}
        self.fps: dict[int, float] = {}
        self.frames: dict[int, Queue[dict[FrameType, ndarray]]] = {}
        self.folder_path: Path = Path()

        for c in range(settings.num_cameras):
            self.recorders[c] = {}
            self.fps[c] = settings.fps
            self.frames[c] = Queue()
            for t in self.settings.recorder.video_frame_types:
                self.recorders[c][t] = FFmpegRecorder(EncoderString[settings.recorder.video_format][settings.recorder.video_encoder])

        self.start_time: float
        self.chunk_index = 0
        self.rec_name: str
        self.suffix: str = settings.recorder.video_format.value

        self.state: RecState = RecState.IDLE
        self.state_lock = Lock()
        self.settings_lock = Lock()
        self.group_id_lock = Lock()
        self.stop_event = Event()

        self.group_id: str = 'no_id'

        # Bind recorder settings callbacks
        self.settings.recorder.bind(RecorderSettings.start, self._on_start)
        self.settings.recorder.bind(RecorderSettings.stop, self._on_stop)
        self.settings.recorder.bind(RecorderSettings.group_id, self.set_group_id)

        # OSC bridge (moved from SyncRecorderGui)
        self.osc_client = SimpleUDPClient("10.0.0.148", 8600)
        self.osc_receive: Dispatcher = Dispatcher()
        self.osc_receive.set_default_handler(self._receive_osc)
        self.osc_server = ThreadingOSCUDPServer(('0.0.0.0', 8601), self.osc_receive)
        self.osc_server_thread: Thread = Thread(target=self.osc_server.serve_forever, daemon=True)
        self.osc_server_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        self._set_state(RecState.IDLE)

        while not self.stop_event.is_set():

            if self._get_state() == RecState.STOP:
                self._stop_recording()
                self._set_state(RecState.IDLE)
            if self._get_state() == RecState.REC:
                self._update_recording()
            if self._get_state() == RecState.START:
                self._start_recording()
                self._set_state(RecState.REC)

            time.sleep(0.01)

        self._stop_recording()

    def _start_recording(self) -> None:

        self.folder_path = self.output_path / make_folder_name(self.settings.num_cameras, self.settings.square, self.settings.color, self.settings.stereo, self.get_group_id())
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.settings.num_cameras):
            fps: float = self.get_fps(c)
            for t in self.settings.recorder.video_frame_types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                self.recorders[c][t].start(str(path), fps)

        self.start_time = time.time()
        self.recording = True

    def _stop_recording(self) -> None:
        for c in range(self.settings.num_cameras):
            for t in self.settings.recorder.video_frame_types:
                self.recorders[c][t].stop()

    def _update_recording(self) -> None:
        if time.time() - self.start_time > self.settings.recorder.video_chunk_length:
            self.chunk_index += 1

            for c in range(self.settings.num_cameras):
                fps: float = self.get_fps(c)
                for t in self.settings.recorder.video_frame_types:
                    path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                    self.recorders[c][t].split(str(path), fps)
            self.start_time += self.settings.recorder.video_chunk_length

        for c in range(self.settings.num_cameras):
            try:
                frames: dict[FrameType, ndarray] = self.frames[c].get(timeout=0.01)
            except Empty:
                continue
            for t in self.settings.recorder.video_frame_types:
                if t in frames:
                    self.recorders[c][t].add_frame(frames[t])

    def _get_state(self) -> RecState:
        with self.state_lock:
            return self.state

    def _set_state(self, state: RecState) -> None:
        with self.state_lock:
            if state == RecState.START and self.state != RecState.IDLE:
                return
            if  state != RecState.IDLE and self.state == RecState.STOP:
                return

            self.state = state

    # EXTERNAL METHODS
    def set_synced_frames(self, cam_id: int, frames: dict[FrameType, ndarray], fps: float) -> None:
        self.set_fps(cam_id, fps)
        if self._get_state() == RecState.REC:
            self.frames[cam_id].put(frames)

    def set_fps(self, cam_id: int, fps: float) -> None:
        with self.settings_lock:
            self.fps[cam_id] = fps

    def get_fps(self, cam_id) -> float:
        with self.settings_lock:
            return self.fps[cam_id]

    def record(self, value: bool) -> None:
        if value:
            self._set_state(RecState.START)
            self.settings.recorder.recording = True
        else:
            self._set_state(RecState.STOP)
            self.settings.recorder.recording = False

    def set_group_id(self, value: str) -> None:
        with self.group_id_lock:
            self.group_id = value

    def get_group_id(self) -> str:
        with self.group_id_lock:
            return self.group_id

    # SETTINGS CALLBACKS
    def _on_start(self, _=None) -> None:
        self.set_group_id('no_id')
        self.settings.recorder.group_id = 'no_id'
        self.record(True)

    def _on_stop(self, _=None) -> None:
        self.record(False)

    # OSC BRIDGE
    def _receive_osc(self, address, *args) -> None:
        print(address, args)
        if address == '/group/id':
            self.set_group_id(args[0])
            self.settings.recorder.group_id = args[0]
        if address == '/start/recording':
            self.record(True)
        if address == '/stop/recording':
            self.record(False)
        self.osc_client.send_message(address, args)
