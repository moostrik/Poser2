
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"



from threading import Event

import psutil
p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)


from argparse import ArgumentParser, Namespace
from os import path
from signal import signal, SIGINT

from modules.Main import Main
from modules.Settings import Settings, ModelType, ModelSize

import multiprocessing as mp

if __name__ == '__main__': # For Windows compatibility with multiprocessing
    mp.freeze_support()
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-fps',     '--fps',            type=float, default=23.0,   help='frames per second')
    parser.add_argument('-pls',     '--players',        type=int,   default=3,      help='number of players')
    parser.add_argument('-cms',     '--cameras',        type=int,   default=3,      help='number of cameras')
    parser.add_argument('-ny',      '--noyolo',         action='store_true',        help='do not do yolo person detection')
    parser.add_argument('-np',      '--nopose',         action='store_true',        help='do not do pose detection')
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecored video with camera')
    parser.add_argument('-cm',      '--cammanual',      action='store_true',        help='camera manual settings')

    parser.add_argument('-s',      '--settings',        type=str, default='default',help='settings file')

    args: Namespace = parser.parse_args()

    settings_path: str = f"files/settings/{args.settings}.json"

    print(f"Loading settings from: {settings_path}")
    settings: Settings = Settings.load(settings_path)

    if args.cameras < len(settings.camera.ids):
        settings.camera.ids = settings.camera.ids[:args.cameras]

    settings.camera.fps = args.fps
    settings.camera.sim_enabled = args.simulation
    settings.camera.yolo = not args.noyolo
    settings.camera.manual = args.cammanual

    settings.pose.max_poses = args.players
    if args.nopose:
        settings.pose.model_type = ModelType.NONE

    settings.pd_stream.max_poses = args.players
    settings.pd_stream.corr_rate = args.fps
    settings.pd_stream.stream_capacity = int(10 * args.fps)
    settings.pd_stream.corr_buffer_duration = int(3 * args.fps)

    settings.sound_osc.num_players = args.players

    settings.render.num_players = args.players
    settings.render.num_cams = args.cameras

    # Settings.make_paths_absolute(settings, path.dirname(__file__))
    # print(settings)

    app = Main(settings)
    app.start()

    shutdown_event = Event()

    def signal_handler_exit(sig, frame) -> None:
        print("Received interrupt signal, shutting down...")
        shutdown_event.set()
        if app.is_running:
            app.stop()

    signal(SIGINT, signal_handler_exit)

    while not app.is_finished and not shutdown_event.is_set():
        shutdown_event.wait(0.01)


    # Hard Exit for a problem that arises from GLFW not closing properly
    from os import _exit
    _exit(1)
