# TODO
# Save videos to temporary folder until finished

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


import psutil
import os

p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)


from argparse import ArgumentParser, Namespace
from os import path
from signal import signal, SIGINT
from time import sleep

from modules.Main import Main
from modules.Settings import Settings, CamSettings, PoseSettings, GuiSettings
from modules.Settings import ModelType, FrameType, TrackerType

import multiprocessing as mp

if __name__ == '__main__': # For Windows compatibility with multiprocessing
    mp.freeze_support()
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-fps',     '--fps',            type=float, default=23.0,   help='frames per second')
    parser.add_argument('-pls',     '--players',        type=int,   default=8,      help='number of players')
    parser.add_argument('-cms',     '--cameras',        type=int,   default=4,      help='number of cameras')
    parser.add_argument('-ny',      '--noyolo',         action='store_true',        help='do not do yolo person detection')
    parser.add_argument('-np',      '--nopose',         action='store_true',        help='do not do pose detection')
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecored video with camera')
    parser.add_argument('-cm',      '--cammanual',      action='store_true',        help='camera manual settings')

    parser.add_argument('-hdt',     '--hdtrio',         action='store_true',        help='run in Harmonic Dissonance mode')
    parser.add_argument('-ws',      '--whitespace',     action='store_true',        help='run in Whitespace mode')
    parser.add_argument('-tm',      '--testminimal',    action='store_true',        help='test with minimum setup only')
    args: Namespace = parser.parse_args()


    app_name: str = "Poser"
    num_players: int =  args.players

    current_path: str = path.dirname(__file__)
    file_path: str =    path.join(current_path, 'files')
    model_path: str =   path.join(current_path, 'models')
    video_path: str =   path.join(current_path, 'recordings')
    temp_path: str =    path.join(current_path, 'temp')

    settings: Settings =  Settings()

    art_type_default = Settings.ArtType.HDT

    settings.art_type = art_type_default
    if art_type_default == Settings.ArtType.WS and args.hdtrio:
        settings.art_type = Settings.ArtType.HDT
    if art_type_default == Settings.ArtType.HDT and args.whitespace:
        settings.art_type = Settings.ArtType.WS

    camera_list: list[str] = ['14442C101136D1D200',
                              '14442C10F124D9D600',
                              '14442C10110AD3D200',
                              '14442C1031DDD2D200']
    if art_type_default == Settings.ArtType.HDT:
        camera_list = ['19443010D153874800',
                       '19443010D1E4974800',
                       '19443010D14C874800']
        camera_list = ['14442C101136D1D200', # STUDIO
                       '14442C10F124D9D600',
                       '14442C10110AD3D200']

    if args.cameras < len(camera_list):
        camera_list = camera_list[:args.cameras]


    cam_settings: CamSettings = CamSettings(
        ids=camera_list,
        model_path=model_path,
        video_path=video_path,
        temp_path=temp_path,

        fps=args.fps,
        sim_fps=args.fps,
        sim_enabled=args.simulation,
        yolo=not args.noyolo,
        manual=args.cammanual,

        perspective =   -0.22,
        flip_h =        False,
        flip_v =        True,
    )

    pose_settings = PoseSettings(
        num_warmups =   num_players,
        model_path =    model_path,

        active =        not args.nopose,
        model_type =    ModelType.NONE if args.nopose else ModelType.SMALL,
        stream_capacity = int(10 * args.fps),

        confidence_threshold = 0.5,
        crop_expansion = 0.1,
        verbose =       True
    )

    gui_settings = GuiSettings(
        title=          app_name,
        file_path=      file_path,
        on_top =        True,
        location_x =    1100,
        location_y =    -900,
        default_file =  "default"
    )

    settings.num_players =          num_players
    settings.path_root =            current_path

    settings.path_file =            file_path
    settings.path_model =           model_path
    settings.path_video =           video_path
    settings.path_temp =            temp_path


    settings.camera =               cam_settings
    settings.pose =                 pose_settings
    settings.gui =                  gui_settings


    settings.tracker_type =         TrackerType.PANORAMIC


    settings.corr_rate_hz =         args.fps
    settings.corr_num_workers =     10
    settings.corr_buffer_duration = int(3 * args.fps)
    settings.corr_stream_timeout =  2.0 # seconds
    settings.corr_max_nan_ratio =   0.15 # maximum ratio of NaN values in a window
    settings.corr_dtw_band =        10 # maximum distance between two points in a window
    settings.corr_similarity_exp =  4.0 # exponent for similarity calculation
    settings.corr_stream_capacity = int(10 * args.fps)


    settings.udp_port =             8888
    settings.udp_ips_sound =        '127.0.0.1'
    settings.udp_ips_light =        '127.0.0.1'

    settings.render_title =         'White Space'
    settings.render_x =             0
    settings.render_y =             100
    settings.render_width =         1920 * 2
    settings.render_height =        1080 * 2 - settings.render_y
    settings.render_fullscreen =    False
    settings.render_fps =           0
    settings.render_v_sync =        True
    settings.render_cams_a_row =    2
    settings.render_monitor =       0
    settings.render_R_num =         3
    settings.render_secondary_list =         [2]

    if settings.art_type == Settings.ArtType.HDT:
        settings.render_title =     'Harmonic Dissonance'
        settings.num_players =      3
        settings.tracker_type =     TrackerType.ONEPERCAM
        # settings.render_monitor =   1
        settings.render_secondary_list =     [1,2,3]
        # settings.render_x =         0
        # settings.render_y =         1500
        # settings.render_width =     2160
        # settings.render_height =    3000
        settings.render_fps =       60

    settings.check_values()
    settings.check_cameras()

    settings.save("files/settings.json")
    settings.load("files/settings.json")

    app = Main(settings)
    app.start()

    import threading
    # ...existing code...

    shutdown_event = threading.Event()


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
