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
from modules.Settings import Settings, PoseModelType, FrameType, TrackerType

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
    parser.add_argument('-simpt',   '--passthrough',    action='store_true',        help='use prerecored video without camera')
    parser.add_argument('-cm',      '--cammanual',      action='store_true',        help='camera manual settings')

    parser.add_argument('-hdt',     '--hdtrio',         action='store_true',        help='run in Harmonic Dissonance mode')
    parser.add_argument('-ws',      '--whitespace',     action='store_true',        help='run in Whitespace mode')
    parser.add_argument('-tm',      '--testminimal',    action='store_true',        help='test with minimum setup only')
    args: Namespace = parser.parse_args()


    currentPath: str = path.dirname(__file__)

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
        camera_list = ['19443010D1E4974800',
                       '19443010D153874800',

                       '19443010D14C874800']

    if args.cameras < len(camera_list):
        camera_list = camera_list[:args.cameras]

    udp_list_sound: list[str] = ['127.0.0.1','10.0.0.81']
    udp_list_light: list[str] = []

    settings.num_players =          args.players


    settings.path_root =            currentPath
    settings.path_file =            path.join(currentPath, 'files')
    settings.path_model =           path.join(currentPath, 'models')
    settings.path_video =           path.join(currentPath, 'recordings')
    settings.path_temp =            path.join(currentPath, 'temp')

    settings.camera_list =          camera_list
    settings.camera_num =           len(camera_list)
    settings.camera_fps =           args.fps
    settings.camera_square =        False
    settings.camera_color =         True
    settings.camera_stereo =        False
    settings.camera_yolo =      not args.noyolo
    settings.camera_show_stereo =   False
    settings.camera_simulation =    args.simulation or args.passthrough
    settings.camera_passthrough =   args.passthrough
    settings.camera_manual =        args.cammanual
    settings.camera_flip_h =        False
    settings.camera_flip_v =        False
    settings.camera_perspective =   0.1

    settings.video_chunk_length =   10 # in seconds
    settings.video_encoder =        Settings.CoderType.iGPU
    settings.video_decoder =        Settings.CoderType.iGPU
    settings.video_format =         Settings.CoderFormat.H264
    settings.video_frame_types =    [FrameType.VIDEO, FrameType.LEFT_, FrameType.RIGHT] if settings.camera_stereo else [FrameType.VIDEO]

    settings.tracker_type =         TrackerType.PANORAMIC
    settings.tracker_min_age =      5 # in frames
    settings.tracker_min_height =   0.25 # * height of the camera
    settings.tracker_timeout =      2.0 # in seconds

    settings.pose_active =      not args.nopose
    settings.pose_model_type =      PoseModelType.NONE if args.nopose else PoseModelType.SMALL
    settings.pose_model_warmups =    settings.num_players
    settings.pose_conf_threshold =  0.5
    settings.pose_crop_expansion =  0.1 # * height of the camera
    settings.pose_stream_capacity = int(10 * args.fps)
    settings.pose_verbose =         False

    settings.corr_rate_hz =         args.fps
    settings.corr_num_workers =     10
    settings.corr_buffer_duration = int(3 * args.fps)
    settings.corr_stream_timeout =  settings.tracker_timeout # seconds
    settings.corr_max_nan_ratio =   0.15 # maximum ratio of NaN values in a window
    settings.corr_dtw_band =        10 # maximum distance between two points in a window
    settings.corr_similarity_exp =  2.0 # exponent for similarity calculation
    settings.corr_stream_capacity = int(30 * args.fps)

    settings.light_resolution =     3600
    settings.light_rate =           60

    settings.udp_port =             8888
    settings.udp_ips_sound =        udp_list_sound
    settings.udp_ips_light =        udp_list_light

    settings.render_title =         'White Space'
    settings.render_x =             0
    settings.render_y =             100
    settings.render_width =         1920 * 2
    settings.render_height =        1080 * 2 - settings.render_y
    settings.render_fullscreen =    False
    settings.render_fps =           0
    settings.render_v_sync =        True
    settings.render_cams_a_row =    2
    settings.render_monitor =       1
    settings.render_R_num =         3
    settings.render_secondary_list =         [2]

    settings.gui_location_x =       2400
    settings.gui_location_y =       -900
    settings.gui_on_top =           True
    settings.gui_default_file =     'default'

    if settings.art_type == Settings.ArtType.HDT:
        settings.camera_fps =       24
        settings.render_title =     'Harmonic Dissonance'
        settings.num_players =      3
        settings.camera_num =       settings.num_players
        settings.camera_list =      camera_list[:settings.camera_num]
        settings.camera_flip_h =        True
        settings.camera_flip_v =        False
        settings.camera_perspective =   0.28
        settings.tracker_type =     TrackerType.ONEPERCAM
        settings.render_monitor =   1
        settings.render_secondary_list =     [2,1,3]
        settings.render_x =         0
        settings.render_y =         100
        settings.render_fps =       60
        settings.gui_location_x =   2200
        settings.gui_location_y =   -700
        settings.gui_on_top =       True

        settings.pose_model_type =  PoseModelType.NONE if args.nopose else PoseModelType.LARGE
        settings.pose_verbose =     True

        settings.camera_color =     True
        settings.camera_square =    True

    if args.testminimal:
        settings.render_title =     'Minimal Test'
        settings.num_players =      2
        settings.camera_num =       settings.num_players
        settings.camera_list =      camera_list[:settings.camera_num]
        settings.tracker_type =     TrackerType.ONEPERCAM
        settings.pose_verbose =     True

    settings.check_values()
    settings.check_cameras()

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
