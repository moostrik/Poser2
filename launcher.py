import psutil
import os

p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)


from argparse import ArgumentParser, Namespace
from os import path
from signal import signal, SIGINT
from time import sleep

from modules.Main import Main
from modules.Settings import Settings, ModelType, FrameType, TrackerType

import multiprocessing as mp

if __name__ == '__main__': # For Windows compatibility with multiprocessing
    mp.freeze_support()
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-fps',     '--fps',            type=float, default=23.0,   help='frames per second')
    parser.add_argument('-pl',      '--players',        type=int,   default=8,      help='number of players')
    parser.add_argument('-c',       '--color',          action='store_true',        help='use color input instead of left mono')
    parser.add_argument('-sq',      '--square',         action='store_true',        help='use centre square of the camera')
    parser.add_argument('-ss',      '--showstereo',     action='store_true',        help='queue stereo frames')
    parser.add_argument('-st',      '--stereo',         action='store_true',        help='use stereo depth')
    parser.add_argument('-ny',      '--noyolo',         action='store_true',        help='do not do yolo person detection')
    parser.add_argument('-np',      '--nopose',         action='store_true',        help='do not do pose detection')
    parser.add_argument('-cl',      '--chunklength',    type=float, default=6.0,    help='duration of video chunks in seconds')
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecored video with camera')
    parser.add_argument('-pt',      '--passthrough',    action='store_true',        help='use prerecored video without camera')
    parser.add_argument('-nc',      '--numcameras',     type=int,   default=4,      help='number of cameras')
    parser.add_argument('-as',      '--autoset',        action='store_true',        help='camera auto settings')
    parser.add_argument('-ad',      '--debug',          action='store_true',        help='run analysis in debug mode')
    parser.add_argument('-hdt',     '--harmonictrio',   action='store_true',        help='run in Harmonic Dissonance mode')
    parser.add_argument('-ws',      '--whitespace',     action='store_true',        help='run in Whitespace mode')
    parser.add_argument('-tm',      '--testminimal',    action='store_true',        help='test with minimum setup only')
    args: Namespace = parser.parse_args()

    currentPath: str = path.dirname(__file__)

    camera_list: list[str] = ['14442C101136D1D200',
                              '14442C10F124D9D600',
                              '14442C10110AD3D200',
                              '14442C1031DDD2D200']

    if args.numcameras < len(camera_list):
        camera_list = camera_list[:args.numcameras]

    udp_list_sound: list[str] = ['127.0.0.1','10.0.0.81']
    udp_list_light: list[str] = []

    # camera_list: list[str] = ['14442C10F124D9D600']

    settings: Settings =            Settings()

    settings.art_type =             Settings.ArtType.WS

    settings.max_players =          args.players

    settings.gui_location_x =       1920
    settings.gui_location_y =       -1500
    settings.gui_on_top =           True
    settings.gui_default_file =     'default'

    settings.path_root =            currentPath
    settings.path_file =            path.join(currentPath, 'files')
    settings.path_model =           path.join(currentPath, 'models')
    settings.path_video =           path.join(currentPath, 'recordings')
    settings.path_temp =            path.join(currentPath, 'temp')

    settings.camera_list =          camera_list
    settings.camera_num =           len(camera_list)
    settings.camera_fps =           args.fps
    settings.camera_square =        args.square
    settings.camera_color =         args.color
    settings.camera_stereo =        args.stereo
    settings.camera_yolo =      not args.noyolo
    settings.camera_show_stereo =   args.showstereo
    settings.camera_simulation =    args.simulation or args.passthrough
    settings.camera_passthrough =   args.passthrough
    settings.camera_manual =    not args.autoset
    settings.camera_flip_h =        False
    settings.camera_flip_v =        False
    settings.camera_perspective =   0.1

    settings.video_chunk_length =   args.chunklength
    settings.video_encoder =        Settings.CoderType.iGPU
    settings.video_decoder =        Settings.CoderType.iGPU
    settings.video_format =         Settings.CoderFormat.H264
    settings.video_frame_types =    [FrameType.VIDEO, FrameType.LEFT_, FrameType.RIGHT] if settings.camera_stereo else [FrameType.VIDEO]

    settings.tracker_type =         TrackerType.PANORAMIC
    settings.tracker_min_age =      5 # in frames
    settings.tracker_min_height =   0.25 # * height of the camera
    settings.tracker_timeout =      2.0 # in seconds

    settings.pose_active =      not args.nopose
    settings.pose_model_type =      ModelType.NONE if args.nopose else ModelType.SMALL
    settings.pose_conf_threshold =  0.3
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
    settings.render_y =             0
    settings.render_width =         1920 * 2
    settings.render_height =        1080 * 2 - settings.render_y
    settings.render_fullscreen =    False
    settings.render_fps =           0
    settings.render_v_sync =        True
    settings.render_cams_a_row =    2
    settings.render_monitor =       2
    settings.render_R_num =         3

    #settings.art_type =             Settings.ArtType.HDT
    if args.whitespace:
        settings.art_type = Settings.ArtType.WS

    if settings.art_type == Settings.ArtType.HDT:
        settings.render_title =     'Harmonic Dissonance'
        settings.max_players =      3
        settings.camera_num =       settings.max_players
        settings.camera_list =      camera_list[:settings.camera_num]
        settings.tracker_type =     TrackerType.ONEPERCAM

    if args.testminimal:
        settings.render_title =     'Minimal Test'
        settings.max_players =      2
        settings.camera_num =       settings.max_players
        settings.camera_list =      camera_list[:settings.camera_num]
        settings.tracker_type =     TrackerType.ONEPERCAM

    settings.check_values()
    # settings.check_cameras()

    app = Main(settings)
    app.start()

    def signal_handler_exit(sig, frame) -> None:
        print("Received interrupt signal, shutting down...")
        if app.is_running:
            app.stop()

    signal(SIGINT, signal_handler_exit)

    while not app.is_finished:
        sleep(0.1)
        continue


    # Hard Exit for a problem that arises from GLFW not closing properly
    # from os import _exit
    # _exit(1)
