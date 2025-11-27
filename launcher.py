
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
from modules.Settings import Settings, CamSettings, PoseSettings, PDStreamSettings, GuiSettings, SoundOSCConfig, RenderSettings
from modules.Settings import ModelType, TrackerType, DataType

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

    args: Namespace = parser.parse_args()


    app_name: str = "Poser"
    num_players: int =  args.players

    current_path: str = path.dirname(__file__)
    file_path: str =    path.join(current_path, 'files')
    model_path: str =   path.join(current_path, 'models')
    video_path: str =   path.join(current_path, 'recordings')
    temp_path: str =    path.join(current_path, 'temp')

    # WHITE SPACE CAMS (STUDIO CAMS)
    camera_list: list[str] = ['14442C101136D1D200',
                              '14442C10F124D9D600',
                              '14442C10110AD3D200',
                              '14442C1031DDD2D200']
    # UMU CAMS
    # camera_list = ['19443010D153874800',
    #                '19443010D1E4974800',
    #                '19443010D14C874800']

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
        max_poses =     num_players,
        model_path =    model_path,

        active =        not args.nopose,
        model_type =    ModelType.NONE if args.nopose else ModelType.SMALL,

        confidence_threshold = 0.5,
        crop_expansion = 0.1,
        verbose =       True
    )
    pd_stream_settings = PDStreamSettings(
        max_poses =             num_players,
        stream_capacity =       int(10 * args.fps),
        corr_rate =             args.fps,
        corr_buffer_duration =  int(3 * args.fps),
    )

    gui_settings = GuiSettings(
        title=          app_name,
        file_path=      file_path,
        on_top =        True,
        location_x =    1100,
        location_y =    -900,
        default_file =  "default"
    )

    sound_osc_config = SoundOSCConfig(
        ip_addresses =  '127.0.0.1',
        port =          8888,
        num_players =   num_players,
        data_type =     DataType.pose_I
    )

    render_settings = RenderSettings(
        title =         app_name,
        num_cams=       cam_settings.num,
        num_players=    num_players,
        monitor=        0,
        width=          1920,
        height=         1000,
        x=              0,
        y=              80,
        fullscreen=     False,
        fps=            60,
        v_sync=         True,

        secondary_list= [1,2,3]
    )

    settings: Settings =  Settings()
    settings.num_players =          num_players
    settings.tracker_type =         TrackerType.ONEPERCAM
    settings.camera =               cam_settings
    settings.pose =                 pose_settings
    settings.gui =                  gui_settings
    settings.sound_osc =            sound_osc_config
    settings.render =               render_settings

    # settings.save("files/settings.json")
    # settings.load("files/settings.json")

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
