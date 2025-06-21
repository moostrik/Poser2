# TODO
# Save videos to temporary folder until finished

from argparse import ArgumentParser, Namespace
from os import path
from signal import signal, SIGINT
from sys import exit
from time import sleep

from modules.Main import Main
from modules.Settings import Settings, ModelType, FrameType

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-fps',     '--fps',        type=float, default=23.0,   help='frames per second')
parser.add_argument('-pl',      '--players',    type=int,   default=8,      help='number of players')
parser.add_argument('-c',       '--color',      action='store_true',        help='use color input instead of left mono')
parser.add_argument('-sq',      '--square',     action='store_true',        help='use centre square of the camera')
parser.add_argument('-ss',      '--showstereo', action='store_true',        help='queue stereo frames')
parser.add_argument('-ll',      '--lightning',  action='store_true',        help='use low latency movenet model')
parser.add_argument('-st',      '--stereo',     action='store_true',        help='use stereo depth')
parser.add_argument('-ny',      '--noyolo',     action='store_true',        help='do not do yolo person detection')
parser.add_argument('-np',      '--nopose',     action='store_true',        help='do not do pose detection')
parser.add_argument('-cl',      '--chunklength',type=float, default=6.0,    help='duration of video chunks in seconds')
parser.add_argument('-sim',     '--simulation', action='store_true',        help='use prerecored video with camera')
parser.add_argument('-pt',      '--passthrough',action='store_true',        help='use prerecored video without camera')
parser.add_argument('-nc',      '--numcameras', type=int,   default=4,      help='number of cameras')
parser.add_argument('-as',      '--autoset',    action='store_true',        help='camera auto settings')
args: Namespace = parser.parse_args()

currentPath: str = path.dirname(__file__)

camera_list: list[str] = ['14442C101136D1D200',
                          '14442C10F124D9D600',
                          '14442C10110AD3D200',
                          '14442C1031DDD2D200']

if args.numcameras < len(camera_list):
    camera_list = camera_list[:args.numcameras]

# camera_list: list[str] = ['14442C10F124D9D600']

settings: Settings =            Settings()

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

settings.pose_num =             args.players
settings.pose_active =      not args.nopose
settings.pose_model_type =      ModelType.NONE if args.nopose else ModelType.LIGHTNING if args.lightning else ModelType.THUNDER

settings.video_chunk_length =   args.chunklength
settings.video_encoder =        Settings.CoderType.iGPU
settings.video_decoder =        Settings.CoderType.iGPU
settings.video_format =         Settings.CoderFormat.H264
settings.video_frame_types =    [FrameType.VIDEO, FrameType.LEFT_, FrameType.RIGHT] if settings.camera_stereo else [FrameType.VIDEO]

settings.render_title =         'White Space'
settings.render_width =         1920
settings.render_height =        1080
settings.render_x =             0
settings.render_y =             0
settings.render_fullscreen=     False
settings.render_v_sync =        True
settings.render_cams_a_row=     2

settings.light_resolution =     3840
settings.light_rate =           30

settings.check_values()
# settings.check_cameras()

app: Main = Main(settings)
app.start()

def signal_handler_exit(sig, frame) -> None:
    if app.isRunning():
        app.stop()
    exit()

signal(SIGINT, signal_handler_exit)

while app.isRunning():
    sleep(0.05)
    continue