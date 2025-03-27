# TODO
# Save videos to temporary folder until finished

from argparse import ArgumentParser, Namespace
from os import path
from signal import signal, SIGINT
from sys import exit
from time import sleep
from modules.DepthPose import DepthPose

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-fps',     '--fps',        type=int,   default=30,     help='frames per second')
parser.add_argument('-pl',      '--players',    type=int,   default=6,      help='num players')
parser.add_argument('-mono',    '--mono',       action='store_true',        help='use left mono input instead of color')
parser.add_argument('-high',    '--highres',    action='store_true',        help='high resolution mono (720p instead of 400p)')
parser.add_argument('-ss',      '--showstereo', action='store_true',        help='queue stereo frames')
parser.add_argument('-ll',      '--lightning',  action='store_true',        help='use low latency movenet model')
parser.add_argument('-ns',      '--nostereo',   action='store_true',        help='do not use stereo depth')
parser.add_argument('-ny',      '--noyolo',     action='store_true',        help='do not do yolo person detection')
parser.add_argument('-np',      '--nopose',     action='store_true',        help='do not do pose detection')
parser.add_argument('-sim',     '--simulation', action='store_true',        help='use simulated depth camera')
args: Namespace = parser.parse_args()

currentPath: str = path.dirname(__file__)

camera_list: list[str] = ['14442C10F124D9D600',
                          '14442C10F124D9D601',
                          '14442C10F124D9D602',
                          '14442C10F124D9D603']

camera_list: list[str] = ['14442C10F124D9D600']

app: DepthPose = DepthPose(currentPath, camera_list, args.fps, args.players,
                           not args.mono, not args.nostereo, not args.noyolo, not args.highres, args.showstereo,
                           args.lightning, args.nopose, args.simulation)
app.start()

def signal_handler_exit(sig, frame) -> None:
    if app.isRunning():
        app.stop()
    exit()

signal(SIGINT, signal_handler_exit)

while app.isRunning():
    sleep(0.05)
    continue