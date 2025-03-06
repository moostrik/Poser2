# TODO
#

from argparse import ArgumentParser
from os import path
from signal import signal, SIGINT
from sys import exit
from time import sleep
from modules.DepthPose import DepthPose

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-fps',     '--fps',        type=int,   default=60,     help='frames per second')
parser.add_argument('-mono',    '--mono',       action='store_true',        help='use left mono input instead of color')
parser.add_argument('-low',     '--lowres',     action='store_true',        help='low resolution camera (400p instead of 720p)')
parser.add_argument('-left',    '--queueleft',  action='store_true',        help='queue left monochrome camera frames')
parser.add_argument('-nopose',  '--nopose',     action='store_true',        help='do not do pose detection')
args = parser.parse_args()

currentPath: str = path.dirname(__file__)

app: DepthPose = DepthPose(currentPath, args.fps, args.mono, args.lowres, args.queueleft, args.nopose)
app.start()

def signal_handler_exit(sig, frame) -> None:
    if app.isRunning():
        app.stop()
    exit()

signal(SIGINT, signal_handler_exit)

while app.isRunning():
    sleep(0.05)
    continue