# TODO
#

from argparse import ArgumentParser
from os import path
from signal import signal, SIGINT
from sys import exit
from time import sleep
from modules.DepthPose import DepthPose

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-cw',  '--width',          type=int,   default=1280,   help='camera width')
parser.add_argument('-ch',  '--height',         type=int,   default=720,   help='camera height')
parser.add_argument('-p',   '--portrait',       type=bool,  default=False,  help='portrait mode')
args = parser.parse_args()

currentPath: str = path.dirname(__file__)

modWidth :int = round(args.width / 4) * 4
modHeight : int = round(args.height / 4) * 4

app: DepthPose = DepthPose(currentPath, modWidth, modHeight, args.portrait)
app.start()

def signal_handler_exit(sig, frame) -> None:
    if app.isRunning():
        app.stop()
    exit()

signal(SIGINT, signal_handler_exit)

while app.isRunning():
    sleep(0.05)
    continue