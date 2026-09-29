from argparse import ArgumentParser
from os import path
from signal import signal, SIGINT
from sys import exit
from time import sleep
from modules.log_config import setup_logging, install_thread_excepthook
from modules.Forgettable import Forgettable, CamType

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-dw',  '--width',          type=int,   default=1152,    help='image width')
parser.add_argument('-dh',  '--height',         type=int,   default=896,    help='image height')
parser.add_argument('-wc',  '--useWebCam',      action='store_true',        help='use webcam instead of default cam')
parser.add_argument('-vc',  '--useVimbaCam',    action='store_true',        help='use vimba camera instead of default cam')
parser.add_argument('-ic',  '--useImageCam',    action='store_true',        help='use image instead of default cam')
parser.add_argument('-pm',  '--portraitMode',   action='store_true',        help='use portrait mode (default)')
parser.add_argument('-lm',  '--landscapeMode',  action='store_true',        help='use landscape mode')
parser.add_argument('-v',   '--verbose',        action='store_true',        help='enable verbose (DEBUG) console output')
args = parser.parse_args()

setup_logging(verbose=args.verbose)
install_thread_excepthook()

currentPath: str = path.dirname(__file__)

modWidth :int = round(args.width / 4) * 4
modHeight : int = round(args.height / 4) * 4

camType = CamType.VIMBA
if args.useVimbaCam:
    camType = CamType.VIMBA
if args.useWebCam:
    camType = CamType.WEB
if args.useImageCam:
    camType = CamType.IMAGE

aiia = Forgettable(currentPath, modWidth, modHeight, not(bool(args.landscapeMode)), camType)
aiia.start()

def signal_handler_exit(sig, frame) -> None:
    if aiia.isRunning():
        aiia.stop()
    exit()

signal(SIGINT, signal_handler_exit)

while aiia.isRunning():
    sleep(0.05)
    continue