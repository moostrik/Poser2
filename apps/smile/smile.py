from argparse import ArgumentParser
from os import _exit
from signal import signal, SIGINT
from time import sleep
from modules.log_config import setup_logging, install_excepthooks
from modules.main import Main, CamType

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-dw',  '--width',          type=int,   default=1280,   help='image width (4:5 like the generated portraits; 1280 x 1024 is the display in portrait mode)')
parser.add_argument('-dh',  '--height',         type=int,   default=1024,   help='image height')
parser.add_argument('-wc',  '--useWebCam',      action='store_true',        help='use webcam instead of default cam')
parser.add_argument('-vc',  '--useVimbaCam',    action='store_true',        help='use vimba camera instead of default cam')
parser.add_argument('-pm',  '--portraitMode',   action='store_true',        help='use portrait mode (default)')
parser.add_argument('-lm',  '--landscapeMode',  action='store_true',        help='use landscape mode')
parser.add_argument('-fs',  '--fullscreen',     action='store_true',        help='open borderless fullscreen (the installation) instead of the window mode of the preset')
parser.add_argument('-v',   '--verbose',        action='store_true',        help='enable verbose (DEBUG) console output')
args = parser.parse_args()

setup_logging(verbose=args.verbose)
install_excepthooks()

modWidth :int = round(args.width / 4) * 4
modHeight : int = round(args.height / 4) * 4

camType = CamType.VIMBA
if args.useVimbaCam:
    camType = CamType.VIMBA
if args.useWebCam:
    camType = CamType.WEB

app = Main(modWidth, modHeight, not(bool(args.landscapeMode)), camType, args.fullscreen)
app.start()

def signal_handler_exit(sig, frame) -> None:
    if app.isRunning():
        app.stop()

signal(SIGINT, signal_handler_exit)

while app.isRunning():
    sleep(0.05)

# GLFW does not shut down cleanly outside the main thread (as in poser2)
_exit(0)
