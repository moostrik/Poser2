# TODO
# Clean up old sdw repo
# Replace PySimpleGui with Gradio

# OPTIONAL
# clean up sd module
# less random audio

from argparse import ArgumentParser
from os import path
from signal import signal, SIGINT
from sys import exit
from time import sleep
from modules.Forgettable import Forgettable, CamType

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-dw',  '--width',          type=int,   default=1152,    help='image width')
parser.add_argument('-dh',  '--height',         type=int,   default=896,    help='image height')
parser.add_argument('-gt',  '--guiTop',         action='store_true',        help='gui window always on top')
parser.add_argument('-wc',  '--useWebCam',      action='store_true',        help='use webcam instead of default cam')
parser.add_argument('-vc',  '--useVimbaCam',    action='store_true',        help='use vimba camera instead of default cam')
parser.add_argument('-ic',  '--useImageCam',    action='store_true',        help='use image instead of default cam')
parser.add_argument('-pm',  '--portraitMode',   action='store_true',        help='use portrait mode (default)')
parser.add_argument('-lm',  '--landscapeMode',  action='store_true',        help='use landscape mode')
args = parser.parse_args()

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

aiia = Forgettable(currentPath, modWidth, modHeight, not(bool(args.landscapeMode)), args.guiTop, camType)
aiia.start()

def signal_handler_exit(sig, frame) -> None:
    if aiia.isRunning():
        aiia.stop()
    exit()

signal(SIGINT, signal_handler_exit)

while aiia.isRunning():
    sleep(0.05)
    continue