from argparse import ArgumentParser
from os import path, _exit
from signal import signal, SIGINT
from time import sleep
from player.Player import Player

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-m',   '--monitor',    type=int,   default=1,      help='monitor (0 is the primary, then left to right)')
parser.add_argument('-osc', '--osc_port',   type=int,   default=4115,   help='osc port')
args = parser.parse_args()

currentPath: str = path.dirname(__file__)

trail = 0.5         # the grin plays at 15 fps; a long trail would smear it
interval = 1.0 / 15.0
wait = 3.0
num_portraits = 24
player: Player = Player(args.osc_port, 'images/', args.monitor, trail, interval, wait, num_portraits)
player.start()

def signal_handler_exit(sig, frame) -> None:
    if player.isRunning():
        player.stop()

signal(SIGINT, signal_handler_exit)

while player.isRunning():
    sleep(0.05)

# GLFW does not shut down cleanly outside the main thread (as in poser2)
_exit(0)
