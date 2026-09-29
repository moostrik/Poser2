from argparse import ArgumentParser
from os import path
from signal import signal, SIGINT
from sys import exit
from time import sleep
from player.Player import Player

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-pX',  '--posX',       type=int,   default=2000,   help='window x position')
parser.add_argument('-pY',  '--posY',       type=int,   default=0,      help='window y position')
parser.add_argument('-osc', '--osc_port',   type=int,   default=4115,   help='osc port')
args = parser.parse_args()

currentPath: str = path.dirname(__file__)

trail = 0.95
interval = 0.5
wait = 3.0
num_portraits = 24
player: Player = Player(args.osc_port, 'images/', args.posX, args.posY, trail, interval, wait, num_portraits)
player.start()

def signal_handler_exit(sig, frame) -> None:
    if player.isRunning():
        player.stop()
    exit()

signal(SIGINT, signal_handler_exit)

while player.isRunning():
    sleep(0.05)
    continue
