from argparse import ArgumentParser
from os import path, _exit
from signal import signal, SIGINT
from time import sleep
from player.Player import Player

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-m',   '--monitor',    type=int,   default=1,      help='monitor (0 is the primary, then left to right)')
parser.add_argument('-osc', '--osc_port',   type=int,   default=4115,   help='osc port')
parser.add_argument('-n',   '--clips',      type=int,   default=24,     help='number of clips playing at once (cells)')
parser.add_argument('--pool',               type=int,   default=24,     help='number of newest clips kept decoded in RAM (about 200 MB each)')
parser.add_argument('--root',               type=str,   default='recordings/', help='recordings folder (clips in its grins/ subfolder)')
parser.add_argument('--hwdecode',           type=int,   default=None,   help='decode on the GPU with this DXGI adapter index (default: software)')
args = parser.parse_args()

currentPath: str = path.dirname(__file__)

trail = 0.5         # the grin plays at 15 fps; a long trail would smear it
wait = 3.0
player: Player = Player(args.osc_port, args.root, args.monitor, trail, wait, args.clips, args.pool, args.hwdecode)
player.start()

def signal_handler_exit(sig, frame) -> None:
    if player.isRunning():
        player.stop()

signal(SIGINT, signal_handler_exit)

while player.isRunning():
    sleep(0.05)

# GLFW does not shut down cleanly outside the main thread (as in poser2)
_exit(0)
