from argparse import ArgumentParser
from os import _exit
from signal import signal, SIGINT
from time import sleep
from modules.credit import NAME, URL
from modules.log_config import setup_logging, install_excepthooks
from modules.player.Player import Player

parser: ArgumentParser = ArgumentParser()
parser.add_argument('-m',   '--monitor',    type=int,   default=1,      help='monitor (0 is the primary, then left to right)')
parser.add_argument('-osc', '--osc_port',   type=int,   default=4115,   help='osc port')
parser.add_argument('-l',   '--listen',     type=int,   default=0,      help="port to listen on for /light from SMILE's laughter test, 0 = none")
parser.add_argument('-n',   '--clips',      type=int,   default=24,     help='number of clips playing at once (cells)')
parser.add_argument('--pool',               type=int,   default=30,     help='number of clips kept decoded in RAM (about 200 MB each, 5.9 GB for 30)')
parser.add_argument('--recent',             type=int,   default=4,      help='the last recordings, always on the wall; 0 = none')
parser.add_argument('--recycle_every',      type=float, default=20.0,   help='a clip from disk joins when no clip has joined for this many seconds; 0 = never')
parser.add_argument('--fps',                type=float, default=20.0,   help='playback speed of the clips (frames per second); 0 plays each clip at its own rate')
parser.add_argument('--spread',             type=float, default=0.1,    help='random speed difference per clip, as a fraction of --fps (0.1 = 10 percent slower to faster), 0 = none')
parser.add_argument('--trail',              type=float, default=0.0,    help='blend of each frame with the previous output, 0 = none (0.5 was the old look)')
parser.add_argument('--no_credit',          action='store_true',        help="start without the cell with the artist's name and QR code (SMILE's credit_qr_players switch sets it while SMILE runs)")
parser.add_argument('--credit_name',        type=str,   default=NAME,   help='name on the credit cell')
parser.add_argument('--credit_url',         type=str,   default=URL,    help='page the credit cell\'s QR code opens')
parser.add_argument('--root',              type=str,   default='output/', help='output folder (clips in its grins/ subfolder)')
parser.add_argument('--hwdecode',           type=int,   default=None,   help='decode on the GPU with this DXGI adapter index (default: software)')
parser.add_argument('--window',             action='store_true',        help='run in a window instead of borderless fullscreen (for testing)')
parser.add_argument('-v',   '--verbose',    action='store_true',        help='enable verbose (DEBUG) console output')
args = parser.parse_args()

setup_logging(verbose=args.verbose, name=f'player{args.monitor}')
install_excepthooks()

player: Player = Player(args.osc_port, args.root, args.monitor, args.trail, args.clips, args.pool, args.hwdecode, args.window, args.fps, args.spread, args.recent, args.recycle_every, args.listen,
                        not args.no_credit, args.credit_name, args.credit_url)
player.start()

def signal_handler_exit(sig, frame) -> None:
    if player.isRunning():
        player.stop()

signal(SIGINT, signal_handler_exit)

while player.isRunning():
    sleep(0.05)

# GLFW does not shut down cleanly outside the main thread (as in poser2)
_exit(0)
