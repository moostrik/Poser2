
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


import logging
from threading import Event

import psutil
p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)


from argparse import ArgumentParser, Namespace
from signal import signal, SIGINT

from modules.log_config import setup_logging
from modules.main import Main

import multiprocessing as mp

# Enable PyTorch optimizations for modern GPUs (RTX 30xx/40xx/50xx)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__': # For Windows compatibility with multiprocessing
    mp.freeze_support()
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecorded video with camera')
    parser.add_argument('-nc',      '--num_cameras',    type=int,   default=3,      help='set the number of cameras (default: 3)')
    parser.add_argument('-fps',     '--fps',            type=float, default=0.0,    help='set the frames per second for the camera and pose processing, 0 for settings default')
    parser.add_argument('-v',       '--verbose',        action='store_true',        help='enable verbose (DEBUG) console output')


    args: Namespace = parser.parse_args()

    log_file = setup_logging(verbose=args.verbose)
    logging.info("Logging to: %s", log_file)
    logging.info("Process PID: %s", os.getpid())

    app = Main(simulation=args.simulation, num_cameras=args.num_cameras, fps=args.fps)
    app.start()

    shutdown_event = Event()

    def signal_handler_exit(sig, frame) -> None:
        logging.info("Received interrupt signal, shutting down...")
        shutdown_event.set()
        if app.is_running:
            app.stop()

    signal(SIGINT, signal_handler_exit)

    while not app.is_finished and not shutdown_event.is_set():
        shutdown_event.wait(0.01)

    # Hard Exit for a problem that arises from GLFW not closing properly
    from os import _exit
    _exit(1)
