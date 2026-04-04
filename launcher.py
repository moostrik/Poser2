import logging
import os
from argparse import ArgumentParser, Namespace
from signal import signal, SIGINT
from threading import Event

import psutil
import torch

from modules.log_config import setup_logging
from modules.main import Main


if __name__ == '__main__':
    process_id = os.getpid()
    psutil.Process(process_id).nice(psutil.HIGH_PRIORITY_CLASS)  # Prioritize over normal apps, below system processes
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 tensor cores for faster matmul on Ampere+ GPUs

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecorded video with camera')
    parser.add_argument('-nc',      '--num_cameras',    type=int,   default=3,      help='set the number of cameras (default: 3)')
    parser.add_argument('-fps',     '--fps',            type=float, default=0.0,    help='set the frames per second for the camera and pose processing, 0 for settings default')
    parser.add_argument('-v',       '--verbose',        action='store_true',        help='enable verbose (DEBUG) console output')
    args: Namespace = parser.parse_args()

    log_file = setup_logging(verbose=args.verbose)
    logging.info("Logging to: %s", log_file)
    logging.info("Process PID: %s", process_id)

    app = Main(simulation=args.simulation)
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

    # Hard exit required: GLFW does not release its main-thread context cleanly
    from os import _exit
    _exit(0)
