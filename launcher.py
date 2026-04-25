import logging
import os
from argparse import ArgumentParser, Namespace
from signal import signal, SIGINT
from threading import Event

import psutil

from modules.log_config import setup_logging, install_thread_excepthook
from apps import APP_REGISTRY


if __name__ == '__main__':
    process_id = os.getpid()
    psutil.Process(process_id).nice(psutil.HIGH_PRIORITY_CLASS)  # Prioritize over normal apps, below system processes
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 tensor cores for faster matmul on Ampere+ GPUs
    except ImportError:
        pass

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-app',     '--app',            type=str,   default='white_space', choices=list(APP_REGISTRY.keys()), help='app to launch (default: hd_trio)')
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecorded video with camera')
    parser.add_argument('-v',       '--verbose',        action='store_true',        help='enable verbose (DEBUG) console output')
    args: Namespace = parser.parse_args()

    log_file = setup_logging(verbose=args.verbose)
    install_thread_excepthook()
    logging.info("Logging to: %s", log_file)
    logging.info("Process PID: %s", process_id)
    logging.info("App: %s", args.app)

    AppClass = APP_REGISTRY[args.app]
    app = AppClass(simulation=args.simulation)
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
