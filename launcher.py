import json
import logging
import os
os.environ.setdefault('DEPTHAI_LEVEL', 'error')  # suppress noisy unbooted-device warnings
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from signal import signal, SIGINT
from threading import Event, Thread

import psutil

from modules.log_config import setup_logging, install_thread_excepthook
from apps import APP_REGISTRY


def _read_launcher_file() -> str:
    path = Path(__file__).parent / ".app_select.json"
    try:
        data = json.loads(path.read_text())
        name = data.get("app", "")
        if name in APP_REGISTRY:
            return name
        logging.warning(".app_select.json: unknown app %r, falling back to first in registry", name)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        logging.warning(".app_select.json: invalid JSON (%s), falling back to first in registry", e)
    return next(iter(APP_REGISTRY))


if __name__ == '__main__':
    process_id = os.getpid()
    psutil.Process(process_id).nice(psutil.HIGH_PRIORITY_CLASS)  # Prioritize over normal apps, below system processes
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 tensor cores for faster matmul on Ampere+ GPUs
    except ImportError:
        pass

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-app',     '--app',            type=str,   default=None, choices=list(APP_REGISTRY.keys()), help='app to launch (default: from .app_select.json or first in registry)')
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecorded video with camera')
    parser.add_argument('-v',       '--verbose',        action='store_true',        help='enable verbose (DEBUG) console output')
    args: Namespace = parser.parse_args()
    log_file = setup_logging(verbose=args.verbose)
    install_thread_excepthook()
    if args.app is None:
        args.app = _read_launcher_file()

    logging.info("Starting  >>> %s <<<  pid: %s  simulation: %s", args.app, process_id, args.simulation)
    logging.info("Logging to: %s", log_file)

    AppClass = APP_REGISTRY[args.app]
    app = AppClass(simulation=args.simulation)
    app.start()

    shutdown_event = Event()
    start_time = time.monotonic()

    def _heartbeat() -> None:
        while not shutdown_event.wait(60):
            elapsed = int(time.monotonic() - start_time)
            logging.debug("Heartbeat — running for %d min %d s", elapsed // 60, elapsed % 60)

    Thread(target=_heartbeat, daemon=True, name="heartbeat").start()

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
