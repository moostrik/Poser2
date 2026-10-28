import json
import logging
import os
os.environ.setdefault('DEPTHAI_LEVEL', 'error')  # suppress noisy unbooted-device warnings
import OpenGL
OpenGL.ERROR_CHECKING = False  # must be set before any other OpenGL import; per-call glGetError otherwise dominates the render thread
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from signal import signal, SIGINT
from threading import Event, Thread

import psutil

from modules.log_config import setup_logging, install_excepthooks, crash_summary
from modules.utils import timing
from apps import APP_NAMES, load_app


def _read_launcher_file() -> str:
    path = Path(__file__).parent / ".app_select.json"
    try:
        data = json.loads(path.read_text())
        name = data.get("app", "")
        if name in APP_NAMES:
            return name
        logging.warning(".app_select.json: unknown app %r, falling back to first app", name)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        logging.warning(".app_select.json: invalid JSON (%s), falling back to first app", e)
    return APP_NAMES[0]


if __name__ == '__main__':
    process_id = os.getpid()
    psutil.Process(process_id).nice(psutil.HIGH_PRIORITY_CLASS)  # Prioritize over normal apps, below system processes
    # The GIL switch interval is the unit of time a thread waits for its turn at the interpreter.
    # 0.5 ms (2026-09) starved the pose runner's callback pipeline — its 2-slot result queue
    # overflowed ("Callback queue full") and pose recognition collapsed — but that predates the
    # vectorized filters, batched synth/text and lazy windows that shrank every burst. 2 ms is the
    # cautious middle: shorter waits for the render and reader threads at their deadlines.
    # Watchdogs: the pinned dashboard chips, and "Callback queue full" floods in the log — on
    # either regression, delete this line (5 ms default).
    sys.setswitchinterval(0.002)
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 tensor cores for faster matmul on Ampere+ GPUs
    except ImportError:
        pass

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-app',     '--app',            type=str,   default=None, choices=APP_NAMES, help='app to launch (default: from .app_select.json or first app)')
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecorded video with camera')
    parser.add_argument('-ni',      '--no-interface',   action='store_true',        help='do not open the settings interface in an Edge window')
    parser.add_argument('-v',       '--verbose',        action='store_true',        help='enable verbose (DEBUG) console output')
    parser.add_argument('-timing',  '--timing',         type=float, default=0.0,    help='seconds of timing session: write a timing report to logs/, then quit (0 = off)')
    parser.add_argument('-warmup',  '--timing-warmup',  type=float, default=30.0,   help='seconds before the timing session starts')
    args: Namespace = parser.parse_args()
    if args.app is None:
        args.app = _read_launcher_file()   # its warnings go to stderr: the log file is named after the app
    log_file = setup_logging(verbose=args.verbose, name=args.app)
    install_excepthooks()

    logging.info("Starting  >>> %s <<<  pid: %s  simulation: %s", args.app, process_id, args.simulation)
    logging.info("Logging to: %s", log_file)
    logging.info(crash_summary())

    AppClass = load_app(args.app)
    app = AppClass(simulation=args.simulation, interface=not args.no_interface)
    app.start()

    shutdown_event = Event()
    start_time = time.monotonic()

    def _heartbeat() -> None:
        while not shutdown_event.wait(60):
            elapsed = int(time.monotonic() - start_time)
            logging.debug("Heartbeat — running for %d min %d s", elapsed // 60, elapsed % 60)

    Thread(target=_heartbeat, daemon=True, name="heartbeat").start()

    def _timing_session() -> None:
        stats = timing.run_session(args.timing_warmup, args.timing, shutdown_event)
        if stats is None:
            return
        timing.write_report(log_file.parent, args.app, args.simulation, args.timing_warmup, args.timing, stats)
        if app.is_running:
            app.stop()

    if args.timing > 0:
        if not args.simulation:
            logging.warning("Timing session without -sim: the report is not comparable to another run")
        Thread(target=_timing_session, daemon=True, name="timing").start()

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
