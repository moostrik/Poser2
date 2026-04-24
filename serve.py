import logging
from argparse import ArgumentParser, Namespace
from signal import signal, SIGINT
from threading import Event

from modules.log_config import setup_logging, install_thread_excepthook
from modules.settings import LauncherServer
from apps import APP_REGISTRY


if __name__ == '__main__':
    parser = ArgumentParser(description='Poser dev launcher — start/stop apps from a browser.')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose (DEBUG) console output')
    args: Namespace = parser.parse_args()

    log_file = setup_logging(verbose=args.verbose)
    install_thread_excepthook()
    logging.info("Logging to: %s", log_file)

    server = LauncherServer(port=667, app_registry=APP_REGISTRY)
    server.start()

    shutdown_event = Event()

    def signal_handler_exit(sig, frame) -> None:
        logging.info("Received interrupt signal, shutting down...")
        shutdown_event.set()
        server.stop()

    signal(SIGINT, signal_handler_exit)

    while not shutdown_event.is_set():
        shutdown_event.wait(0.1)

    from os import _exit
    _exit(0)
