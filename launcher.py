
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


import sys
import logging
from datetime import datetime
from pathlib import Path


# Tee class to write output to both console and log file
class Tee:
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file  # Binary unbuffered file

    def write(self, data):
        self.console.write(data)
        self.log_file.write(data.encode('utf-8'))
        return len(data)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return self.console.isatty() if hasattr(self.console, 'isatty') else False


# Setup logging to file with timestamp
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"poser_{timestamp}.log"

# Binary unbuffered - writes immediately to disk, survives crashes
log_handle = open(log_file, 'wb', buffering=0)

# Save original stdout/stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = Tee(original_stdout, log_handle)
sys.stderr = Tee(original_stderr, log_handle)

# Setup logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(original_stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

logging.info(f"Logging to: {log_file}")
logging.info(f"Process PID: {os.getpid()}")



from threading import Event

import psutil
p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)


from argparse import ArgumentParser, Namespace
from os import path
from signal import signal, SIGINT

from modules.Main import Main
from modules.Settings import Settings, ModelType

import multiprocessing as mp

# Enable PyTorch optimizations for modern GPUs (RTX 30xx/40xx/50xx)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__': # For Windows compatibility with multiprocessing
    mp.freeze_support()
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-np',      '--nopose',         action='store_true',        help='do not do pose detection')
    parser.add_argument('-sim',     '--simulation',     action='store_true',        help='use prerecorded video with camera')
    parser.add_argument('-s',       '--settings',       type=str, default='default',help='settings file')

    args: Namespace = parser.parse_args()

    settings_path: str = f"files/settings/{args.settings}.json"

    logging.info(f"Loading settings from: {settings_path}")
    settings: Settings = Settings.load(settings_path)

    if args.nopose:
        settings.pose.model_type = ModelType.NONE

    app = Main(settings, simulation=args.simulation)
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
