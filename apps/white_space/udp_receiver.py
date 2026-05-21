import ctypes
import socket
from threading import Thread
from time import time
from typing import Callable, Optional

from modules.settings import BaseSettings, Field, Widget

import logging
logger = logging.getLogger(__name__)


class UdpReceiverSettings(BaseSettings):
    port:         Field[int]   = Field(9001, min=1024, max=65535, widget=Widget.number_field, description="Incoming UDP port")
    verbose:      Field[bool]  = Field(False,                                                 description="Log received messages")
    counter:      Field[int]   = Field(0,    min=0,    max=99999, access=Field.READ,          description="Message activity")
    last_address: Field[str]   = Field("",                        access=Field.READ,          description="Last received message")
    last_time:    Field[float] = Field(0.0,                       access=Field.READ,          description="Wall-clock time of last received message")
    interval:     Field[float] = Field(0.0,                       access=Field.READ,          description="Interval between last two messages (s)")


class UdpReceiver:
    """Receives plain UDP packets and dispatches by exact string address.

    The sender does not need to support OSC — any UDP payload whose decoded
    text matches a registered address triggers the callbacks for that address.

    Example::

        receiver.bind("/WS/sensor/fall", lambda *_: on_bang())
    """

    def __init__(self, settings: UdpReceiverSettings) -> None:
        self._config   = settings
        self._bindings: dict[str, list[Callable]] = {}
        self._running  = False
        self._thread: Optional[Thread] = None

    @property
    def running(self) -> bool:
        return self._running

    def bind(self, address: str, callback: Callable) -> None:
        if address not in self._bindings:
            self._bindings[address] = []
        self._bindings[address].append(callback)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="UdpReceiver")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, 15)  # THREAD_PRIORITY_TIME_CRITICAL
        except Exception:
            pass

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", self._config.port))
            sock.settimeout(0.5)
            logger.info(f"UdpReceiver: listening on port {self._config.port}")

            prev_time: float = 0.0
            while self._running:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue

                try:
                    address = data.split(b"\x00")[0].decode("utf-8").strip()
                except UnicodeDecodeError:
                    if self._config.verbose:
                        logger.warning(f"UdpReceiver: non-UTF-8 data ({len(data)} bytes) from {_}")
                    continue

                if self._config.verbose:
                    bound = address in self._bindings
                    logger.info(f"UdpReceiver: '{address}'"
                                + ("" if bound else " (no binding)"))

                now = time()
                self._config.counter      = (self._config.counter + 1) % 100000
                self._config.last_address = address
                if prev_time:
                    self._config.interval = now - prev_time
                self._config.last_time    = now
                prev_time = now
                for callback in self._bindings.get(address, []):
                    try:
                        callback()
                    except Exception as e:
                        logger.warning(f"UdpReceiver callback error for '{address}': {e}")

        except OSError as e:
            logger.error(f"UdpReceiver socket error on port {self._config.port}: {e}")
            self._running = False
        finally:
            sock.close()
