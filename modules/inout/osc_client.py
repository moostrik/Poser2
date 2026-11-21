"""OSC client: one UDP socket to its owner's `ip` and `port`, that sends an encoded datagram."""

from ipaddress import ip_address
from threading import Lock
from time import perf_counter
import logging
import socket

from modules.settings import BaseSettings, Field
from .net_probe import is_valid_ip

logger = logging.getLogger(__name__)

ERROR_LOG_INTERVAL: float = 10.0    # s between the log lines of a send that keeps failing


def is_valid_port(port: int) -> bool:
    return port in range(1, 65536)


class OscClient:
    """Sends datagrams to the address in its owner's settings. It binds the two fields at `start` and unbinds them at
    `stop`; an invalid address keeps the current target. A failed send is logged, at most once per
    `ERROR_LOG_INTERVAL`, and the next one is sent."""

    def __init__(self, settings: BaseSettings, ip: Field[str], port: Field[int], name: str) -> None:
        """`ip` and `port` are the fields of `settings` that hold the target; `name` is the owner, for the log."""
        self.settings: BaseSettings = settings
        self._ip_field: Field[str] = ip
        self._port_field: Field[int] = port
        self._name: str = name
        self._lock: Lock = Lock()
        self._socket: socket.socket | None = None
        self._target: tuple[str, int] = ('', 0)
        self._error_time: float = -ERROR_LOG_INTERVAL

    def start(self) -> None:
        self._update()
        self.settings.bind(self._ip_field, self._on_address)
        self.settings.bind(self._port_field, self._on_address)

    def stop(self) -> None:
        self.settings.unbind(self._ip_field, self._on_address)
        self.settings.unbind(self._port_field, self._on_address)
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def send(self, datagram: bytes) -> bool:
        """Sends `datagram`; False when there is no target or the send failed."""
        with self._lock:
            if self._socket is None:
                return False
            try:
                self._socket.sendto(datagram, self._target)
                return True
            except OSError:
                now: float = perf_counter()
                if now - self._error_time > ERROR_LOG_INTERVAL:
                    self._error_time = now
                    logger.exception('%s: OSC send to %s:%s failed', self._name, *self._target)
                return False

    def _on_address(self, _=None) -> None:
        self._update()

    def _update(self) -> None:
        ip: str = getattr(self.settings, self._ip_field.name)
        port: int = getattr(self.settings, self._port_field.name)
        if not is_valid_ip(ip) or not is_valid_port(port):
            logger.warning('%s: invalid OSC address %s:%s, keeping the current one', self._name, ip, port)
            return
        family: int = socket.AF_INET6 if ip_address(ip).version == 6 else socket.AF_INET
        with self._lock:
            if self._socket is not None:
                self._socket.close()
            self._socket = socket.socket(family, socket.SOCK_DGRAM)
            self._target = (ip, port)
        logger.info('%s: sending OSC to %s:%s', self._name, ip, port)
