"""Art-Net sender: a thread that sends its owner's DMX packet to one or more consecutive universes, as ArtDmx
packets. The owner fills the packet in `fill`, called from the thread before every tick."""

from collections import deque
from enum import IntEnum
from threading import Thread, Lock
from typing import Callable
import logging
import socket
import time

from modules.settings import BaseSettings, Field, Widget
from modules.utils import Waker
from .net_probe import is_valid_ip
from .osc_client import ERROR_LOG_INTERVAL

logger = logging.getLogger(__name__)

MAX_FPS: float = 44.0       # Hz: a DMX universe refreshes at most 44 times a second
KEEP_ALIVE: float = 0.9     # s: Art-Net 4 has an unchanged universe resent every 800 to 1000 ms
ARTNET_PORT: int = 6454
MAX_CHANNELS: int = 512     # a DMX universe


def artdmx_header(universe: int, size: int) -> bytes:
    """An ArtDmx packet's 18 bytes before its data: the ID, the opcode (low byte first), the protocol version 14,
    sequence 0 (off), physical port 0, the universe (low byte first) and the data's length (high byte first)."""
    return (b'Art-Net\x00' + bytes([0x00, 0x50, 0x00, 14, 0, 0]) + bytes([universe & 0xFF, universe >> 8])
            + bytes([size >> 8, size & 0xFF]))


class ArtNetSenderSettings(BaseSettings):
    enabled:  Field[bool]  = Field(True, widget=Widget.switch, description="Send Art-Net to the lights; off is black")
    ip:       Field[str]   = Field("127.0.0.1", widget=Widget.ip_field, description="Art-Net node IP address")
    universe: Field[int]   = Field(0, min=0, max=254, widget=Widget.number_field, description="Art-Net universe, the first of its universes")
    fps:      Field[float] = Field(40.0, min=1.0, max=MAX_FPS, step=1.0, description="Ticks per second; a tick sends what changed")


class _Frame(IntEnum):
    """A one-tick frame on every universe in place of the packet."""
    FLASH = 0       # all channels full
    BLACKOUT = 1    # all channels zero


def is_valid_universe(value: int) -> bool:
    return value in range(0, 255)


class ArtNetSender:
    """Sends on a fixed-rate grid, resynced after a late tick. A universe goes out when its part of the packet differs
    from what it last sent, and at least every `KEEP_ALIVE`. A blackout goes out at start, when switched off and at
    stop; with `flash`, a flash goes out before each of them and when switched on or readdressed."""

    def __init__(self, settings: ArtNetSenderSettings, channels: int, fill: Callable[[bytearray], None],
                 universes: int = 1, flash: bool = True) -> None:
        """`channels` DMX channels per universe; `fill` writes the packet, `universes` parts of `packet_size` one
        after the other, before each tick on the sender's thread."""
        self.settings: ArtNetSenderSettings = settings
        self.fill: Callable[[bytearray], None] = fill
        self.universes: int = universes
        self.flash: bool = flash

        # even: an ArtDmx packet's data is an even number of channels, 2 to 512
        self.packet_size: int = max(channels + channels % 2, 2)
        if self.packet_size > MAX_CHANNELS:
            raise ValueError(f'{channels} channels do not fit in a DMX universe ({MAX_CHANNELS})')
        self.packet: bytearray = bytearray(self.packet_size * universes)

        self._lock: Lock = Lock()                           # the socket, its target and the frames
        self._socket: socket.socket | None = None
        self._target: tuple[str, int] = ('', ARTNET_PORT)
        self._headers: list[bytes] = []                     # one per universe
        self._error_time: float = -ERROR_LOG_INTERVAL       # when a failed send was last logged
        self._frames: deque[_Frame] = deque()               # one per tick, before the packet
        self._last_sent: list[bytes | None] = [None] * universes    # None: send the packet on the next tick
        self._last_time: list[float] = [0.0] * universes
        self._running: bool = False
        self._thread: Thread | None = None
        self._waker: Waker | None = None                    # set by stop: ends the wait for the next tick at once

    def start(self) -> None:
        if self._running:
            return
        with self._lock:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._update_target()
        self._queue_off()
        self.settings.bind(ArtNetSenderSettings.ip, self._on_address)
        self.settings.bind(ArtNetSenderSettings.universe, self._on_address)
        self.settings.bind(ArtNetSenderSettings.enabled, self._on_enabled)
        self._waker = Waker()
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name='ArtNetSender')
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self.settings.unbind(ArtNetSenderSettings.ip, self._on_address)
        self.settings.unbind(ArtNetSenderSettings.universe, self._on_address)
        self.settings.unbind(ArtNetSenderSettings.enabled, self._on_enabled)
        self._running = False
        assert self._waker is not None
        self._waker.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        try:
            self._end()
        except Exception:
            logger.exception('Art-Net: the blackout at stop failed')
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
        self._waker.close()
        self._waker = None

    def _run(self) -> None:
        assert self._waker is not None
        next_time: float = time.perf_counter()
        while self._running:
            try:
                self.update(next_time)
            except Exception:
                logger.exception('Art-Net send failed')
            interval: float = self._interval()
            now: float = time.perf_counter()
            if now - next_time > interval:          # fell behind: resync instead of bursting
                next_time = now
            next_time += interval
            self._waker.wait(next_time)

    def _end(self) -> None:
        """The flash and, one interval later, the blackout."""
        if self.flash:
            self._send_frame(_Frame.FLASH)
            assert self._waker is not None
            self._waker.wait(time.perf_counter() + self._interval())
        self._send_frame(_Frame.BLACKOUT)

    def _interval(self) -> float:
        return 1.0 / min(max(self.settings.fps, 1.0), MAX_FPS)

    def update(self, now: float) -> None:
        """One tick at `now`: a queued frame, or the universes whose part changed or is due for its keep-alive."""
        with self._lock:
            frame: _Frame | None = self._frames.popleft() if self._frames else None
        if frame is not None:
            self._send_frame(frame)
            return
        if not self.settings.enabled:
            return
        self.fill(self.packet)
        with self._lock:
            for universe in range(len(self._headers)):
                part = bytes(self.packet[universe * self.packet_size:(universe + 1) * self.packet_size])
                if part != self._last_sent[universe] or now - self._last_time[universe] >= KEEP_ALIVE:
                    self._send(universe, part)
                    self._last_sent[universe] = part
                    self._last_time[universe] = now

    def _send_frame(self, frame: _Frame) -> None:
        data: bytes = bytes([255 if frame == _Frame.FLASH else 0]) * self.packet_size
        with self._lock:
            for universe in range(len(self._headers)):
                self._send(universe, data)
                self._last_sent[universe] = None

    def _send(self, universe: int, data: bytes) -> None:
        """One ArtDmx packet; a failed send is logged at most once per `ERROR_LOG_INTERVAL`. Called under the lock."""
        if self._socket is None:
            return
        try:
            self._socket.sendto(self._headers[universe] + data, self._target)
        except OSError:
            now: float = time.perf_counter()
            if now - self._error_time > ERROR_LOG_INTERVAL:
                self._error_time = now
                logger.exception('Art-Net send to %s universe %s failed', self._target[0],
                                 self.settings.universe + universe)

    def _queue_off(self) -> None:
        """The flash, where it is on, and the blackout, on the next ticks."""
        with self._lock:
            if self.flash:
                self._frames.append(_Frame.FLASH)
            self._frames.append(_Frame.BLACKOUT)

    def _on_address(self, _=None) -> None:
        self._update_target()
        if self.flash and self.settings.enabled:
            with self._lock:
                self._frames.append(_Frame.FLASH)

    def _on_enabled(self, enabled: bool) -> None:
        if not enabled:
            self._queue_off()
        elif self.flash:
            with self._lock:
                self._frames.append(_Frame.FLASH)

    def _update_target(self) -> None:
        ip: str = self.settings.ip
        universe: int = self.settings.universe
        if not is_valid_ip(ip) or not is_valid_universe(universe + self.universes - 1):
            logger.warning('Invalid Art-Net address %s universe %s, keeping the current one', ip, universe)
            return

        with self._lock:
            self._target = (ip, ARTNET_PORT)
            self._headers = [artdmx_header(universe + i, self.packet_size) for i in range(self.universes)]
            self._last_sent = [None] * self.universes
        logger.info('Art-Net sending to %s on universe %s (%s universes)', ip, universe, self.universes)
