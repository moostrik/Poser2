"""OSC sender: a thread that bundles what was queued since its last tick and sends it at the settings' rate."""

from queue import Queue, Empty
from threading import Thread
from time import perf_counter
import logging

from modules.settings import BaseSettings, Field, Widget
from modules.utils import Waker
from .osc_client import OscClient
from .osc_encoder import encode_message

logger = logging.getLogger(__name__)

ArgValue = str | int | float | list | tuple     # a list or tuple goes as separate arguments; a bool is refused


class OscSenderSettings(BaseSettings):
    enabled: Field[bool]  = Field(True, widget=Widget.switch, description="Send OSC messages")
    ip:      Field[str]   = Field("127.0.0.1", widget=Widget.ip_field, description="Target OSC IP address")
    port:    Field[int]   = Field(9000, min=1024, max=65535, widget=Widget.number_field, description="Target OSC port")
    fps:     Field[float] = Field(30.0, min=1.0, max=120.0, step=1.0, description="Bundles sent per second")


class OscSender:
    """Sends what `submit` queued as one bundle per tick, on a fixed-rate grid resynced after a late tick: of a
    state address only the newest value, of an event address (`events`) every message, since each one counts. An
    event goes out at once, with what else is queued; the grid keeps its phase."""

    def __init__(self, settings: OscSenderSettings, events: tuple[str, ...] = ()) -> None:
        self.settings: OscSenderSettings = settings
        self.events: tuple[str, ...] = events

        self._client: OscClient = OscClient(settings, OscSenderSettings.ip, OscSenderSettings.port, 'OscSender')
        self._queue: Queue[tuple[str, ArgValue]] = Queue()
        self._running: bool = False
        self._thread: Thread | None = None
        self._waker: Waker | None = None    # set by an event and by stop: the loop sends, or ends, at once

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._client.start()
        self._waker = Waker()
        self._running = True
        self._thread = Thread(target=self._run, args=(self._waker,), daemon=True, name='OscSender')
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        assert self._waker is not None
        self._waker.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._waker.close()
        self._waker = None
        self._client.stop()

    def submit(self, address: str, args: ArgValue) -> None:
        # only while the thread takes them: nothing empties the queue otherwise
        waker: Waker | None = self._waker
        if self.settings.enabled and self._running and waker is not None:
            self._queue.put((address, args))
            if address in self.events:
                waker.set()

    def _run(self, waker: Waker) -> None:
        next_time: float = perf_counter()
        while self._running:
            try:
                self._send_queued()
            except Exception:
                logger.exception('OscSender: a bundle was not sent')
            interval: float = 1.0 / max(self.settings.fps, 1.0)
            now: float = perf_counter()
            if now - next_time > interval:          # fell behind: resync instead of bursting
                next_time = now
            if not waker.wait(next_time + interval):
                next_time += interval

    def _send_queued(self) -> None:
        """What was queued since the last send, as one bundle."""
        messages: list[tuple[str, ArgValue]] = []
        try:
            while True:
                messages.append(self._queue.get(False))
        except Empty:
            pass
        if messages:
            self._client.send([_encode(address, args) for address, args in _tick_messages(messages, self.events)])


def _tick_messages(messages: list[tuple[str, ArgValue]], events: tuple[str, ...]) -> list[tuple[str, ArgValue]]:
    """The messages of a tick: every one of an event address, the newest value of the others."""
    grouped: dict[str, list[ArgValue]] = {}
    for address, args in messages:
        grouped.setdefault(address, []).append(args)

    tick: list[tuple[str, ArgValue]] = []
    for address, args_list in grouped.items():
        if address in events:
            tick.extend((address, args) for args in args_list)
        else:
            tick.append((address, args_list[-1]))
    return tick


def _encode(address: str, args: ArgValue) -> bytes:
    # a tuple or list goes as separate arguments (an OSC array, which Max does not take, would be one)
    return encode_message(address, args if isinstance(args, (tuple, list)) else (args,))
