"""OSC sender: a thread that bundles what was queued since its last tick and sends it at the settings' rate."""

from queue import Queue, Empty
from threading import Thread, Lock
from time import time, sleep
from typing import Any
import logging

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_bundle import OscBundle
from pythonosc import osc_bundle_builder, osc_message_builder

from modules.settings import BaseSettings, Field, Widget
from .net_probe import is_valid_ip

logger = logging.getLogger(__name__)

ArgValue = str | bytes | bool | int | float | list | tuple

ERROR_LOG_INTERVAL: float = 10.0    # s between the log lines of a send that keeps failing


class OscSenderSettings(BaseSettings):
    enabled: Field[bool]  = Field(True, widget=Widget.switch, description="Send OSC messages")
    ip:      Field[str]   = Field("127.0.0.1", widget=Widget.ip_field, description="Target OSC IP address")
    port:    Field[int]   = Field(9000, min=1024, max=65535, widget=Widget.number_field, description="Target OSC port")
    fps:     Field[float] = Field(30.0, min=1.0, max=120.0, step=1.0, description="Bundles sent per second")


def is_valid_port(port: int) -> bool:
    return port in range(1, 65536)


class OscSender(Thread):
    """Sends what `submit` queued as one bundle per tick: of a state address only the newest value, of an
    event address (`events`) every message, since each one counts."""

    def __init__(self, settings: OscSenderSettings, events: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.settings: OscSenderSettings = settings
        self.events: tuple[str, ...] = events

        self.client: SimpleUDPClient | None = None

        self.running: bool = False
        self.queue: Queue = Queue()
        self.mutex: Lock = Lock()
        self.last_message_time: float = time()
        self._error_time: float = 0.0       # when a failed send was last logged

    def start(self) -> None:
        if self.running:
            return
        self._update_client()
        self.settings.bind(OscSenderSettings.ip, self._on_address)
        self.settings.bind(OscSenderSettings.port, self._on_address)
        self.running = True
        super().start()

    def stop(self) -> None:
        if self.running:
            self.settings.unbind(OscSenderSettings.ip, self._on_address)
            self.settings.unbind(OscSenderSettings.port, self._on_address)
            self.running = False
            self.join()

    def run(self) -> None:
        while self.running:
            start_time = time()
            data_list = []
            try:
                while True:
                    data: Any = self.queue.get(False)
                    data_list.append(data)
            except Empty:
                pass

            if data_list:
                relevant_data = self.get_last_data(data_list)
                with self.mutex:
                    try:
                        if self.client:
                            bundle = self.build_bundle(relevant_data)
                            self.client.send(bundle)
                            self.last_message_time = time()
                    except Exception:
                        # this bundle is lost, the next one is sent: a sender that stopped here would
                        # stay silent for the rest of the run while its queue grows
                        if time() - self._error_time > ERROR_LOG_INTERVAL:
                            self._error_time = time()
                            logger.exception('osc error, bundle not sent')

            elapsed_time: float = time() - start_time
            update_interval: float = 1.0 / max(self.settings.fps, 1.0)
            sleep(max(0, update_interval - elapsed_time))

    def get_last_data(self, data_list: list[tuple[str, ArgValue]]) -> list[tuple[str, ArgValue]]:
        """The messages of a tick: every one of an event address, the newest value of the others."""
        grouped_data: dict[str, list[ArgValue]] = {}
        for address, args in data_list:
            grouped_data.setdefault(address, []).append(args)

        processed_data: list[tuple[str, ArgValue]] = []
        for address, args_list in grouped_data.items():
            if address in self.events:
                processed_data.extend((address, args) for args in args_list)
            else:
                processed_data.append((address, args_list[-1]))
        return processed_data

    def build_bundle(self, data_list: list[tuple[str, ArgValue]]) -> OscBundle:
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        for address, args in data_list:
            msg = osc_message_builder.OscMessageBuilder(address=address)
            # a tuple or list goes as separate arguments (as one add_arg it would be an OSC array, which Max does not take)
            for arg in (args if isinstance(args, (tuple, list)) else (args,)):
                msg.add_arg(arg)
            bundle.add_content(msg.build()) # type: ignore
        return bundle.build()

    def submit(self, address: str, args: ArgValue) -> None:
        # only while the thread takes them: nothing empties the queue otherwise
        if self.settings.enabled and self.running:
            self.queue.put((address, args))

    def _on_address(self, _=None) -> None:
        self._update_client()

    def _update_client(self) -> None:
        ip: str = self.settings.ip
        port: int = self.settings.port
        if not is_valid_ip(ip) or not is_valid_port(port):
            logger.warning('Invalid OSC address %s:%s, keeping the current client', ip, port)
            return
        with self.mutex:
            self.client = SimpleUDPClient(ip, port)
        logger.info('Osc Client Sending to %s %s', ip, port)
