
from threading import Thread, Lock
from time import time
from typing import Callable


from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

from modules.settings import BaseSettings, Field, Widget

import logging
logger = logging.getLogger(__name__)

class OscReceiverSettings(BaseSettings):
    port_in         = Field(9000,        min=1024, max=65535, widget=Widget.number,   description="Incoming OSC port")
    return_messages = Field(True,                                                     description="Echo received messages back to sender")
    port_out        = Field(9001,        min=1024, max=65535, widget=Widget.number,   description="Outgoing OSC port")
    ip_address_out  = Field("127.0.0.1",                      widget=Widget.ip_field, description="Outgoing OSC IP address")
    verbose         = Field(False,                                                    description="Log received messages")
    counter         = Field(0,           min=0,    max=99999, widget=Widget.number,   access=Field.READ, description="Message activity")
    last_address    = Field("",                                                       access=Field.READ, description="Last received OSC address")
    last_time       = Field(0.0,                                                      access=Field.READ, description="Wall-clock time of last received message")


class OscReceiver:

    def __init__(self, settings: OscReceiverSettings) -> None:
        self.settings: OscReceiverSettings = settings
        self._bindings: dict[str, list[Callable]] = {}
        self.client_lock = Lock()
        self.osc_return_client = SimpleUDPClient(settings.ip_address_out, settings.port_out)
        self.server: BlockingOSCUDPServer | None = None

        settings.bind(OscReceiverSettings.ip_address_out, self._on_out_change)
        settings.bind(OscReceiverSettings.port_out,       self._on_out_change)
        settings.bind(OscReceiverSettings.port_in,        self._on_in_change)

    def start(self) -> None:
        self.server = self._start_server(self.settings.port_in)

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server = None

    def bind(self, address: str, callback: Callable) -> None:
        if address not in self._bindings:
            self._bindings[address] = []
            if self.server is not None:
                self.server.dispatcher.map(address, self._handle)
        self._bindings[address].append(callback)

    def _handle(self, address: str, *args) -> None:
        if self.settings.verbose:
            logger.info(f"{address} {args}")
        if self.settings.return_messages:
            with self.client_lock:
                self.osc_return_client.send_message(address, args)
        self.settings.counter      = (self.settings.counter + 1) % 100000
        self.settings.last_address = address
        self.settings.last_time    = time()
        for callback in self._bindings.get(address, []):
            try:
                callback(*args)
            except TypeError as e:
                logger.warning(f"Argument mismatch for '{address}': {e}")

    def _start_server(self, port: int) -> BlockingOSCUDPServer:
        dispatcher = Dispatcher()
        for address in self._bindings:
            dispatcher.map(address, self._handle)
        server = BlockingOSCUDPServer(('0.0.0.0', port), dispatcher)
        Thread(target=server.serve_forever, daemon=True).start()
        logger.info(f"OscReceiver: listening on port {port}")
        return server

    def _on_out_change(self, _=None) -> None:
        with self.client_lock:
            self.osc_return_client = SimpleUDPClient(self.settings.ip_address_out, self.settings.port_out)
        logger.info(f"Return address updated to {self.settings.ip_address_out}:{self.settings.port_out}")

    def _on_in_change(self, _=None) -> None:
        if self.server is None:
            return
        old_server = self.server
        def _restart() -> None:
            if old_server is not None:
                old_server.shutdown()
            self.server = self._start_server(self.settings.port_in)
        Thread(target=_restart, daemon=True).start()