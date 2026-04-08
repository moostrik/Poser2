
from threading import Thread, Lock
from typing import Callable


from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import ThreadingOSCUDPServer
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
    counter         = Field(0,           min=0,    max=99999, widget=Widget.number,   access=Field.READ,      description="Message activity")


class OscReceiver:

    def __init__(self, settings: OscReceiverSettings) -> None:
        self.settings: OscReceiverSettings = settings
        self._bindings: dict[str, list[Callable]] = {}
        self.client_lock = Lock()
        self.osc_return_client = SimpleUDPClient(settings.ip_address_out, settings.port_out)
        self.server = self._start_server(settings.port_in)

        settings.bind(OscReceiverSettings.ip_address_out, self._on_out_change)
        settings.bind(OscReceiverSettings.port_out,       self._on_out_change)
        settings.bind(OscReceiverSettings.port_in,        self._on_in_change)

    def bind(self, address: str, callback: Callable) -> None:
        if address not in self._bindings:
            self._bindings[address] = []
            self.server.dispatcher.map(address, self._handle)
        self._bindings[address].append(callback)

    def _handle(self, address: str, *args) -> None:
        if self.settings.verbose:
            logger.info(f"{address} {args}")
        if self.settings.return_messages:
            with self.client_lock:
                self.osc_return_client.send_message(address, args)
        self.settings.counter = (self.settings.counter + 1) % 100000
        for callback in self._bindings.get(address, []):
            try:
                callback(*args)
            except TypeError as e:
                logger.warning(f"Argument mismatch for '{address}': {e}")

    def _start_server(self, port: int) -> ThreadingOSCUDPServer:
        dispatcher = Dispatcher()
        for address in self._bindings:
            dispatcher.map(address, self._handle)
        server = ThreadingOSCUDPServer(('0.0.0.0', port), dispatcher)
        Thread(target=server.serve_forever, daemon=True).start()
        logger.info(f"Listening on port {port}")
        return server

    def _on_out_change(self, _=None) -> None:
        with self.client_lock:
            self.osc_return_client = SimpleUDPClient(self.settings.ip_address_out, self.settings.port_out)
        logger.info(f"Return address updated to {self.settings.ip_address_out}:{self.settings.port_out}")

    def _on_in_change(self, _=None) -> None:
        old_server = self.server
        def _restart() -> None:
            old_server.shutdown()
            self.server = self._start_server(self.settings.port_in)
        Thread(target=_restart, daemon=True).start()