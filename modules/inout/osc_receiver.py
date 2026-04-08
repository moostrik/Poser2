
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
        self._bindings: dict[str, Callable] = {}

        self.osc_receive: Dispatcher = Dispatcher()
        self.osc_receive.set_default_handler(self._osc_handler, needs_reply_address=True)
        self.server = ThreadingOSCUDPServer(('0.0.0.0', settings.port_in), self.osc_receive)
        self.server_thread: Thread = Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.osc_return_client = SimpleUDPClient(settings.ip_address_out, settings.port_out)
        self.client_lock = Lock()

        settings.bind(OscReceiverSettings.ip_address_out, self._on_out_change)
        settings.bind(OscReceiverSettings.port_out,       self._on_out_change)
        settings.bind(OscReceiverSettings.port_in,        self._on_in_change)

    def bind(self, address: str, callback: Callable) -> None:
        self._bindings[address] = callback

    def _osc_handler(self, client_address, address, *args) -> None:
        if self.settings.verbose:
            logger.info(f"From {client_address}: {address} {args}")

        if self.settings.return_messages:
            with self.client_lock:
                self.osc_return_client.send_message(address, args)

        self.settings.counter = (self.settings.counter + 1) % 100000

        if address in self._bindings:
            try:
                self._bindings[address](*args)
            except TypeError as e:
                logger.warning(f"Argument mismatch for '{address}': {e}")
        else:
            logger.debug(f"Unbound address: {address}")

    def _on_out_change(self, _=None) -> None:
        with self.client_lock:
            self.osc_return_client = SimpleUDPClient(self.settings.ip_address_out, self.settings.port_out)
        logger.info(f"Return address updated to {self.settings.ip_address_out}:{self.settings.port_out}")

    def _on_in_change(self, _=None) -> None:
        def _restart() -> None:
            self.server.shutdown()
            self.osc_receive = Dispatcher()
            self.osc_receive.set_default_handler(self._osc_handler, needs_reply_address=True)
            self.server = ThreadingOSCUDPServer(('0.0.0.0', self.settings.port_in), self.osc_receive)
            self.server_thread = Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            logger.info(f"Listening on port {self.settings.port_in}")
        Thread(target=_restart, daemon=True).start()