
from dataclasses import dataclass
from threading import Thread, Lock
from typing import Callable


from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

from modules.settings import BaseSettings, Field, Widget

import logging
logger = logging.getLogger(__name__)

class OscControlSettings(BaseSettings):
    port_in         = Field(9000,        min=1024, max=65535, widget=Widget.number,   description="Incoming OSC port")
    return_messages = Field(True,                                                     description="Echo received messages back to sender")
    port_out        = Field(9001,        min=1024, max=65535, widget=Widget.number,   description="Outgoing OSC port")
    ip_address_out  = Field("127.0.0.1",                      widget=Widget.ip_field, description="Outgoing OSC IP address")
    verbose         = Field(False,                                                    description="Log received messages")


@dataclass
class ControlMessage:
    address: str
    arguments: list

ControlMessageCallback = Callable[[ControlMessage], None]


class OscControl:

    def __init__(self, config: OscControlSettings) -> None:

        self.config: OscControlSettings = config

        self.osc_receive: Dispatcher = Dispatcher()
        self.osc_receive.set_default_handler(self._osc_handler, needs_reply_address=True)
        self.server = ThreadingOSCUDPServer(('0.0.0.0', config.port_in), self.osc_receive)
        self.server_thread: Thread = Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.osc_return_client = SimpleUDPClient(config.ip_address_out, config.port_out)
        self.client_lock = Lock()

        self.callback_lock = Lock()
        self.message_callbacks: list[ControlMessageCallback] = []

        config.bind(OscControlSettings.ip_address_out, self._on_out_change)
        config.bind(OscControlSettings.port_out,       self._on_out_change)
        config.bind(OscControlSettings.port_in,        self._on_in_change)


    def _osc_handler(self, client_address, address, *args) -> None:
        if self.config.verbose:
            logger.info(f"ControlOsc: From {client_address}: {address} {args}")

        if self.config.return_messages:
            with self.client_lock:
                self.osc_return_client.send_message(address, args)

        message = ControlMessage(address=address, arguments=list(args))

        self._notify_message(message)

    def _on_out_change(self, _=None) -> None:
        with self.client_lock:
            self.osc_return_client = SimpleUDPClient(self.config.ip_address_out, self.config.port_out)
        logger.info(f"ControlOsc: Return address updated to {self.config.ip_address_out}:{self.config.port_out}")

    def _on_in_change(self, _=None) -> None:
        def _restart() -> None:
            self.server.shutdown()
            self.osc_receive = Dispatcher()
            self.osc_receive.set_default_handler(self._osc_handler, needs_reply_address=True)
            self.server = ThreadingOSCUDPServer(('0.0.0.0', self.config.port_in), self.osc_receive)
            self.server_thread = Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            logger.info(f"ControlOsc: Listening on port {self.config.port_in}")
        Thread(target=_restart, daemon=True).start()


    def _notify_message(self, message: ControlMessage) -> None:
        with self.callback_lock:
            for callback in self.message_callbacks:
                callback(message)


    def register_message_callback(self, callback: ControlMessageCallback) -> None:
        with self.callback_lock:
            if callback not in self.message_callbacks:
                self.message_callbacks.append(callback)


    def unregister_message_callback(self, callback: ControlMessageCallback) -> None:
        with self.callback_lock:
            if callback in self.message_callbacks:
                self.message_callbacks.remove(callback)