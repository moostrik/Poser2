
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
    port_in        = Field(9000, min=1024, max=65535, access=Field.INIT, widget=Widget.number, description="Incoming OSC port")
    ip_address_in  = Field("127.0.0.1",              access=Field.INIT, widget=Widget.ip_field,     description="Incoming OSC IP address")
    return_messages = Field(True,                                                               description="Echo received messages back to sender")
    port_out       = Field(9001, min=1024, max=65535, access=Field.INIT, widget=Widget.number, description="Outgoing OSC port")
    ip_address_out = Field("127.0.0.1",              access=Field.INIT, widget=Widget.ip_field,     description="Outgoing OSC IP address")


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
        self.server = ThreadingOSCUDPServer((config.ip_address_in, config.port_in), self.osc_receive)
        self.server_thread: Thread = Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.osc_return_client = SimpleUDPClient(config.ip_address_out, config.port_out)

        self.callback_lock = Lock()
        self.message_callbacks: list[ControlMessageCallback] = []


    def _osc_handler(self, client_address, address, *args) -> None:
        if client_address[0] != self.config.ip_address_in:
            logger.info(f"ControlOsc: Ignoring message from unauthorized IP: {client_address[0]}")
            return

        logger.info(f"ControlOsc: From {client_address}: {address} {args}")

        if self.config.return_messages:
            self.osc_return_client.send_message(address, args)

        message = ControlMessage(address=address, arguments=list(args))

        self._notify_message(message)


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