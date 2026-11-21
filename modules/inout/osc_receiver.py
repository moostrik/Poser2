
from threading import Thread, Lock
from time import perf_counter
from typing import Callable

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

from modules.settings import BaseSettings, Field, Widget
from modules.utils import Broadcast
from .net_probe import is_valid_ip
from .osc_client import ERROR_LOG_INTERVAL, is_valid_port

import logging
logger = logging.getLogger(__name__)

class OscReceiverSettings(BaseSettings):
    port_in:         Field[int]   = Field(9000,        min=1024, max=65535, widget=Widget.number,   description="Incoming OSC port")
    return_messages: Field[bool]  = Field(True,                                                     description="Echo received messages back to sender")
    port_out:        Field[int]   = Field(9001,        min=1024, max=65535, widget=Widget.number,   description="Outgoing OSC port")
    ip_out:          Field[str]   = Field("127.0.0.1",                     widget=Widget.ip_field, description="Outgoing OSC IP address")
    verbose:         Field[bool]  = Field(False,                                                    description="Log received messages")
    counter:         Field[int]   = Field(0,           min=0,    max=99999, widget=Widget.number,   access=Field.READ, description="Message activity")
    last_address:    Field[str]   = Field("",                                                       access=Field.READ, description="Last received OSC address")
    unbound_counter: Field[int]   = Field(0,           min=0,    max=99999, widget=Widget.number,   access=Field.READ, description="Messages received on an address nothing is bound to")
    last_unbound:    Field[str]   = Field("",                                                       access=Field.READ, description="Last received OSC address nothing is bound to")


class OscReceiver:
    """Calls what is bound to an address with the arguments of each message to it, on the server's thread, then
    echoes the message to the return address. It binds its settings at `start` and unbinds them at `stop`; a port
    change restarts the server with the same bindings, and an invalid return address keeps the current one."""

    def __init__(self, settings: OscReceiverSettings) -> None:
        self.settings: OscReceiverSettings = settings
        self._bindings: dict[str, Broadcast] = {}
        self._warned_unbound: set[str] = set()
        self._client_lock: Lock = Lock()
        self._return_client: SimpleUDPClient | None = None
        self._echo_error_time: float = -ERROR_LOG_INTERVAL     # when a failed echo was last logged
        self._server_lock: Lock = Lock()     # the server, and the bindings it maps
        self._server: BlockingOSCUDPServer | None = None
        self._running: bool = False

    def start(self) -> None:
        self._on_out_change()
        with self._server_lock:
            self._running = True
            self._server = self._start_server(self.settings.port_in)
        self.settings.bind(OscReceiverSettings.ip_out,   self._on_out_change)
        self.settings.bind(OscReceiverSettings.port_out, self._on_out_change)
        self.settings.bind(OscReceiverSettings.port_in,  self._on_in_change)

    def stop(self) -> None:
        self.settings.unbind(OscReceiverSettings.ip_out,   self._on_out_change)
        self.settings.unbind(OscReceiverSettings.port_out, self._on_out_change)
        self.settings.unbind(OscReceiverSettings.port_in,  self._on_in_change)
        with self._server_lock:
            self._running = False
            if self._server is not None:
                _close(self._server)
                self._server = None
        with self._client_lock:
            if self._return_client is not None:
                self._return_client.close()
                self._return_client = None

    def bind(self, address: str, callback: Callable) -> None:
        with self._server_lock:
            if address not in self._bindings:
                self._bindings[address] = Broadcast(name=f'OSC {address}')
                if self._server is not None:
                    self._server.dispatcher.map(address, self._handle)
            self._bindings[address].add_callback(callback)

    def _handle(self, address: str, *args) -> None:
        if self.settings.verbose:
            logger.info(f"{address} {args}")
        self.settings.counter      = (self.settings.counter + 1) % 100000
        self.settings.last_address = address
        callbacks: Broadcast | None = self._bindings.get(address)
        if callbacks is not None:
            callbacks(*args)
        if self.settings.return_messages:
            self._echo(address, args)

    def _echo(self, address: str, args: tuple) -> None:
        """The message back to the return address; a failed echo is logged at most once per `ERROR_LOG_INTERVAL`."""
        with self._client_lock:
            if self._return_client is None:
                return
            try:
                self._return_client.send_message(address, args)
            except OSError:
                now: float = perf_counter()
                if now - self._echo_error_time > ERROR_LOG_INTERVAL:
                    self._echo_error_time = now
                    logger.exception('OSC receiver: the echo of %s to %s:%s failed', address, self.settings.ip_out,
                                     self.settings.port_out)

    def _handle_unbound(self, address: str, *args) -> None:
        if self.settings.verbose:
            logger.info(f"unbound {address} {args}")
        if address not in self._warned_unbound:
            self._warned_unbound.add(address)
            logger.warning(f"Received '{address}' but nothing is bound to it; bound: {sorted(self._bindings)}")
        self.settings.unbound_counter = (self.settings.unbound_counter + 1) % 100000
        self.settings.last_unbound    = address

    def _start_server(self, port: int) -> BlockingOSCUDPServer:
        """A server on `port` that maps every bound address; called under the server lock."""
        dispatcher = Dispatcher()
        for address in self._bindings:
            dispatcher.map(address, self._handle)
        dispatcher.set_default_handler(self._handle_unbound)
        server = BlockingOSCUDPServer(('0.0.0.0', port), dispatcher)
        Thread(target=server.serve_forever, daemon=True, name=f'OscReceiver-{port}').start()
        logger.info(f"listening on port {port}")
        return server

    def _on_out_change(self, _=None) -> None:
        ip: str = self.settings.ip_out
        port: int = self.settings.port_out
        if not is_valid_ip(ip) or not is_valid_port(port):
            logger.warning('OSC receiver: invalid return address %s:%s, keeping the current one', ip, port)
            return
        client = SimpleUDPClient(ip, port)
        with self._client_lock:
            if self._return_client is not None:
                self._return_client.close()
            self._return_client = client
        logger.info(f"Return address updated to {ip}:{port}")

    def _on_in_change(self, _=None) -> None:
        # off the caller's thread: a shutdown waits for the server's loop to end
        Thread(target=self._restart, daemon=True, name='OscReceiver-restart').start()

    def _restart(self) -> None:
        with self._server_lock:
            if not self._running:
                return
            if self._server is not None:
                _close(self._server)
            try:
                self._server = self._start_server(self.settings.port_in)
            except OSError:
                self._server = None
                logger.exception('OSC receiver: port %s cannot be opened', self.settings.port_in)


def _close(server: BlockingOSCUDPServer) -> None:
    server.shutdown()
    server.server_close()
