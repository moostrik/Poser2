"""OSC in: a server on one port that hands each message's arguments to the handler of its address."""

import logging
from threading import Thread
from time import time
from typing import Callable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

logger = logging.getLogger(__name__)

ERROR_LOG_INTERVAL: float = 10.0    # s between the log lines of a handler that keeps raising


class Receiver:
    """OSC messages on `port` from this PC: `handlers` maps an address to a callable that gets the
    message's arguments. The server runs on a daemon thread; a handler that raises is logged (not at
    every message) and the server goes on (python-osc's own report goes to stderr, which pythonw has
    not). A port in use is a warning and the receiver stays off. `stop()` may run twice."""

    def __init__(self, port: int, handlers: dict[str, Callable[..., None]]) -> None:
        self.port: int = port
        self._handlers: dict[str, Callable[..., None]] = dict(handlers)
        self._server: BlockingOSCUDPServer | None = None
        self._thread: Thread | None = None
        self._started: bool = False     # the serving thread runs: only then may shutdown() be called
        self._error_time: float = 0.0   # when a failing handler was last logged

    @property
    def running(self) -> bool:
        return self._started and self._server is not None

    @property
    def address(self) -> tuple[str, int] | None:
        """The ip and port bound (port 0 lets the system pick one), None while not listening."""
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._server is not None:
            return
        dispatcher = Dispatcher()
        for address, handler in self._handlers.items():
            dispatcher.map(address, self._wrap(address, handler))
        try:
            self._server = BlockingOSCUDPServer(('127.0.0.1', self.port), dispatcher)
        except OSError as error:
            logger.warning('OSC port %d cannot be opened (%s): not listening', self.port, error)
            return
        self._thread = Thread(target=self._server.serve_forever, kwargs={'poll_interval': 0.1},
                              name=f'OscReceiver{self.port}', daemon=True)
        self._thread.start()
        self._started = True
        logger.info('Osc Receiver listening on %s:%d', *self.address)  # type: ignore[misc]

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        if self._started:
            # shutdown() waits for the serving loop to leave: only safe once the thread runs it
            self._started = False
            server.shutdown()
            if self._thread is not None:
                self._thread.join()
        server.server_close()

    def _wrap(self, address: str, handler: Callable[..., None]) -> Callable[..., None]:
        def handle(_address: str, *args) -> None:
            # returns None: a value would go back to the sender as a reply
            try:
                handler(*args)
            except Exception:
                if time() - self._error_time > ERROR_LOG_INTERVAL:
                    self._error_time = time()
                    logger.exception('OSC handler of %s failed on %s', address, args)
        return handle
