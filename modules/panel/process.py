"""The panel process: NiceGUI serving the page of a settings mirror, linked to the app that holds the tree.

``PanelServer`` starts it as a new Python running ``main()``. It reads the link's key from stdin,
listens on a free localhost port, prints the port and takes the app's one connection. What the page writes
changes the mirror, and the mirror sends it on; what the app sends is set on the mirror on NiceGUI's loop.
It ends with the connection: on the app's ``stop``, or when the app is gone.
"""

import logging
import os
import sys
import threading
import time
from functools import partial
from multiprocessing.connection import Connection, Listener
from typing import Any, Callable

from nicegui import Client, app as nicegui_app, core, ui

from modules.log_config import LogEntry, LogRingBuffer
from modules.settings import BaseSettings, Field, Widget, mirror, presets

from .app_window import APP_QUERY, SELF_CLOSE_SCRIPT, open_app_window
from .page import STATIC_DIR, STATIC_URL, build_page, local_ips
from .state import STATE_FILE, PanelState

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
CONNECT_TIMEOUT = 15.0  # seconds the process waits for the app's connection before it ends
REPLY_TIMEOUT = 5.0     # seconds a preset load or save waits for the app's answer


class AppLink:
    """The panel's end of the link: sends what the mirror is written, sets on it what the app sends.

    It is the page's `PresetStore` too: a load or save waits for the app's answer, on the loop that
    called it, which the receiver thread does not need. *on_loop* runs a call on the page's loop from
    the receiver thread; *on_end* is called when the app stops the link or is gone."""

    def __init__(self, conn: Connection, log_buffer: LogRingBuffer, *,
                 on_loop: Callable[..., Any], on_end: Callable[[], Any]) -> None:
        self.conn = conn
        self.log_buffer = log_buffer
        self.on_loop = on_loop
        self.on_end = on_end
        self._send_lock = threading.RLock()
        self._from_app = False      # set while the app's values are applied: those are not sent back
        self._bound: list[tuple[BaseSettings, Field, Callable[[Any], None]]] = []
        self._replies: dict[int, tuple[threading.Event, list[bool]]] = {}
        self._request_lock = threading.Lock()
        self._next_request = 0
        self.root: BaseSettings | None = None

    # -- the mirror ---------------------------------------------------------------

    def watch(self, root: BaseSettings) -> None:
        """Send every write to *root*'s fields and every press of its buttons."""
        self.root = root
        for path, settings in mirror.walk(root):
            for name, field in settings.fields.items():
                if field.widget is Widget.button:
                    callback: Callable[[Any], None] = partial(self._fired, path, name)
                else:
                    callback = partial(self._written, path, name, field)
                settings.bind(field, callback)
                self._bound.append((settings, field, callback))

    def close(self) -> None:
        for settings, field, callback in self._bound:
            settings.unbind(field, callback)
        self._bound.clear()

    def _written(self, path: str, name: str, field: Field, value: Any) -> None:
        if not self._from_app:
            self.send(("set", path, name, mirror.encode_value(field, value)))

    def _fired(self, path: str, name: str, _value: Any) -> None:
        self.send(("fire", path, name))

    def _apply(self, values: mirror.Values) -> None:
        """The app's values, on NiceGUI's loop, where the page reads the mirror."""
        assert self.root is not None
        self._from_app = True
        try:
            mirror.apply(self.root, values)
        finally:
            self._from_app = False

    # -- the page's requests ------------------------------------------------------

    def load(self, name: str) -> bool:
        return self._request("load", name)

    def save(self, name: str) -> bool:
        return self._request("save", name)

    def exit(self) -> None:
        self.send(("exit",))

    def _request(self, kind: str, name: str) -> bool:
        with self._request_lock:
            self._next_request += 1
            request = self._next_request
        event, answer = threading.Event(), [False]
        self._replies[request] = (event, answer)
        self.send((kind, request, name))
        if not event.wait(REPLY_TIMEOUT):
            logger.warning("Preset %s of '%s': no answer from the app in %.0f s", kind, name, REPLY_TIMEOUT)
        self._replies.pop(request, None)
        return answer[0]

    # -- the connection -----------------------------------------------------------

    def send(self, message: tuple) -> None:
        with self._send_lock:
            try:
                self.conn.send(message)
            except (OSError, ValueError):
                pass    # the app is gone; the receiver thread ends the process

    def receive(self) -> None:
        """The receiver thread: hand on what the app sends until it stops or is gone, then shut down."""
        while True:
            try:
                message = self.conn.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            if kind == "values":
                self.on_loop(self._apply, message[1])
            elif kind == "log":
                for entry in message[1]:
                    self.log_buffer.add(LogEntry(*entry))
            elif kind == "reply":
                reply = self._replies.get(message[1])
                if reply is not None:
                    reply[1][0] = message[2]
                    reply[0].set()
            elif kind == "stop":
                break
            else:
                logger.warning("Unknown message %r from the app", kind)
        self.on_end()


class LinkHandler(logging.Handler):
    """Sends the process's log records to the app, which logs them as its own: one log, the drawer's too."""

    def __init__(self, link: AppLink) -> None:
        super().__init__()
        self.link = link

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.link.send(("log", {
                "name": record.name, "levelno": record.levelno, "levelname": record.levelname,
                "msg": LogRingBuffer.format_message(record), "created": record.created, "msecs": record.msecs,
                "threadName": f"panel {record.threadName}",
            }))
        except Exception:
            self.handleError(record)


def _connect() -> Connection:
    """The app's connection: the key from stdin, the port on stdout."""
    authkey = bytes.fromhex(sys.stdin.readline().strip())
    listener = Listener((HOST, 0), authkey=authkey)
    port: int = listener.address[1]
    print(port, flush=True)
    if sys.stderr is not None:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())   # stdout's pipe carries the port alone
    watchdog = threading.Timer(CONNECT_TIMEOUT, os._exit, (1,))
    watchdog.daemon = True
    watchdog.start()
    conn = listener.accept()
    watchdog.cancel()
    listener.close()
    return conn


def main() -> None:
    conn = _connect()
    _, description, values, setup, entries = conn.recv()

    log_buffer = LogRingBuffer()
    for entry in entries:
        log_buffer.add(LogEntry(*entry))
    def on_loop(call: Callable[..., Any], *args: Any) -> None:
        assert core.loop is not None    # the receiver thread starts with NiceGUI's loop
        core.loop.call_soon_threadsafe(call, *args)

    link = AppLink(conn, log_buffer, on_loop=on_loop, on_end=nicegui_app.shutdown)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(LinkHandler(link))
    root_logger.setLevel(setup["log_level"])

    root = mirror.build_mirror(description)
    mirror.apply(root, values)
    link.watch(root)

    presets.set_app(setup["app"], setup["default_startup"])
    state = PanelState(presets.app_dir() / STATE_FILE)
    state.load()
    title: str = setup["title"]
    port: int = setup["port"]

    # The page's stylesheet and script; a short cache age, so an edit shows at the next reload.
    nicegui_app.add_static_files(STATIC_URL, STATIC_DIR, max_cache_age=10)

    @ui.page("/")
    def index(client: Client):
        started = time.perf_counter()
        if APP_QUERY in client.request.query_params:
            ui.add_body_html(f"<script>{SELF_CLOSE_SCRIPT}</script>")
        ui.dark_mode(True)   # the panel is dark; the client's preference is not asked
        ui.add_head_html('<style>* { transition-duration: 0s !important; animation-duration: 0s !important; }</style>')
        with ui.column().classes("w-full max-w-3xl mx-auto p-4"):
            controls = build_page(root, state, title=title, port=port, on_exit=link.exit, log_buffer=log_buffer,
                                  presets_store=link)
        # The page is built on the event loop, which holds every other client's poll and commit
        # meanwhile: the number says whether the build must get cheaper.
        logger.info("Panel built in %.0f ms, %d polled controls", (time.perf_counter() - started) * 1000, controls)

    nicegui_app.on_startup(lambda: threading.Thread(target=link.receive, daemon=True, name="receive").start())

    if setup["browser"]:
        threading.Thread(
            target=open_app_window, args=(port,),
            kwargs=dict(x=setup["browser_x"], y=setup["browser_y"],
                        width=setup["browser_width"], height=setup["browser_height"]),
            daemon=True, name="window",
        ).start()
    for ip in local_ips():
        logger.info("Panel: http://%s:%s", ip, port)
    logger.info("Panel: http://localhost:%s", port)

    ui.run(
        port=port,
        title=title,
        reload=False,
        show=False,
        dark=True,
        show_welcome_message=False,
        uvicorn_logging_level="warning",
        log_config=None,
    )
    link.close()
    conn.close()
    os._exit(0)     # NiceGUI's threads do not hold the process open
