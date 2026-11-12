"""The panel server: the panel in a process of its own, and the link that keeps it and the settings tree in step.

The panel process (``process.py``) runs NiceGUI on a mirror of the tree (``modules.settings.mirror``) and
never imports the app. This side starts it, applies what the page writes and sends what changes, on one
thread; nothing else of the panel runs in the app's process, so a page build never holds the render's GIL.

What the link carries, as tuples of plain values:

    app → panel   ("init", description, values, setup, log entries), ("values", changes),
                  ("log", entries), ("reply", request, ok), ("stop",)
    panel → app   ("set", path, name, value), ("fire", path, name), ("load" | "save", request, preset name),
                  ("exit",), ("log", record)
"""

import logging
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client, Connection
from typing import Any, Callable

from modules.log_config import get_log_buffer
from modules.settings import BaseSettings, Field, mirror, presets

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
LINK_INTERVAL = 0.1     # seconds between the sends of changed values and new log lines
RESTART_DELAY = 2.0     # seconds before a panel process that ended on its own starts again
STOP_TIMEOUT = 3.0      # seconds the panel process gets to end before it is killed


class PanelSettings(BaseSettings):
    """Configuration of the panel server and its window."""
    title: Field[str] = Field("Settings", access=Field.INIT, visible=False)
    port: Field[int] = Field(666, access=Field.INIT, visible=False)
    browser: Field[bool] = Field(False, access=Field.INIT, visible=False)
    browser_x: Field[int] = Field(0, access=Field.INIT, visible=False)
    browser_y: Field[int] = Field(0, access=Field.INIT, visible=False)
    browser_width: Field[int] = Field(900, access=Field.INIT, visible=False)
    browser_height: Field[int] = Field(1100, access=Field.INIT, visible=False)


class PanelServer:
    """Runs the panel in its own process; `start()` after the settings are initialized, `stop()` on exit.

    A panel process that ends while the app runs is logged and started again."""

    def __init__(self, root: BaseSettings, settings: PanelSettings, on_exit: Callable[[], None] | None = None):
        self.root = root
        self.settings = settings
        self.on_exit = on_exit
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._conn: Connection | None = None
        self._send_lock = threading.Lock()

    def start(self) -> None:
        """Start the link thread, which starts the panel process."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Panel server already running")
            return
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="panel-link")
        self._thread.start()

    def stop(self) -> None:
        """End the panel process and the link."""
        self._stopped.set()
        self._send(("stop",))
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Panel process did not end in %.0f s; killed", STOP_TIMEOUT)
                process.kill()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=STOP_TIMEOUT)
        self._thread = None
        logger.info("Panel server stopped")

    # -- the link thread -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stopped.is_set():
            code: int | None = None
            try:
                code = self._serve()
            except Exception:
                logger.exception("Panel link failed")
            if self._stopped.is_set():
                return
            logger.warning("Panel process ended (exit code %s); starting it again in %.0f s", code, RESTART_DELAY)
            self._stopped.wait(RESTART_DELAY)

    def _serve(self) -> int | None:
        """Start a panel process and keep it in step until it ends; its exit code."""
        authkey: bytes = secrets.token_bytes(32)
        # Its own process group: the console's Ctrl+C reaches the app alone, which then stops the panel.
        # Started through -c, not -m, so the module keeps its name and its log lines say where they come from.
        process = subprocess.Popen(
            [sys.executable, "-c", "from modules.panel.process import main; main()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        self._process = process
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(authkey.hex().encode() + b"\n")
            process.stdin.close()
            line: bytes = process.stdout.readline()     # the port it listens on; empty when it failed to start
            process.stdout.close()
            conn: Connection | None = Client((HOST, int(line)), authkey=authkey) if line.strip() else None
        except (OSError, ValueError):
            logger.warning("Panel process could not be reached", exc_info=True)
            conn = None
        if conn is not None:
            self._conn = conn
            try:
                self._link(conn)
            finally:
                self._conn = None
                conn.close()
        try:
            return process.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()

    def _link(self, conn: Connection) -> None:
        """Send the tree, then apply what comes and send what changes, until the connection ends."""
        buffer = get_log_buffer()
        cursor, entries = buffer.get_entries_since(0) if buffer is not None else (0, [])
        diff = mirror.ValueDiff(self.root)      # before the values, so a change in between is sent again, not lost
        s = self.settings
        setup: dict[str, Any] = {      # what the panel process starts with besides the tree
            "title": s.title, "port": s.port, "browser": s.browser,
            "browser_x": s.browser_x, "browser_y": s.browser_y,
            "browser_width": s.browser_width, "browser_height": s.browser_height,
            "app": presets.current_app(), "default_startup": presets.default_startup(),
            "log_level": logging.getLogger().getEffectiveLevel(),
        }
        self._send(("init", mirror.describe(self.root), mirror.values(self.root), setup, [tuple(e) for e in entries]))
        due: float = time.perf_counter()
        while not self._stopped.is_set():
            try:
                if conn.poll(max(0.0, due - time.perf_counter())):
                    self._handle(conn.recv(), diff)
            except (EOFError, OSError):
                return
            now: float = time.perf_counter()
            if now < due:
                continue
            due = now + LINK_INTERVAL
            changes = diff.changes()
            if changes:
                self._send(("values", changes))
            if buffer is not None:
                cursor, entries = buffer.get_entries_since(cursor)
                if entries:
                    self._send(("log", [tuple(e) for e in entries]))

    def _send(self, message: tuple) -> None:
        conn = self._conn
        if conn is None:
            return
        with self._send_lock:
            try:
                conn.send(message)
            except (OSError, ValueError):
                pass    # the panel process is gone; the link thread sees it and starts another

    def _handle(self, message: tuple, diff: mirror.ValueDiff) -> None:
        kind = message[0]
        if kind == "set":
            _, path, name, value = message
            try:
                setattr(mirror.find(self.root, path), name, value)
            except (KeyError, TypeError, ValueError, AttributeError):
                logger.warning("Panel: %r for %s.%s refused", value, path, name, exc_info=True)
            diff.written(path, name, value)
        elif kind == "fire":
            _, path, name = message
            try:
                settings = mirror.find(self.root, path)
                settings.fields[name].fire(settings)
            except KeyError:
                logger.warning("Panel: no action %s.%s", path, name)
        elif kind in ("load", "save"):
            _, request, name = message
            self._send(("reply", request, self._preset(kind, name)))
        elif kind == "exit":
            if self.on_exit is not None:
                self.on_exit()
        elif kind == "log":
            record = logging.makeLogRecord(message[1])
            logging.getLogger(record.name).handle(record)
        else:
            logger.warning("Panel: unknown message %r", kind)

    def _preset(self, kind: str, name: str) -> bool:
        """Load or save the preset *name* on the tree; whether it worked."""
        try:
            presets.validate_name(name)
            path = presets.path(name)
            if kind == "load":
                return presets.load(self.root, path)
            presets.save(self.root, path)
            return True
        except (OSError, ValueError):
            logger.warning("Panel: preset %s of '%s' failed", kind, name, exc_info=True)
            return False
