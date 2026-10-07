"""NiceGUI settings server — runs on a background thread alongside the main app."""

import importlib
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional

from nicegui import ui, app as nicegui_app

from .base_settings import BaseSettings
from .field import Field
from .nice_util import SafeTimer
from . import nice_panel as nice_panel_module

logger = logging.getLogger(__name__)

# Dedicated Edge profile: forces a browser process we own, and holds Edge's own record of the
# app window's last position/size. Deleting the directory resets the window to the preset defaults.
_EDGE_PROFILE_DIR = Path(tempfile.gettempdir()) / "poser_settings_edge"


class NiceSettings(BaseSettings):
    """Configuration for the NiceGUI settings server."""
    title: Field[str] = Field("Settings", access=Field.INIT, visible=False)
    port: Field[int] = Field(666, access=Field.INIT, visible=False)
    browser: Field[bool] = Field(False, access=Field.INIT, visible=False)
    browser_x: Field[int] = Field(0, access=Field.INIT, visible=False)
    browser_y: Field[int] = Field(0, access=Field.INIT, visible=False)
    browser_width: Field[int] = Field(900, access=Field.INIT, visible=False)
    browser_height: Field[int] = Field(1100, access=Field.INIT, visible=False)


def _find_edge() -> Optional[Path]:
    """Resolve the Edge executable from its standard install locations."""
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if base is None:
            continue
        path = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        if path.is_file():
            return path
    return None


class NiceServer:
    """NiceGUI settings server that runs on a background thread."""

    def __init__(self, root: BaseSettings, settings: NiceSettings, on_exit: Optional[Callable[[], None]] = None):
        self.root = root
        self.settings = settings
        self.on_exit = on_exit
        self._thread: Optional[threading.Thread] = None
        self._browser_process: Optional[subprocess.Popen] = None
        self._page_registered = False
        self._panel_file = Path(nice_panel_module.__file__).resolve() if nice_panel_module.__file__ else None

    def start(self) -> None:
        """Start the NiceGUI settings UI in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Settings server already running")
            return

        root = self.root
        title = self.settings.title
        port = self.settings.port
        on_exit = self.on_exit

        if not self._page_registered:
            self._page_registered = True

            @ui.page("/")
            def index():
                panel_module = nice_panel_module
                ui.dark_mode(True)
                ui.add_head_html('<style>* { transition-duration: 0s !important; animation-duration: 0s !important; }</style>')

                panel_file = self._panel_file
                if panel_file is not None:
                    last_panel_mtime = {'value': panel_file.stat().st_mtime}

                    def _reload_panel_if_changed() -> None:
                        try:
                            current_mtime = panel_file.stat().st_mtime
                        except OSError:
                            return
                        if current_mtime <= last_panel_mtime['value']:
                            return
                        last_panel_mtime['value'] = current_mtime
                        try:
                            importlib.reload(panel_module)
                            ui.navigate.reload()
                        except Exception:
                            logger.warning('Settings UI reload failed', exc_info=True)

                    SafeTimer(0.5, _reload_panel_if_changed)

                with ui.column().classes("w-full max-w-3xl mx-auto p-4"):
                    panel_module.create_settings_panel(root, title=title, port=port, on_exit=on_exit)

        def _run():
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

        self._thread = threading.Thread(target=_run, daemon=True, name="settings-ui")
        self._thread.start()

        if self.settings.browser:
            threading.Thread(target=self._open_browser, daemon=True, name="settings-browser").start()

        # Log connection URLs (skip loopback and link-local addresses)
        for ip in nice_panel_module._get_local_ips():
            logger.info("Settings UI: http://%s:%s", ip, port)
        logger.info("Settings UI: http://localhost:%s", port)

    def _open_browser(self) -> None:
        """Open the settings UI in an Edge app-mode window once the server accepts connections."""
        port = self.settings.port
        url = f"http://localhost:{port}"

        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            logger.warning("Settings server not reachable on port %s; opening browser anyway", port)

        edge = _find_edge()
        if edge is None:
            logger.warning("Edge not found; opening settings UI in the default browser")
            webbrowser.open(url)
            return

        s = self.settings
        args = [
            str(edge),
            f"--app={url}",
            f"--user-data-dir={_EDGE_PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        # Geometry flags only on the first run; afterwards Edge restores the window where the
        # user last left it (clamped onto a connected monitor), which the flags would override.
        if not _EDGE_PROFILE_DIR.exists():
            args += [
                f"--window-position={s.browser_x},{s.browser_y}",
                f"--window-size={s.browser_width},{s.browser_height}",
            ]
        try:
            self._browser_process = subprocess.Popen(args)
            logger.info("Settings UI opened in Edge app window")
        except OSError:
            logger.warning("Failed to launch Edge; opening settings UI in the default browser", exc_info=True)
            webbrowser.open(url)

    def _close_browser(self) -> None:
        """Close the Edge app window if we opened one and it is still running."""
        process = self._browser_process
        self._browser_process = None
        if process is None or process.poll() is not None:
            return
        # Graceful close first (taskkill without /F posts WM_CLOSE), so Edge saves the window
        # placement and doesn't mark the profile as crashed. Hard-terminate only as fallback.
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid)], capture_output=True)
            process.wait(timeout=3)
            return
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("Settings browser window did not close gracefully, terminating")
        try:
            process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("Settings browser window did not close cleanly", exc_info=True)

    def stop(self) -> None:
        """Shut down the NiceGUI settings server."""
        self._close_browser()
        thread = self._thread
        try:
            nicegui_app.shutdown()
            logger.info("Settings server stopped")
        except Exception:
            logger.warning("Settings server shutdown failed", exc_info=True)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._thread = None
