"""NiceGUI settings server — runs on a background thread alongside the main app."""

import importlib
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from nicegui import ui, app as nicegui_app

from .base_settings import BaseSettings
from .field import Field
from . import nice_panel as nice_panel_module

logger = logging.getLogger(__name__)


class NiceSettings(BaseSettings):
    """Configuration for the NiceGUI settings server."""
    title: Field[str] = Field("Settings", access=Field.INIT, visible=False)
    port: Field[int] = Field(666, access=Field.INIT, visible=False)


class NiceServer:
    """NiceGUI settings server that runs on a background thread."""

    def __init__(self, root: BaseSettings, settings: NiceSettings, on_exit: Optional[Callable[[], None]] = None):
        self.root = root
        self.settings = settings
        self.on_exit = on_exit
        self._thread: Optional[threading.Thread] = None
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

                    ui.timer(0.5, _reload_panel_if_changed)

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

        # Log connection URLs (skip loopback and link-local addresses)
        for ip in nice_panel_module._get_local_ips():
            logger.info("Settings UI: http://%s:%s", ip, port)
        logger.info("Settings UI: http://localhost:%s", port)

    def stop(self) -> None:
        """Shut down the NiceGUI settings server."""
        thread = self._thread
        try:
            nicegui_app.shutdown()
            logger.info("Settings server stopped")
        except Exception:
            logger.warning("Settings server shutdown failed", exc_info=True)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._thread = None
