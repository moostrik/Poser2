"""NiceGUI settings server — runs on a background thread alongside the main app."""

import logging
import threading
import webbrowser
from typing import Callable, Optional

from nicegui import ui, app as nicegui_app

from modules.settings.base_settings import BaseSettings
from modules.settings.setting import Setting
from modules.settings.panel import create_settings_panel
from modules.settings.registry import SettingsRegistry

logger = logging.getLogger(__name__)


class ServerSettings(BaseSettings):
    """Configuration for the NiceGUI settings server."""
    title: Setting[str] = Setting("Settings", init_only=True, visible=False)
    port: Setting[int] = Setting(666, init_only=True, visible=False)


class SettingsServer:
    """NiceGUI settings server that runs on a background thread."""

    def __init__(self, registry: SettingsRegistry, settings: ServerSettings, on_exit: Optional[Callable[[], None]] = None):
        self.registry = registry
        self.settings = settings
        self.on_exit = on_exit
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the NiceGUI settings UI in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Settings server already running")
            return

        registry = self.registry
        title = self.settings.title
        port = self.settings.port
        on_exit = self.on_exit

        @ui.page("/")
        def index():
            ui.dark_mode(True)
            ui.add_head_html('<style>* { transition-duration: 0s !important; animation-duration: 0s !important; }</style>')
            with ui.column().classes("w-full max-w-3xl mx-auto p-4"):
                create_settings_panel(registry, title=title, on_exit=on_exit)

        def _run():
            ui.run(
                port=port,
                title=title,
                reload=False,
                show=False,
                dark=True,
                uvicorn_logging_level="warning",
                log_config=None,
            )

        self._thread = threading.Thread(target=_run, daemon=True, name="settings-ui")
        self._thread.start()
        logger.info(f"Settings UI started on http://localhost:{port}")

        webbrowser.open(f"http://localhost:{port}")

    def stop(self) -> None:
        """Shut down the NiceGUI settings server."""
        try:
            nicegui_app.shutdown()
            logger.info("Settings server stopped")
        except Exception:
            logger.warning("Settings server shutdown failed", exc_info=True)
        self._thread = None
