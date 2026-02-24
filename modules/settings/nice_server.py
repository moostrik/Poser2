"""NiceGUI settings server — runs on a background thread alongside the main app."""

import logging
import threading
from typing import Callable, Optional

from nicegui import ui, app as nicegui_app

from modules.settings.settings import Settings
from modules.settings.field import Field
from modules.settings.nice_panel import create_settings_panel, _get_local_ips

logger = logging.getLogger(__name__)


class NiceSettings(Settings):
    """Configuration for the NiceGUI settings server."""
    title: Field[str] = Field("Settings", access=Field.INIT, visible=False)
    port: Field[int] = Field(666, access=Field.INIT, visible=False)


class NiceServer:
    """NiceGUI settings server that runs on a background thread."""

    def __init__(self, root: Settings, settings: NiceSettings, on_exit: Optional[Callable[[], None]] = None):
        self.root = root
        self.settings = settings
        self.on_exit = on_exit
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the NiceGUI settings UI in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Settings server already running")
            return

        root = self.root
        title = self.settings.title
        port = self.settings.port
        on_exit = self.on_exit

        @ui.page("/")
        def index():
            ui.dark_mode(True)
            ui.add_head_html('<style>* { transition-duration: 0s !important; animation-duration: 0s !important; }</style>')
            with ui.column().classes("w-full max-w-3xl mx-auto p-4"):
                create_settings_panel(root, title=title, port=port, on_exit=on_exit)

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
        import sys
        use_ansi = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        BLUE = "\033[94m" if use_ansi else ""
        RESET = "\033[0m" if use_ansi else ""
        for ip in _get_local_ips():
            print(f"Settings UI: {BLUE}http://{ip}:{port}{RESET}", flush=True)
        print(f"Settings UI: {BLUE}http://localhost:{port}{RESET}", flush=True)

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
