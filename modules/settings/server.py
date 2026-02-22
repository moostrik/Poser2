"""NiceGUI settings server — runs on a background thread alongside the main app."""

import logging
import threading

from nicegui import ui, app as nicegui_app

from modules.settings.panel import create_settings_panel

logger = logging.getLogger(__name__)

_server_thread = None
_server = None


def start(registry, port=666):
    """Start the NiceGUI settings UI on *port* in a daemon thread.

    Call once during app startup. The server shuts down automatically
    when the main process exits (daemon thread).
    """
    global _server_thread, _server

    if _server_thread is not None and _server_thread.is_alive():
        logger.warning("Settings server already running")
        return

    @ui.page("/")
    def index():
        ui.dark_mode(True)
        with ui.column().classes("w-full max-w-3xl mx-auto p-4"):
            ui.label("Settings").classes("text-2xl font-bold mb-4")
            create_settings_panel(registry)

    def _run():
        ui.run(
            port=port,
            title="Poser Settings",
            reload=False,
            show=False,
            dark=True,
            uvicorn_logging_level="warning",
            log_config=None,
        )

    _server_thread = threading.Thread(target=_run, daemon=True, name="settings-ui")
    _server_thread.start()
    logger.info(f"Settings UI started on http://localhost:{port}")

    import webbrowser
    webbrowser.open(f"http://localhost:{port}")


def stop():
    """Shut down the NiceGUI settings server."""
    global _server_thread
    try:
        nicegui_app.shutdown()
        logger.info("Settings server stopped")
    except Exception:
        logger.warning("Settings server shutdown failed", exc_info=True)
    _server_thread = None
