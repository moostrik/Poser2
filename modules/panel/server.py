"""The panel server: NiceGUI on a background thread, serving the page built from the settings tree."""

import logging
import threading
import time
from typing import Callable

from nicegui import ui, app as nicegui_app, Client

from modules.log_config import get_log_buffer
from modules.settings import BaseSettings, Field

from .app_window import APP_QUERY, SELF_CLOSE_SCRIPT, open_app_window
from .page import build_page, local_ips

logger = logging.getLogger(__name__)


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
    """Runs the panel on a daemon thread; `start()` after the settings are initialized, `stop()` on exit."""

    def __init__(self, root: BaseSettings, settings: PanelSettings, on_exit: Callable[[], None] | None = None):
        self.root = root
        self.settings = settings
        self.on_exit = on_exit
        self._thread: threading.Thread | None = None
        self._page_registered = False

    def start(self) -> None:
        """Start the panel server in a daemon thread and, with `browser` set, open its window."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Panel server already running")
            return

        root = self.root
        title = self.settings.title
        port = self.settings.port
        on_exit = self.on_exit
        log_buffer = get_log_buffer()

        if not self._page_registered:
            self._page_registered = True

            @ui.page("/")
            def index(client: Client):
                started = time.perf_counter()
                if APP_QUERY in client.request.query_params:
                    ui.add_body_html(f"<script>{SELF_CLOSE_SCRIPT}</script>")
                ui.dark_mode(True)   # the panel is dark; the client's preference is not asked
                ui.add_head_html('<style>* { transition-duration: 0s !important; animation-duration: 0s !important; }</style>')
                with ui.column().classes("w-full max-w-3xl mx-auto p-4"):
                    controls = build_page(root, title=title, port=port, on_exit=on_exit, log_buffer=log_buffer)
                # The page is built on the event loop, which holds every other client's poll and
                # commit meanwhile: the number says whether the build must get cheaper.
                logger.info("Panel built in %.0f ms, %d polled controls", (time.perf_counter() - started) * 1000, controls)

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

        self._thread = threading.Thread(target=_run, daemon=True, name="panel")
        self._thread.start()

        if self.settings.browser:
            s = self.settings
            threading.Thread(
                target=open_app_window, args=(port,),
                kwargs=dict(x=s.browser_x, y=s.browser_y, width=s.browser_width, height=s.browser_height),
                daemon=True, name="panel-window",
            ).start()

        for ip in local_ips():
            logger.info("Panel: http://%s:%s", ip, port)
        logger.info("Panel: http://localhost:%s", port)

    def stop(self) -> None:
        """Shut down the panel server."""
        thread = self._thread
        try:
            nicegui_app.shutdown()
            logger.info("Panel server stopped")
        except Exception:
            logger.warning("Panel server shutdown failed", exc_info=True)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._thread = None
