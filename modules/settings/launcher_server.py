"""Launcher server — NiceGUI page on a separate port that starts/stops the app subprocess."""

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from nicegui import ui, app as nicegui_app

from . import nice_panel as nice_panel_module

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LauncherServer:
    """Dev tool: NiceGUI page on a separate port to start/stop the app subprocess.

    Serves two states on the same URL:
    - Idle: form to pick app + simulation flag, with a Start button.
    - Running: full-viewport iframe pointing to the app's own NiceGUI server.

    A per-client timer polls subprocess liveness and triggers ui.navigate.reload()
    when the state transitions, swapping the two views without changing the URL bar.
    """

    def __init__(self, port: int, app_registry: dict[str, type], app_port: int = 666) -> None:
        self._port = port
        self._app_port = app_port
        self._app_registry = app_registry
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._last_app_name: Optional[str] = None
        self._last_sim: bool = False
        self._last_exit_code: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._page_registered = False

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    def _is_alive(self) -> bool:
        """Return True if the app subprocess is running. Clears _proc on exit."""
        with self._lock:
            proc = self._proc
            if proc is None:
                return False
            code = proc.poll()
            if code is not None:
                self._last_exit_code = code
                self._proc = None
                logger.info("App subprocess exited with code %s", code)
                return False
            return True

    def launch(self, app_name: str, simulation: bool) -> None:
        """Spawn the app as a subprocess. No-op if already running."""
        if self._is_alive():
            logger.warning("App already running, ignoring launch request")
            return
        args = [sys.executable, str(_PROJECT_ROOT / "launcher.py"), "-app", app_name]
        if simulation:
            args.append("-sim")
        logger.info("Launching: %s", " ".join(args))
        with self._lock:
            self._proc = subprocess.Popen(args, cwd=str(_PROJECT_ROOT))
        self._last_app_name = app_name
        self._last_sim = simulation
        self._last_exit_code = None

    def _terminate_app(self) -> None:
        """Terminate the app subprocess (fallback path; prefer in-app Quit)."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        logger.info("Terminating app subprocess...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("App subprocess did not terminate, killing...")
            proc.kill()
            proc.wait()
        with self._lock:
            if self._proc is proc:
                self._last_exit_code = proc.poll()
                self._proc = None
        logger.info("App subprocess terminated")

    # ------------------------------------------------------------------
    # NiceGUI server
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the launcher NiceGUI server in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("LauncherServer already running")
            return

        port = self._port
        app_port = self._app_port

        if not self._page_registered:
            self._page_registered = True

            @ui.page("/")
            def index():
                ui.dark_mode(True)
                ui.add_head_html(
                    '<style>* { transition-duration: 0s !important; animation-duration: 0s !important; }</style>'
                )

                is_alive = self._is_alive()
                snapshot = [is_alive]

                def _check_state():
                    current = self._is_alive()
                    if current != snapshot[0]:
                        snapshot[0] = current
                        ui.navigate.reload()

                ui.timer(0.5, _check_state)

                if is_alive:
                    self._build_running(app_port)
                else:
                    self._build_idle()

        def _run():
            ui.run(
                port=port,
                title="Poser Launcher",
                reload=False,
                show=False,
                dark=True,
                show_welcome_message=False,
                uvicorn_logging_level="warning",
                log_config=None,
            )

        self._thread = threading.Thread(target=_run, daemon=True, name="launcher-ui")
        self._thread.start()

        for ip in nice_panel_module._get_local_ips():
            logger.info("Launcher UI: http://%s:%s", ip, port)
        logger.info("Launcher UI: http://localhost:%s", port)

    def stop(self) -> None:
        """Shut down the launcher server and any running app subprocess."""
        self._terminate_app()
        try:
            nicegui_app.shutdown()
            logger.info("Launcher server stopped")
        except Exception:
            logger.warning("Launcher server shutdown failed", exc_info=True)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._thread = None

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------

    def _build_idle(self) -> None:
        app_names = list(self._app_registry.keys())
        initial_app = self._last_app_name if self._last_app_name in app_names else app_names[0]
        initial_sim = self._last_sim

        with ui.column().classes("w-full h-screen items-center justify-center"):
            with ui.card().classes("w-80 gap-4 p-6"):
                ui.label("POSER LAUNCHER").classes("text-xl font-bold text-center w-full")

                app_select = ui.select(
                    options=app_names,
                    value=initial_app,
                    label="App",
                ).props("dense outlined").classes("w-full")

                sim_switch = ui.switch("Simulation", value=initial_sim)

                if self._last_exit_code is not None and self._last_exit_code != 0:
                    ui.label(f"Last run exited with code {self._last_exit_code}").classes(
                        "text-negative text-sm"
                    )

                def on_start():
                    self.launch(app_select.value or app_names[0], sim_switch.value)
                    ui.navigate.reload()

                ui.button("Start", on_click=on_start).props("color=primary").classes("w-full")

    def _build_running(self, app_port: int) -> None:
        # Hide all NiceGUI page chrome — the iframe is the entire viewport.
        ui.add_head_html(
            '<style>html, body { margin:0; padding:0; overflow:hidden; background:#000; }</style>'
        )

        # Inject the iframe via JavaScript so the src uses the browser's own hostname,
        # which makes LAN access from phones/tablets work without hardcoding an IP.
        # A polling fetch retries every second while the app's NiceGUI is still booting
        # (torch / onnx runtime imports take several seconds).
        ui.add_head_html(f'''<script>(function() {{
          var src = 'http://' + location.hostname + ':{app_port}/';
          var alive = false;
          function setup() {{
            var iframe = document.createElement('iframe');
            iframe.id = 'app-iframe';
            iframe.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;border:0;display:block';
            iframe.src = src;
            document.body.appendChild(iframe);
            var poll = setInterval(function() {{
              if (alive) {{ clearInterval(poll); return; }}
              fetch(src, {{mode: 'no-cors'}})
                .then(function() {{ alive = true; clearInterval(poll); }})
                .catch(function() {{
                  var f = document.getElementById('app-iframe');
                  if (f) {{ f.src = ''; setTimeout(function() {{ f.src = src; }}, 50); }}
                }});
            }}, 1000);
          }}
          if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', setup);
          }} else {{
            setup();
          }}
        }})();</script>''')

        # Stop App button — fallback for when the in-iframe Quit is not accessible.
        # Primary clean-shutdown path is the Quit button inside the iframe itself.
        with ui.element('div').style(
            'position:fixed;top:8px;right:8px;z-index:9999;pointer-events:auto'
        ):
            def on_stop():
                self._terminate_app()
                ui.navigate.reload()

            ui.button('■ Stop App', on_click=on_stop).props('dense color=negative size=sm')
