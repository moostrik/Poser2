"""The panel's own window: an Edge app-mode window that opens on the panel and closes when the app ends."""

import logging
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

# Dedicated Edge profile: forces a browser process we own, and holds Edge's own record of the
# window's last position/size. Deleting the directory resets the window to the preset defaults.
# Lives under LOCALAPPDATA, not TEMP: temp cleaners prune half a profile, and its broken HTTP
# cache then serves truncated page resources (the "ES modules" banner, unstyled panels).
EDGE_PROFILE_DIR = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "Poser2" / "settings_edge"

# The script the page injects into the app window (?app=1) so the window closes itself when the
# server is gone; see the file for how it tells a dead server from a stalled one.
SELF_CLOSE_SCRIPT = (Path(__file__).parent / "self_close.js").read_text(encoding="utf-8")

APP_QUERY = "app"   # the query parameter that marks the app window's page


def _find_edge() -> Path | None:
    """Resolve the Edge executable from its standard install locations."""
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if base is None:
            continue
        path = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        if path.is_file():
            return path
    return None


def _wait_for_server(url: str, port: int) -> None:
    """Block until the server answers HTTP; a bare TCP accept races the app's heavy startup.

    HEAD is answered immediately (405) without building the page, so any HTTP response, status
    errors included, means the server is up."""
    probe = urllib.request.Request(url, method="HEAD")
    for _ in range(50):
        try:
            with urllib.request.urlopen(probe, timeout=1.0):
                return
        except urllib.error.HTTPError:
            return
        except OSError:
            time.sleep(0.1)
    logger.warning("Panel server not ready on port %s; opening the window anyway", port)


def open_app_window(port: int, *, x: int, y: int, width: int, height: int) -> None:
    """Open the panel in an Edge app-mode window once the server accepts connections.

    Falls back to the default browser when Edge is not there. *x*, *y*, *width* and *height* place
    the window on its first run only; afterwards Edge restores it where the user last left it."""
    # localhost, never 127.0.0.1: Edge keys the window's saved position by host, and dots in the
    # key are swallowed by Chromium's pref writer, so the position never restores.
    url = f"http://localhost:{port}"
    _wait_for_server(url, port)

    edge = _find_edge()
    if edge is None:
        logger.warning("Edge not found; opening the panel in the default browser")
        webbrowser.open(url)
        return

    # The ?app=1 tag makes the page inject the self-close script, so the window closes itself when
    # the server dies; remote tabs stay untagged. The Popen handle is useless for closing:
    # msedge.exe relaunches itself, so launch is fire-and-forget.
    args = [
        str(edge),
        f"--app={url}/?{APP_QUERY}=1",
        f"--user-data-dir={EDGE_PROFILE_DIR}",
        "--disable-http-cache",
        "--disable-background-mode",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    # Geometry flags only on the first run; afterwards Edge restores the window where the user
    # last left it (clamped onto a connected monitor), which the flags would override.
    if not EDGE_PROFILE_DIR.exists():
        args += [f"--window-position={x},{y}", f"--window-size={width},{height}"]
    try:
        subprocess.Popen(args)
        logger.info("Panel opened in an Edge app window")
    except OSError:
        logger.warning("Failed to launch Edge; opening the panel in the default browser", exc_info=True)
        webbrowser.open(url)
