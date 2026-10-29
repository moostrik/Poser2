"""The panel state: what the panel remembers between page loads, in the app's ``.ui_state.json``.

One state per app, shared by every client, as the file is: the open cards, the active tab and the
three visibility toggles per tab. The keys are the file's format: ``"<tab>.<card>…"`` for a card,
``"__active_tab__"`` for the tab and ``"<tab>.__hide_<kind>__"`` for a toggle.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = ".ui_state.json"
ACTIVE_TAB_KEY = "__active_tab__"


class PanelState:
    """The open cards, the active tab and the toggles, loaded from and saved to *path*."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, bool | str] = {}

    def load(self) -> None:
        """Read the file; a missing file is an empty state, an unreadable one a warning and an empty state."""
        self._data = {}
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            logger.warning("Panel state %s is unreadable; starting from an empty one", self.path, exc_info=True)
            return
        if isinstance(data, dict):
            self._data = data
        else:
            logger.warning("Panel state %s does not hold a dictionary; starting from an empty one", self.path)

    def save(self) -> None:
        """Write the file; a failed write is a warning, the state stays in memory."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data))
        except OSError:
            logger.warning("Panel state %s could not be saved", self.path, exc_info=True)

    # -- cards ---------------------------------------------------------------------

    @staticmethod
    def card_key(path: str, name: str) -> str:
        """The key of a card: its tab and the cards above it, dot-joined."""
        return f"{path}.{name}" if path else name

    def card_open(self, key: str) -> bool:
        """Whether a card is open; closed unless the state says open."""
        return self._data.get(key) is True

    def set_card_open(self, key: str, open: bool) -> None:
        self._data[key] = open
        self.save()

    # -- tabs ----------------------------------------------------------------------

    def active_tab(self, tabs: list[str]) -> str:
        """The tab to show: the saved one when it is among *tabs*, else the first."""
        saved = self._data.get(ACTIVE_TAB_KEY)
        return saved if saved in tabs else tabs[0]

    def set_active_tab(self, tab: str) -> None:
        self._data[ACTIVE_TAB_KEY] = tab
        self.save()

    # -- toggles -------------------------------------------------------------------

    @staticmethod
    def toggle_key(tab: str, toggle: str) -> str:
        return f"{tab}.{toggle}"

    def hidden(self, tab: str, toggle: str, default: bool) -> bool:
        """Whether a tab hides the fields of a toggle; *default* when the state does not say."""
        value = self._data.get(self.toggle_key(tab, toggle), default)
        return value if isinstance(value, bool) else default

    def set_hidden(self, tab: str, toggle: str, hidden: bool) -> None:
        self._data[self.toggle_key(tab, toggle)] = hidden
        self.save()
