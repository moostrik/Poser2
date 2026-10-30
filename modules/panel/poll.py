"""The poll: the page's timer pushes every setting value that changed into the control that shows it.

All control updates run on NiceGUI's event loop, never on the thread that writes the setting: a
bind per write would flood the websocket and call NiceGUI from the GL thread.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.25    # seconds between ticks, per client


class Polled(Protocol):
    """What the tick needs of a control: its field, and a poll that shows the field's value when it changed."""
    name: str

    @property
    def settings(self) -> object: ...

    def poll(self) -> None: ...


def tick(controls: Sequence[Polled], failed: set[int]) -> None:
    """Poll every control.

    A control that raises is logged at its first failure and skipped until it polls without one;
    the other controls go on. *failed* holds the indexes of the controls that failed last.
    """
    for index, control in enumerate(controls):
        try:
            control.poll()
        except Exception:
            if index not in failed:
                failed.add(index)
                logger.exception("Panel control %s.%s failed", type(control.settings).__name__, control.name)
            continue
        failed.discard(index)
