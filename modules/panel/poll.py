"""The poll: the page's timer pushes every setting value that changed into the control that shows it.

All control updates run on NiceGUI's event loop, never on the thread that writes the setting: a
bind per write would flood the websocket and call NiceGUI from the GL thread.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.25    # seconds between ticks, per client

# One entry per polled control: (settings, field name, [last value seen], show). The last value is
# a one-element list so a control that commits a value can record it and not get it pushed back.
PollEntry = tuple[object, str, list, Callable[[object], None]]


def tick(polls: list[PollEntry], failed: set[int]) -> None:
    """Push every value that changed since the last tick into its control.

    A control that raises is logged at its first failure and skipped until it shows a value again;
    the other controls go on. *failed* holds the indexes of the controls that failed last.
    """
    for index, (settings, name, last, show) in enumerate(polls):
        try:
            value = getattr(settings, name)
            if value == last[0]:
                continue
            last[0] = value
            show(value)
        except Exception:
            if index not in failed:
                failed.add(index)
                logger.exception("Panel control %s.%s failed", type(settings).__name__, name)
            continue
        failed.discard(index)
