"""What a depthai 2 message holds, as the backend hands it on: its moment of capture and its latency."""

import logging
import time
from collections.abc import Callable
from datetime import timedelta
from typing import cast

import depthai as dai

logger = logging.getLogger(__name__)

LATENCY_SMOOTHING: float = 0.1      # how hard each message pulls the latency reading

_future_capture_warned: bool = False     # a capture time after now is logged once per run
clock_now: Callable[[], timedelta] = cast(Callable[[], timedelta], dai.Clock.now)  # static; the stub says a method


def capture_time(message: dai.ImgFrame | dai.Tracklets) -> float:
    """The message's moment of capture on `time.perf_counter()`: now less the message's age on depthai's clock,
    which holds the device's capture time on the host's. A capture time after now cannot be, so such a message is
    stamped with its arrival instead."""
    global _future_capture_warned
    now: float = time.perf_counter()
    latency: float = (clock_now() - message.getTimestamp()).total_seconds()
    if latency >= 0.0:
        return now - latency
    if not _future_capture_warned:
        _future_capture_warned = True
        logger.warning(f'a message was captured {-latency * 1000.0:.1f} ms in the future by depthai\'s clock: '
                       f'stamped with its arrival (logged once)')
    return now


def smoothed_latency(previous: float, message: dai.ImgFrame | dai.Tracklets) -> float:
    """The latency reading in ms after this message, capture to arrival, on depthai's clock because the message's
    timestamp is. A reading for the panel; the frame's time is `capture_time`."""
    latency: float = (clock_now() - message.getTimestamp()).total_seconds() * 1000.0
    return latency if previous == 0.0 else previous + LATENCY_SMOOTHING * (latency - previous)
