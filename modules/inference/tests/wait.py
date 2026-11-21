"""What the package's tests wait with: a condition polled until it holds or its time is up."""

import time
from typing import Callable


def wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> bool:
    """Whether `condition` came to hold within `seconds`, polled every 2 ms."""
    end: float = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.002)
    return condition()
