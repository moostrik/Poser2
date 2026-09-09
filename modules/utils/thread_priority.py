"""OS thread priority for real-time threads (Windows only; a no-op elsewhere).

The ctypes prototypes are declared once here on purpose: calling
``kernel32.GetCurrentThread()`` without a declared ``restype`` returns the pseudo-handle
(-2) as a 32-bit int, which is truncated when passed back as a HANDLE, so
``SetThreadPriority`` silently fails with an invalid handle.
"""

import sys
from enum import IntEnum

import logging
logger = logging.getLogger(__name__)


class ThreadPriority(IntEnum):
    """Windows THREAD_PRIORITY_* levels (relative to the process priority class)."""
    NORMAL        = 0
    ABOVE_NORMAL  = 1
    HIGHEST       = 2
    TIME_CRITICAL = 15


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetCurrentThread.restype    = wintypes.HANDLE
    _kernel32.SetThreadPriority.argtypes  = [wintypes.HANDLE, ctypes.c_int]
    _kernel32.SetThreadPriority.restype   = wintypes.BOOL
    _kernel32.GetThreadPriority.argtypes  = [wintypes.HANDLE]
    _kernel32.GetThreadPriority.restype   = ctypes.c_int

    def set_current_thread_priority(level: ThreadPriority) -> bool:
        """Raise (or lower) the calling thread's scheduling priority. Returns success."""
        if not _kernel32.SetThreadPriority(_kernel32.GetCurrentThread(), int(level)):
            logger.warning("SetThreadPriority(%s) failed: error %d", level.name, ctypes.GetLastError())
            return False
        return True

    def get_current_thread_priority() -> int:
        """The calling thread's current THREAD_PRIORITY_* value."""
        return _kernel32.GetThreadPriority(_kernel32.GetCurrentThread())

else:
    def set_current_thread_priority(level: ThreadPriority) -> bool:
        return False

    def get_current_thread_priority() -> int:
        return int(ThreadPriority.NORMAL)
