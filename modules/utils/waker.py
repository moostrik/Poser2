"""Waker: a wait for a `perf_counter` deadline, to about a millisecond, that `set()` ends at once."""

import ctypes
import sys
import threading
import time


if sys.platform == 'win32':
    from ctypes import wintypes

    _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x2
    _TIMER_ALL_ACCESS = 0x1F0003
    _INFINITE = 0xFFFFFFFF
    _WAIT_OBJECT_0 = 0x0
    _WAIT_FAILED = 0xFFFFFFFF

    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    _CreateWaitableTimerExW = _kernel32.CreateWaitableTimerExW
    _CreateWaitableTimerExW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD)
    _CreateWaitableTimerExW.restype = wintypes.HANDLE

    _CreateEventW = _kernel32.CreateEventW
    _CreateEventW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
    _CreateEventW.restype = wintypes.HANDLE

    _SetWaitableTimer = _kernel32.SetWaitableTimer
    _SetWaitableTimer.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.LARGE_INTEGER), wintypes.LONG,
                                  ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL)
    _SetWaitableTimer.restype = wintypes.BOOL

    _CancelWaitableTimer = _kernel32.CancelWaitableTimer
    _CancelWaitableTimer.argtypes = (wintypes.HANDLE,)
    _CancelWaitableTimer.restype = wintypes.BOOL

    _SetEvent = _kernel32.SetEvent
    _SetEvent.argtypes = (wintypes.HANDLE,)
    _SetEvent.restype = wintypes.BOOL

    _WaitForSingleObject = _kernel32.WaitForSingleObject
    _WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _WaitForSingleObject.restype = wintypes.DWORD

    _WaitForMultipleObjects = _kernel32.WaitForMultipleObjects
    _WaitForMultipleObjects.argtypes = (wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL,
                                        wintypes.DWORD)
    _WaitForMultipleObjects.restype = wintypes.DWORD

    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = (wintypes.HANDLE,)
    _CloseHandle.restype = wintypes.BOOL

    def _check(ok: object, call: str) -> None:
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error(), f'{call} failed')

    class Waker:
        """A wait for a `perf_counter` deadline that `set()` ends at once: a high-resolution waitable timer and an
        event, waited on together. A producer calls `set()` for a new item, and stop calls it after setting its stop
        flag. A `set()` stays until the next wait takes it, so a wake is never lost. One thread waits on a Waker;
        `close()` frees its handles once that thread has stopped, and a `set()` after it does nothing."""

        def __init__(self) -> None:
            timer = _CreateWaitableTimerExW(None, None, _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, _TIMER_ALL_ACCESS)
            _check(timer, 'CreateWaitableTimerExW')
            event = _CreateEventW(None, False, False, None)
            if not event:
                error = ctypes.get_last_error()
                _CloseHandle(timer)
                raise ctypes.WinError(error, 'CreateEventW failed')
            self._timer: int | None = timer
            self._event: int | None = event
            self._handles = (wintypes.HANDLE * 2)(event, timer)
            self._lock = threading.Lock()   # guards the event handle between set() and close()

        def set(self) -> None:
            """Wake the waiting thread, or the next wait if none waits."""
            with self._lock:
                if self._event is not None:
                    _check(_SetEvent(self._event), 'SetEvent')

        def wait(self, deadline: float | None) -> bool:
            """True when woken by `set()`, False at `deadline` (a `perf_counter` time); `None` waits for a wake only."""
            assert self._event is not None and self._timer is not None, 'Waker is closed'
            if deadline is None:
                result = _WaitForSingleObject(self._event, _INFINITE)
                _check(result != _WAIT_FAILED, 'WaitForSingleObject')
                return True
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return False
                due = wintypes.LARGE_INTEGER(-max(1, int(remaining * 1e7 + 0.5)))     # relative, in 100 ns
                _check(_SetWaitableTimer(self._timer, ctypes.byref(due), 0, None, None, False), 'SetWaitableTimer')
                result = _WaitForMultipleObjects(2, self._handles, False, _INFINITE)
                _check(result != _WAIT_FAILED, 'WaitForMultipleObjects')
                if result == _WAIT_OBJECT_0:
                    _CancelWaitableTimer(self._timer)
                    return True

        def close(self) -> None:
            """Free the timer and the event."""
            with self._lock:
                for handle in (self._timer, self._event):
                    if handle:
                        _CloseHandle(handle)
                self._timer = None
                self._event = None

else:
    class Waker:
        """A wait for a `perf_counter` deadline that `set()` ends at once: a `threading.Event`, whose timed wait is
        precise here. A producer calls `set()` for a new item, and stop calls it after setting its stop flag. A
        `set()` stays until the next wait takes it, so a wake is never lost. One thread waits on a Waker."""

        def __init__(self) -> None:
            self._event = threading.Event()

        def set(self) -> None:
            """Wake the waiting thread, or the next wait if none waits."""
            self._event.set()

        def wait(self, deadline: float | None) -> bool:
            """True when woken by `set()`, False at `deadline` (a `perf_counter` time); `None` waits for a wake only."""
            if deadline is None:
                self._event.wait()
                self._event.clear()
                return True
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return False
                if self._event.wait(remaining):
                    self._event.clear()
                    return True

        def close(self) -> None:
            """Nothing to free."""
