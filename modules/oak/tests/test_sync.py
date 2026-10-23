"""Tests for Sync: the bang reaches its callbacks in registration order."""

import unittest
from functools import partial

from modules.oak import Sync, SyncSettings


class _Recorder:
    """Appends its name to a shared list each time the bang arrives."""

    def __init__(self, name: int, calls: list[int]) -> None:
        self._name = name
        self._calls = calls

    def bang(self) -> None:
        self._calls.append(self._name)


class SyncCallbackTest(unittest.TestCase):
    def test_callbacks_run_in_registration_order(self) -> None:
        sync = Sync(SyncSettings())
        calls: list[int] = []
        recorders = [_Recorder(i, calls) for i in range(16)]
        for i, recorder in enumerate(recorders):
            sync.add_sync_callback(recorder.bang if i % 2 else partial(_Recorder.bang, recorder))
        sync.add_sync_callback(recorders[1].bang)                 # registered twice: called once, first position
        sync.submit_frame(0)                                      # one camera: its first frame bangs
        self.assertEqual(calls, list(range(16)))


if __name__ == '__main__':
    unittest.main()
