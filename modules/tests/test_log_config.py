"""Tests for the crash log: the exception hooks write it, every start reads it (a temp folder)."""

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from modules import log_config


class TestCrashLog(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / 'crashes.log'

    def test_no_crash_log_is_no_crashes(self):
        self.assertEqual(log_config.read_crashes(self.path), (0, None, None))
        self.assertEqual(log_config.crash_summary(self.path), 'No crashes recorded')

    def test_the_hooks_write_a_line_each(self):
        hooks = sys.excepthook, threading.excepthook
        self.addCleanup(lambda: setattr(sys, 'excepthook', hooks[0]))
        self.addCleanup(lambda: setattr(threading, 'excepthook', hooks[1]))
        with mock.patch.object(log_config, 'CRASH_LOG', self.path), mock.patch.object(log_config, '_app_name', 'test'):
            log_config.install_excepthooks()
            with self.assertLogs(level='CRITICAL'):
                threading.excepthook(threading.ExceptHookArgs([ValueError, ValueError('NULL pointer access\nmore'), None,
                                                               threading.current_thread()]))
                sys.excepthook(RuntimeError, RuntimeError('x'), None)
        count, since, last = log_config.read_crashes(self.path)
        lines = self.path.read_text(encoding='utf-8').splitlines()
        self.assertEqual(count, 2)
        self.assertEqual(since, lines[0][:10])
        self.assertTrue(lines[0].endswith(f' test {threading.current_thread().name} ValueError: NULL pointer access'))
        self.assertEqual(last, lines[1])
        self.assertTrue(last.endswith(' test main RuntimeError: x'))

    def test_the_summary_counts_since_the_first_line(self):
        self.path.write_text('2026-10-02 10:17:49 white_space Thread-2 RuntimeError: Failed to create GLFW window\n'
                             '2026-10-02 11:10:33 white_space Thread-7 (_run) ValueError: NULL pointer access\n'
                             '2026-10-03 08:00:00 hd_trio main RuntimeError: x\n', encoding='utf-8')
        summary = log_config.crash_summary(self.path)
        self.assertTrue(summary.startswith('3 crashes since 2026-10-02 ('))
        self.assertTrue(summary.endswith('), last 2026-10-03 08:00:00 hd_trio main RuntimeError: x'))


if __name__ == '__main__':
    unittest.main()
