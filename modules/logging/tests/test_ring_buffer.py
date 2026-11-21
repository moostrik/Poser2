"""Tests for the log buffer (cursor, eviction, add) and the short-name formatter."""

import logging
import unittest

from modules.logging import LogEntry, LogRingBuffer
from modules.logging.setup import ShortNameFormatter


def _entry(i: int) -> LogEntry:
    return LogEntry(timestamp='00:00:00', level='INFO', source='test', message=str(i))


class TestLogRingBuffer(unittest.TestCase):
    def test_cursor_returns_only_new_entries(self):
        buffer = LogRingBuffer(maxlen=10)
        for i in range(3):
            buffer.add(_entry(i))
        cursor, entries = buffer.get_entries_since(0)
        self.assertEqual(cursor, 3)
        self.assertEqual([e.message for e in entries], ['0', '1', '2'])
        buffer.add(_entry(3))
        cursor, entries = buffer.get_entries_since(cursor)
        self.assertEqual(cursor, 4)
        self.assertEqual([e.message for e in entries], ['3'])
        self.assertEqual(buffer.get_entries_since(cursor), (4, []))
        self.assertEqual(buffer.cursor, 4)

    def test_evicted_entries_are_skipped(self):
        buffer = LogRingBuffer(maxlen=3)
        for i in range(5):
            buffer.add(_entry(i))
        cursor, entries = buffer.get_entries_since(1)
        self.assertEqual(cursor, 5)
        self.assertEqual([e.message for e in entries], ['2', '3', '4'])

    def test_a_record_becomes_an_entry(self):
        buffer = LogRingBuffer()
        logger = logging.getLogger('modules.camera.sync')
        logger.addHandler(buffer)
        self.addCleanup(logger.removeHandler, buffer)
        logger.warning('hello %s', 'there')
        _, entries = buffer.get_entries_since(0)
        self.assertEqual(len(entries), 1)
        self.assertEqual((entries[0].level, entries[0].source, entries[0].message), ('WARNING', 'sync', 'hello there'))

    def test_an_entry_is_a_tuple(self):
        entry = _entry(0)
        self.assertEqual(LogEntry(*tuple(entry)), entry)


class TestShortNameFormatter(unittest.TestCase):
    def test_the_last_segment_padded(self):
        record = logging.LogRecord('modules.camera.sync', logging.INFO, __file__, 1, 'msg', None, None)
        text = ShortNameFormatter('%(short_name)s %(message)s').format(record)
        self.assertEqual(text, 'sync............ msg')


if __name__ == '__main__':
    unittest.main()
